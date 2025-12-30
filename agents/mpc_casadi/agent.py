# agents/mpc/casadi_bicycle_agent.py
# CasADi nonlinear MPC (dynamic bicycle model w/ simple motor + Pacejka-like lateral tires)
#
# Assumptions:
# - env obs: [x, y, yaw, vx, progress]  (vx = longitudinal speed in body frame)
#   If your obs differs, adjust unpacking in act().
# - env provides track_view() with:
#   - L (track length)
#   - wrap_s(s)
#   - sample_center(s) -> dict keys ["x","y","yaw","kappa"] (kappa optional here)
# - env action: [throttle, steer] both in [-1, 1]
#
# Install:
#   pip install casadi
#
from dataclasses import dataclass
import math
from typing import Dict, Any, Optional, Tuple

import numpy as np

try:
  import casadi as ca
except Exception:
  ca = None


DEFAULT_PARAMS = {
  "Cm1": 0.287,
  "Cm2": 0.0545,
  "Cr0": 0.0518,
  "Cr2": 0.00035,
  "Br": 3.3852,
  "Cr": 1.2691,
  "Dr": 0.1737,
  "Bf": 2.579,
  "Cf": 1.2,
  "Df": 0.192,
  "m": 0.041,
  "Iz": 27.8e-6,
  "lf": 0.029,
  "lr": 0.033,
  "car_l": 0.06,
  "car_w": 0.03,
  "g": 9.81,
  "R_in": 0.14,
  "R_out": 0.14,
  "max_dist_proj": 0.2,
  "E_long": 0.9,
  "E_eps": 0.95,
  "maxAlpha": 0.6,
  "initial_velocity": 1.0,
  "s_trust_region": 0.2,
  "vx_zero": 0.3,
}


def wrap_to_pi(a: float) -> float:
  return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class BicycleMPCConfig:
  # horizon
  N: int = 20
  dt: float = 0.05

  # steering bounds (rad) and throttle bounds (normalized)
  delta_max: float = 0.6
  throttle_min: float = -1.0
  throttle_max: float = 1.0

  # reference speed (m/s)
  v_ref: float = 4.0

  # cost weights
  q_pos: float = 6.0    # position tracking (x,y)
  q_yaw: float = 2.0    # heading tracking
  q_vx: float = 0.5     # vx tracking
  r_throttle: float = 0.05
  r_delta: float = 0.10
  r_dthrottle: float = 0.20
  r_ddelta: float = 0.50

  # solver options
  ipopt_max_iter: int = 80
  ipopt_tol: float = 1e-3
  warm_start: bool = True

  # reference rollout
  lookahead_s: float = 0.0   # arc-length offset for ref start


class CasadiBicycleMPC:
  """
  Nonlinear MPC:
    state X = [x, y, psi, vx, vy, r]
    input U = [throttle, delta]

  Dynamics:
    - Longitudinal force: Fx = (Cm1 - Cm2*vx)*throttle - (Cr0 + Cr2*vx^2)*sign(vx)
    - Lateral tire forces (simple Pacejka):
      Fyf = Df*sin(Cf*atan(Bf*alpha_f))
      Fyr = Dr*sin(Cr*atan(Br*alpha_r))
    with alpha_f = atan2(vy+lf*r, max(|vx|,vx_zero)) - delta
       alpha_r = atan2(vy-lr*r, max(|vx|,vx_zero))
    alpha is softly clipped to +/- maxAlpha
  """
  def __init__(self, params: Dict[str, float], cfg: BicycleMPCConfig):
    if ca is None:
      raise RuntimeError("CasADi is not available. Install with: pip install casadi")
    self.p = dict(params)
    self.cfg = cfg

    self._solver = None
    self._lbz = None
    self._ubz = None
    self._lbg = None
    self._ubg = None
    self._P = None
    self._z = None

    self._nX = 6 * (cfg.N + 1)
    self._nU = 2 * cfg.N
    self._nz = self._nX + self._nU

    # warm-start storage (full decision vector)
    self._z_prev: Optional[np.ndarray] = None

    self._build_solver()

  def _build_solver(self) -> None:
    N = self.cfg.N
    dt = float(self.cfg.dt)

    # Symbols
    X = ca.SX.sym("X", 6, N + 1)   # [x,y,psi,vx,vy,r]
    U = ca.SX.sym("U", 2, N)     # [throttle, delta]

    # Parameters vector P:
    #   x0(6) + refs for k=0..N (x_ref,y_ref,psi_ref,vx_ref) => 4*(N+1)
    P = ca.SX.sym("P", 6 + 4 * (N + 1))

    # Unpack model params as constants
    m = float(self.p["m"])
    Iz = float(self.p["Iz"])
    lf = float(self.p["lf"])
    lr = float(self.p["lr"])
    Cm1 = float(self.p["Cm1"])
    Cm2 = float(self.p["Cm2"])
    Cr0 = float(self.p["Cr0"])
    Cr2 = float(self.p["Cr2"])
    Bf = float(self.p["Bf"])
    Cf = float(self.p["Cf"])
    Df = float(self.p["Df"])
    Br = float(self.p["Br"])
    Cr = float(self.p["Cr"])
    Dr = float(self.p["Dr"])
    maxAlpha = float(self.p["maxAlpha"])
    vx_zero = float(self.p["vx_zero"])

    def smooth_clip(x, lo, hi):
      # smooth-ish clip using tanh (keeps NLP nicer than hard fmin/fmax)
      mid = 0.5 * (lo + hi)
      half = 0.5 * (hi - lo)
      return mid + half * ca.tanh((x - mid) / max(1e-6, half))

    def tire_force_lat(alpha, B, C, D):
      return D * ca.sin(C * ca.atan(B * alpha))

    def f(x, u):
      # x: [x,y,psi,vx,vy,r], u: [throttle,delta]
      px, py, psi, vx, vy, rr = x[0], x[1], x[2], x[3], x[4], x[5]
      thr, delta = u[0], u[1]

      # avoid division near 0 speed
      vx_eff = ca.fmax(ca.fabs(vx), vx_zero)

      # slip angles
      alpha_f = ca.atan2(vy + lf * rr, vx_eff) - delta
      alpha_r = ca.atan2(vy - lr * rr, vx_eff)

      # clip slip angles (helps stability)
      alpha_f = smooth_clip(alpha_f, -maxAlpha, maxAlpha)
      alpha_r = smooth_clip(alpha_r, -maxAlpha, maxAlpha)

      # lateral forces
      Fyf = tire_force_lat(alpha_f, Bf, Cf, Df)
      Fyr = tire_force_lat(alpha_r, Br, Cr, Dr)

      # longitudinal force (rear drive + rolling resistance)
      # sign(vx) for resistance: use tanh smoothing
      sign_v = ca.tanh(20.0 * vx)
      Fx_drive = (Cm1 - Cm2 * vx_eff) * thr
      Fx_res = (Cr0 + Cr2 * vx_eff * vx_eff) * sign_v
      Fx = Fx_drive - Fx_res

      # dynamics in body frame
      vx_dot = (Fx - Fyf * ca.sin(delta) + m * vy * rr) / m
      vy_dot = (Fyr + Fyf * ca.cos(delta) - m * vx * rr) / m
      r_dot = (lf * Fyf * ca.cos(delta) - lr * Fyr) / Iz

      # world kinematics
      x_dot = vx * ca.cos(psi) - vy * ca.sin(psi)
      y_dot = vx * ca.sin(psi) + vy * ca.cos(psi)
      psi_dot = rr

      return ca.vertcat(x_dot, y_dot, psi_dot, vx_dot, vy_dot, r_dot)

    # Objective and constraints
    obj = 0
    g = []

    # initial condition constraint
    x0 = P[0:6]
    g.append(X[:, 0] - x0)

    # weights
    q_pos = float(self.cfg.q_pos)
    q_yaw = float(self.cfg.q_yaw)
    q_vx = float(self.cfg.q_vx)
    r_thr = float(self.cfg.r_throttle)
    r_del = float(self.cfg.r_delta)
    r_dthr = float(self.cfg.r_dthrottle)
    r_ddel = float(self.cfg.r_ddelta)

    def yaw_err(a):
      # smooth wrap using atan2(sin,cos)
      return ca.atan2(ca.sin(a), ca.cos(a))

    # stage cost + dynamics
    for k in range(N):
      xk = X[:, k]
      uk = U[:, k]
      xk1 = X[:, k + 1]

      # reference at k: [x_ref,y_ref,psi_ref,vx_ref]
      refk = P[6 + 4 * k : 6 + 4 * (k + 1)]
      xref, yref, psiref, vxref = refk[0], refk[1], refk[2], refk[3]

      # multiple shooting dynamics
      x_next = xk + dt * f(xk, uk)
      g.append(xk1 - x_next)

      ex = xk[0] - xref
      ey = xk[1] - yref
      epsi = yaw_err(xk[2] - psiref)
      evx = xk[3] - vxref

      obj += q_pos * (ex * ex + ey * ey) + q_yaw * (epsi * epsi) + q_vx * (evx * evx)

      # input cost
      obj += r_thr * (uk[0] * uk[0]) + r_del * (uk[1] * uk[1])

      # input smoothness
      if k > 0:
        duk = uk - U[:, k - 1]
        obj += r_dthr * (duk[0] * duk[0]) + r_ddel * (duk[1] * duk[1])

    # terminal cost at N
    refN = P[6 + 4 * N : 6 + 4 * (N + 1)]
    exN = X[0, N] - refN[0]
    eyN = X[1, N] - refN[1]
    epsiN = ca.atan2(ca.sin(X[2, N] - refN[2]), ca.cos(X[2, N] - refN[2]))
    evxN = X[3, N] - refN[3]
    obj += q_pos * (exN * exN + eyN * eyN) + q_yaw * (epsiN * epsiN) + q_vx * (evxN * evxN)

    # Flatten constraints
    g = ca.vertcat(*g)  # size 6 + 6N

    # Decision vector z
    z = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

    nlp = {"x": z, "f": obj, "g": g, "p": P}

    opts = {
      "ipopt.print_level": 0,
      "ipopt.sb": "yes",
      "print_time": 0,
      "ipopt.max_iter": int(self.cfg.ipopt_max_iter),
      "ipopt.tol": float(self.cfg.ipopt_tol),
    }
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    # Bounds for z
    lbz = np.full(self._nz, -np.inf, dtype=float)
    ubz = np.full(self._nz,  np.inf, dtype=float)

    # Input bounds
    nX = self._nX
    # U layout in z starts at nX, each step has [throttle, delta]
    for k in range(N):
      idx_thr = nX + 2 * k + 0
      idx_del = nX + 2 * k + 1
      lbz[idx_thr] = float(self.cfg.throttle_min)
      ubz[idx_thr] = float(self.cfg.throttle_max)
      lbz[idx_del] = -float(self.cfg.delta_max)
      ubz[idx_del] =  float(self.cfg.delta_max)

    # Optional state bounds: keep vx >= 0
    for k in range(N + 1):
      idx_vx = 6 * k + 3
      lbz[idx_vx] = 0.0

    # Equality constraints g == 0
    lbg = np.zeros(6 + 6 * N, dtype=float)
    ubg = np.zeros(6 + 6 * N, dtype=float)

    self._solver = solver
    self._lbz, self._ubz = lbz, ubz
    self._lbg, self._ubg = lbg, ubg
    self._P = P
    self._z = z

  def solve(
    self,
    *,
    x0: np.ndarray,           # shape (6,)
    ref: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],  # (x_ref,y_ref,psi_ref,vx_ref), each (N+1,)
  ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Returns:
      u0: np.array([throttle, delta])   (delta in rad)
      diag: solver diagnostics
    """
    x_ref, y_ref, psi_ref, vx_ref = ref
    N = self.cfg.N
    assert x_ref.shape[0] == N + 1

    # pack parameters P
    p = np.zeros(6 + 4 * (N + 1), dtype=float)
    p[0:6] = x0
    for k in range(N + 1):
      base = 6 + 4 * k
      p[base:base+4] = [x_ref[k], y_ref[k], psi_ref[k], vx_ref[k]]

    arg = {
      "p": p,
      "lbx": self._lbz,
      "ubx": self._ubz,
      "lbg": self._lbg,
      "ubg": self._ubg,
    }

    # Warm-start: shift previous decision vector forward by one step
    if self.cfg.warm_start and (self._z_prev is not None):
      arg["x0"] = self._z_prev

    sol = self._solver(**arg)
    stats = self._solver.stats()
    status = str(stats.get("return_status", "unknown"))
    iters = int(stats.get("iter_count", 0))
    ok = ("Solve_Succeeded" in status) or ("Solved" in status)

    if not ok:
      # fallback: do nothing steering, keep throttle 0
      return np.array([0.0, 0.0], dtype=float), {"ok": False, "status": status, "iters": iters}

    z = np.array(sol["x"]).reshape(-1)
    # cache warm-start vector (simple: reuse as-is; better: shift, but this is already helpful)
    self._z_prev = z.copy()

    # extract first control
    u0_thr = float(z[self._nX + 0])
    u0_del = float(z[self._nX + 1])
    return np.array([u0_thr, u0_del], dtype=float), {"ok": True, "status": status, "iters": iters}


class CasadiBicycleMPCAgent:
  """
  Agent wrapper:
    - builds reference from TrackView
    - estimates missing states (vy,r) as 0 unless provided in info/obs
    - returns env action [throttle, steer_norm] in [-1,1]
  """
  def __init__(self, params: Dict[str, float] = None, cfg: BicycleMPCConfig = None):
    if ca is None:
      raise RuntimeError("CasADi is not available. Install with: pip install casadi")
    self.params = dict(DEFAULT_PARAMS if params is None else params)
    self.cfg = BicycleMPCConfig() if cfg is None else cfg

    self._tv = None
    self._L = None
    self._mpc = CasadiBicycleMPC(self.params, self.cfg)

  def reset(self, env, obs=None, info=None):
    self._tv = env.track_view()
    self._L = float(self._tv.L)

  def _build_reference(self, progress: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    N = self.cfg.N
    dt = float(self.cfg.dt)
    v_ref = float(self.cfg.v_ref)

    s0 = float(progress) * self._L
    s0 = float(self._tv.wrap_s(s0 + self.cfg.lookahead_s))

    x_ref = np.zeros(N + 1, dtype=float)
    y_ref = np.zeros(N + 1, dtype=float)
    psi_ref = np.zeros(N + 1, dtype=float)
    vx_ref = np.full(N + 1, v_ref, dtype=float)

    for k in range(N + 1):
      sk = float(self._tv.wrap_s(s0 + v_ref * k * dt))
      c = self._tv.sample_center(sk)
      x_ref[k] = float(np.asarray(c["x"]))
      y_ref[k] = float(np.asarray(c["y"]))
      psi_ref[k] = float(np.asarray(c["yaw"]))

    return x_ref, y_ref, psi_ref, vx_ref

  def act(self, env, obs: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
    if self._tv is None:
      self.reset(env, obs, info)

    # obs: [x, y, yaw, vx, vy, r, progress]
    x, y, yaw, vx, vy, r, progress = map(float, obs)

    # Build reference
    ref = self._build_reference(progress)

    # Solve MPC for [throttle, delta(rad)]
    x0 = np.array([x, y, yaw, vx, vy, r], dtype=float)
    u0, diag = self._mpc.solve(x0=x0, ref=ref)

    throttle = float(np.clip(u0[0], self.cfg.throttle_min, self.cfg.throttle_max))
    delta = float(np.clip(u0[1], -self.cfg.delta_max, self.cfg.delta_max))

    # Normalize steering to [-1,1] for env action
    steer = float(np.clip(delta / self.cfg.delta_max, -1.0, 1.0))

    return np.array([throttle, steer], dtype=np.float32)

if __name__ == "__main__":
  import os
  import sys
  REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
  if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
  from envs.race_env import RacingEnvBase

  env = RacingEnvBase()
  agent = CasadiBicycleMPCAgent(DEFAULT_PARAMS, BicycleMPCConfig(N=20, dt=0.05, v_ref=4.0))
  obs, info = env.reset()
  agent.reset(env, obs, info)

  for _ in range(8000):
      action = agent.act(env, obs, info)
      obs, reward, terminated, truncated, info = env.step(action)
      env.render("matplotlib")
      if terminated or truncated:
          break
