
"""
CasADi MPCC (Model Predictive Contouring Control) using the SAME vehicle model as sim_model.py.

obs assumption: [x, y, yaw, vx, vy, rr, progress]
action: [throttle, steer_norm] in [-1, 1]

Key differences vs tracking MPC:
- optimize progress 's' explicitly, with control 'sdot'
- objective uses contouring/lag errors based on track tangent/normal
- trust-region on s to stabilize nonconvexity (use your params["s_trust_region"])
"""

from dataclasses import dataclass
import json
import math
import os
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np

try:
  import casadi as ca
except Exception:
  ca = None

base_path = Path(__file__).resolve().parent.parent.parent / "core" / "params"
def load_param(file_name):
  with open(os.path.join(base_path, file_name + ".json")) as json_file:
    return json.load(json_file)

def wrap_to_pi(a: float) -> float:
  return (a + math.pi) % (2.0 * math.pi) - math.pi

@dataclass
class MPCCConfig:
  N: int = 20
  dt: float = 0.05

  # bounds
  delta_max: float = 0.6
  throttle_min: float = -1.0
  throttle_max: float = 1.0
  sdot_min: float = -3.0
  sdot_max: float = 6.0

  # weights
  w_contour: float = 60.0
  w_lag: float = 6.0
  w_epsi: float = 2.0
  w_vx: float = 0.4
  w_u: float = 0.05
  w_du: float = 1.0
  w_progress: float = 6.0   # maximize progress => minimize(-w_progress*sdot)

  # reference speed shaping (optional)
  vx_ref: float = 5.0
  a_lat_max: float = 3.0     # curvature-based cap

  # solver
  ipopt_max_iter: int = 20
  ipopt_tol: float = 1e-3
  warm_start: bool = True

class CasadiMPCC:
  """
  Multiple shooting MPCC:
    State  X_k = [x, y, psi, vx, vy, r, s]
    Input  U_k = [thr, delta, sdot]
  """
  def __init__(self, params: Dict[str, float], cfg: MPCCConfig, track_view):
    if ca is None:
      raise RuntimeError("CasADi is not available. Install with: pip install casadi")
    self.p = load_param("model") if params is None else dict(params)
    self.cfg = cfg
    self.tv = track_view
    self.L = float(track_view.L)

    self._solver = None
    self._lbz = None
    self._ubz = None
    self._lbg = None
    self._ubg = None

    self._nX = 7 * (cfg.N + 1)
    self._nU = 3 * cfg.N
    self._nz = self._nX + self._nU

    self._z_prev: Optional[np.ndarray] = None
    self._s_center: Optional[np.ndarray] = None  # trust-region center for s states

    self._build_solver()

  @staticmethod
  def _smooth_sign(x, k=20.0):
    return ca.tanh(k * x)

  @staticmethod
  def _smooth_clip(x, lo, hi):
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    return mid + half * ca.tanh((x - mid) / max(1e-6, half))

  @staticmethod
  def _tire_lat(alpha, B, C, D):
    return D * ca.sin(C * ca.atan(B * alpha))

  @staticmethod
  def _yaw_err(a):
    return ca.atan2(ca.sin(a), ca.cos(a))

  def _build_solver(self) -> None:
    N = self.cfg.N
    dt = float(self.cfg.dt)

    # decision vars
    X = ca.SX.sym("X", 7, N + 1)  # x,y,psi,vx,vy,r,s
    U = ca.SX.sym("U", 3, N)      # thr,delta,sdot

    # parameters:
    # x0(7) + track ref per stage: xr,yr,psir,tx,ty,nx,ny,vxr  => 8*(N+1)
    P = ca.SX.sym("P", 7 + 8 * (N + 1))

    # model params (match sim_model.py)
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

    def f(x, u):
      # x: [x,y,psi,vx,vy,r,s], u:[thr,delta,sdot]
      px, py, psi, vx, vy, rr, ss = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
      thr, delta, sdot = u[0], u[1], u[2]

      # avoid div0, but keep sign for dynamics elsewhere
      vx_eff = ca.fmax(ca.fabs(vx), vx_zero)

      # slip angles (same structure as sim_model.py)  :contentReference[oaicite:3]{index=3}
      alpha_f = ca.atan2(vy + lf * rr, vx_eff) - delta
      alpha_r = ca.atan2(vy - lr * rr, vx_eff)
      alpha_f = self._smooth_clip(alpha_f, -maxAlpha, maxAlpha)
      alpha_r = self._smooth_clip(alpha_r, -maxAlpha, maxAlpha)

      # lateral forces with MINUS sign (matches sim_model.py) :contentReference[oaicite:4]{index=4}
      Ffy = -self._tire_lat(alpha_f, Bf, Cf, Df)
      Fry = -self._tire_lat(alpha_r, Br, Cr, Dr)

      # longitudinal force (matches sim_model.py):  :contentReference[oaicite:5]{index=5}
      # Frx = Cm1*u - Cm2*u*vx - Cr0*sign(vx) - Cr2*vx^2
      sign_v = self._smooth_sign(vx)
      Frx = (Cm1 * thr) - (Cm2 * thr * vx_eff) - (Cr0 * sign_v) - (Cr2 * vx_eff * vx_eff)

      # body dynamics (matches sim_model.py) :contentReference[oaicite:6]{index=6}
      vx_dot = (vy * rr) + (Frx - Ffy * ca.sin(delta)) / m
      vy_dot = (-vx * rr) + (Fry + Ffy * ca.cos(delta)) / m
      r_dot  = (Ffy * lf * ca.cos(delta) - Fry * lr) / Iz

      # world kinematics
      x_dot = vx * ca.cos(psi) - vy * ca.sin(psi)
      y_dot = vx * ca.sin(psi) + vy * ca.cos(psi)
      psi_dot = rr

      # progress
      s_dot = sdot

      return ca.vertcat(x_dot, y_dot, psi_dot, vx_dot, vy_dot, r_dot, s_dot)

    # constraints and objective
    g = []
    obj = 0

    # initial state constraint
    x0 = P[0:7]
    g.append(X[:, 0] - x0)

    wC = float(self.cfg.w_contour)
    wL = float(self.cfg.w_lag)
    wE = float(self.cfg.w_epsi)
    wV = float(self.cfg.w_vx)
    wU = float(self.cfg.w_u)
    wDU = float(self.cfg.w_du)
    wP = float(self.cfg.w_progress)

    for k in range(N):
      xk = X[:, k]
      uk = U[:, k]
      xk1 = X[:, k + 1]

      # ref from P: xr,yr,psir,tx,ty,nx,ny,vxr
      base = 7 + 8 * k
      xr = P[base + 0]; yr = P[base + 1]; psir = P[base + 2]
      tx = P[base + 3]; ty = P[base + 4]
      nx = P[base + 5]; ny = P[base + 6]
      vxr = P[base + 7]

      # dynamics
      g.append(xk1 - (xk + dt * f(xk, uk)))

      # contouring / lag error
      dx = xk[0] - xr
      dy = xk[1] - yr
      eC = dx * nx + dy * ny
      eL = dx * tx + dy * ty
      epsi = self._yaw_err(xk[2] - psir)
      evx = xk[3] - vxr

      obj += wC * (eC * eC) + wL * (eL * eL) + wE * (epsi * epsi) + wV * (evx * evx)

      # input cost
      obj += wU * (uk[0] * uk[0] + uk[1] * uk[1])

      # progress reward: maximize sdot
      obj += -wP * uk[2]

      # smoothness
      if k > 0:
        duk = uk - U[:, k - 1]
        obj += wDU * (duk[0]*duk[0] + duk[1]*duk[1] + 0.2*duk[2]*duk[2])

    # terminal cost
    baseN = 7 + 8 * N
    xrN = P[baseN + 0]; yrN = P[baseN + 1]; psirN = P[baseN + 2]
    txN = P[baseN + 3]; tyN = P[baseN + 4]
    nxN = P[baseN + 5]; nyN = P[baseN + 6]
    vxrN = P[baseN + 7]

    dxN = X[0, N] - xrN
    dyN = X[1, N] - yrN
    eCN = dxN * nxN + dyN * nyN
    eLN = dxN * txN + dyN * tyN
    epsiN = self._yaw_err(X[2, N] - psirN)
    evxN = X[3, N] - vxrN
    obj += wC*(eCN*eCN) + wL*(eLN*eLN) + wE*(epsiN*epsiN) + wV*(evxN*evxN)

    g = ca.vertcat(*g)  # size: 7 + 7N

    z = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
    nlp = {"x": z, "f": obj, "g": g, "p": P}

    opts = {
      "ipopt.print_level": 0,
      "ipopt.sb": "yes",
      "print_time": 0,
      "ipopt.max_iter": int(self.cfg.ipopt_max_iter),
      "ipopt.tol": float(self.cfg.ipopt_tol),
    }
    self._solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    # bounds
    lbz = np.full(self._nz, -np.inf, dtype=float)
    ubz = np.full(self._nz,  np.inf, dtype=float)

    # vx >= 0
    for k in range(N + 1):
      lbz[7 * k + 3] = 0.0

    # input bounds
    nX = self._nX
    for k in range(N):
      idx_thr = nX + 3 * k + 0
      idx_del = nX + 3 * k + 1
      idx_sdot = nX + 3 * k + 2
      lbz[idx_thr] = float(self.cfg.throttle_min)
      ubz[idx_thr] = float(self.cfg.throttle_max)
      lbz[idx_del] = -float(self.cfg.delta_max)
      ubz[idx_del] =  float(self.cfg.delta_max)
      lbz[idx_sdot] = float(self.cfg.sdot_min)
      ubz[idx_sdot] = float(self.cfg.sdot_max)

    lbg = np.zeros(7 + 7 * N, dtype=float)
    ubg = np.zeros(7 + 7 * N, dtype=float)

    self._lbz, self._ubz = lbz, ubz
    self._lbg, self._ubg = lbg, ubg

  def _vx_ref_from_curvature(self, kappa: float) -> float:
    # optional curvature-based speed cap
    v = float(self.cfg.vx_ref)
    kk = abs(float(kappa))
    if kk > 1e-9:
      v = min(v, math.sqrt(float(self.cfg.a_lat_max) / kk))
    return v

  def _pack_P(self, x0: np.ndarray, s_seq: np.ndarray) -> np.ndarray:
    N = self.cfg.N
    P = np.zeros(7 + 8 * (N + 1), dtype=float)
    P[0:7] = x0

    # sample_center returns tx,ty,nx,ny in your env :contentReference[oaicite:7]{index=7}
    c = self.tv.sample_center(s_seq)
    xr = np.asarray(c["x"], dtype=float).reshape(-1)
    yr = np.asarray(c["y"], dtype=float).reshape(-1)
    psir = np.asarray(c["yaw"], dtype=float).reshape(-1)
    tx = np.asarray(c["tx"], dtype=float).reshape(-1)
    ty = np.asarray(c["ty"], dtype=float).reshape(-1)
    nx = np.asarray(c["nx"], dtype=float).reshape(-1)
    ny = np.asarray(c["ny"], dtype=float).reshape(-1)
    kappa = np.asarray(c["kappa"], dtype=float).reshape(-1)

    vxr = np.array([self._vx_ref_from_curvature(kappa[i]) for i in range(N + 1)], dtype=float)

    for k in range(N + 1):
      base = 7 + 8 * k
      P[base + 0] = xr[k]
      P[base + 1] = yr[k]
      P[base + 2] = psir[k]
      P[base + 3] = tx[k]
      P[base + 4] = ty[k]
      P[base + 5] = nx[k]
      P[base + 6] = ny[k]
      P[base + 7] = vxr[k]

    return P

  def solve(self, x0: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    x0: [x,y,psi,vx,vy,r,s]  (s in meters)
    returns u0: [thr, delta(rad), sdot]
    """
    N = self.cfg.N
    dt = float(self.cfg.dt)
    tr = float(self.p.get("s_trust_region", 0.2))

    s0 = float(x0[6])

    if self._s_center is None:
      # init guess: forward using current vx
      v0 = max(float(x0[3]), float(self.p["vx_zero"]))
      s_seq = np.array([s0 + v0 * k * dt for k in range(N + 1)], dtype=float)
    else:
      # shift previous center
      s_seq = self._s_center.copy()
      s_seq[:-1] = s_seq[1:]
      s_seq[-1] = s_seq[-2] + max(float(x0[3]), float(self.p["vx_zero"])) * dt

    # wrap for sampling
    s_seq = np.asarray(self.tv.wrap_s(s_seq), dtype=float)
    P = self._pack_P(x0, s_seq)

    # trust region bounds for s states
    lbz = self._lbz.copy()
    ubz = self._ubz.copy()
    for k in range(N + 1):
      idx_s = 7 * k + 6
      lbz[idx_s] = s_seq[k] - tr
      ubz[idx_s] = s_seq[k] + tr

    arg = {"p": P, "lbx": lbz, "ubx": ubz, "lbg": self._lbg, "ubg": self._ubg}
    if self.cfg.warm_start and (self._z_prev is not None):
      arg["x0"] = self._z_prev

    sol = self._solver(**arg)
    stats = self._solver.stats()
    status = str(stats.get("return_status", "unknown"))
    iters = int(stats.get("iter_count", 0))
    ok = ("Solve_Succeeded" in status) or ("Solved" in status)
    if not ok:
      return np.array([0.0, 0.0, 0.0], dtype=float), {"ok": False, "status": status, "iters": iters}

    z = np.array(sol["x"]).reshape(-1)
    self._z_prev = z.copy()

    # update center using predicted s from solution
    s_pred = np.array([z[7 * k + 6] for k in range(N + 1)], dtype=float)
    self._s_center = np.asarray(self.tv.wrap_s(s_pred), dtype=float)

    # first control
    nX = self._nX
    thr0 = float(z[nX + 0])
    del0 = float(z[nX + 1])
    sdot0 = float(z[nX + 2])
    return np.array([thr0, del0, sdot0], dtype=float), {"ok": True, "status": status, "iters": iters, "s_pred": s_pred}

class CasadiMPCCAgent:
  def __init__(self, params: Dict[str, float] = None, cfg: MPCCConfig = None):
    if ca is None:
      raise RuntimeError("CasADi is not available. Install with: pip install casadi")
    self.params = load_param("model") if params is None else dict(params)
    self.cfg = MPCCConfig() if cfg is None else cfg
    self._tv = None
    self._L = None
    self._mpcc: Optional[CasadiMPCC] = None

  def reset(self, env, obs=None, info=None):
    self._tv = env.track_view()
    self._L = float(self._tv.L)
    self._mpcc = CasadiMPCC(self.params, self.cfg, self._tv)

  def act(self, env, obs: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
    if self._mpcc is None:
      self.reset(env, obs, info)

    # obs: [x, y, yaw, vx, vy, rr, progress]  :contentReference[oaicite:8]{index=8}
    x, y, yaw, vx, vy, rr, progress = map(float, obs)

    # initial s from progress
    s0 = float(progress) * self._L
    s0 = float(self._tv.wrap_s(s0))

    x0 = np.array([x, y, yaw, vx, vy, rr, s0], dtype=float)
    u0, diag = self._mpcc.solve(x0)

    thr = float(np.clip(u0[0], self.cfg.throttle_min, self.cfg.throttle_max))
    delta = float(np.clip(u0[1], -self.cfg.delta_max, self.cfg.delta_max))

    steer = float(np.clip(delta / self.cfg.delta_max, -1.0, 1.0))
    return np.array([thr, steer], dtype=np.float32)

if __name__ == "__main__":
  # quick smoke run (same style as your agent.py main)
  import sys
  REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
  if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
  from envs.race_env import RacingEnvBase
  from tools.runner import run_episode, RunConfig

  env = RacingEnvBase()
  agent = CasadiMPCCAgent(None, MPCCConfig(N=20, dt=0.05))

  metrics = run_episode(env, agent, RunConfig(
    max_steps=3000,
    max_laps=3,
    render="matplotlib",
    seed=0,
    reset_on_done=False,
  ))
  print("best_lap_time:", metrics.get("best_lap_time"))
  env.close()
