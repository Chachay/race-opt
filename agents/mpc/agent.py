from dataclasses import dataclass
import math
import numpy as np

def wrap_to_pi(a: float) -> float:
  return (a + math.pi) % (2 * math.pi) - math.pi

@dataclass
class MPCConfig:
  horizon: int = 15      # prediction steps
  dt: float = 0.05       # seconds (fallback; can be overridden from env)
  wheelbase: float = 0.062    # meters
  delta_max: float = 0.60    # rad (physical steering limit)
  v_ref: float = 1.5       # m/s
  lookahead_s: float = 0.0   # optional arc-length lookahead for reference start

  # cost weights
  q_ey: float = 6.0
  q_epsi: float = 2.0
  r_u: float = 0.2
  r_du: float = 2.0

  # speed control (separate from lateral MPC)
  speed_kp: float = 0.6
  throttle_min: float = -1.0
  throttle_max: float = 1.0


class MPCAgent:
  """
  Minimal lateral MPC (linearized error dynamics) + simple speed P controller.

  - Uses TrackView only: env.track_view(), sample_center(s)
  - Does NOT access env.path or env.v internals.
  - obs expected: [x, y, yaw, v, progress]
  - action output: [throttle, steer] in [-1, 1]
  """

  def __init__(self, cfg: MPCConfig):
    self.cfg = cfg
    self._tv = None
    self._L = None
    self._dt = cfg.dt
    self._u_prev = 0.0  # previous steering error command (rad)

  def reset(self, env, obs=None, info=None):
    self._tv = env.track_view()
    self._L = float(self._tv.L)

    # try to read dt from env if available
    self._dt = float(getattr(env, "dt", getattr(env, "_dt", self.cfg.dt)))
    self._u_prev = 0.0

  def act(self, env, obs: np.ndarray, info: dict) -> np.ndarray:
    x, y, yaw, v, _, _, progress = map(float, obs)
    if self._tv is None:
      self.reset(env)

    # reference rollout (centerline) along s
    s0 = float(progress) * self._L
    s0 = float(s0 + self.cfg.lookahead_s)

    # build per-step feedforward steering from curvature: delta_ff ~ atan(L * kappa)
    kappa = np.zeros(self.cfg.horizon, dtype=float)
    delta_ff = np.zeros(self.cfg.horizon, dtype=float)

    # use v_ref in rollout; keep it constant for minimal example
    v_ref = float(self.cfg.v_ref)

    for k in range(self.cfg.horizon):
      sk = float(s0 + v_ref * k * self._dt)
      c = self._tv.sample_center(sk)
      kk = float(np.asarray(c["kappa"]))
      kappa[k] = kk
      delta_ff[k] = math.atan(self.cfg.wheelbase * kk)

    # compute current tracking errors w.r.t. centerline at s0
    c0 = self._tv.sample_center(s0)
    x_ref0 = float(np.asarray(c0["x"]))
    y_ref0 = float(np.asarray(c0["y"]))
    yaw_ref0 = float(np.asarray(c0["yaw"]))

    # lateral error sign using normal to ref heading
    dx = x - x_ref0
    dy = y - y_ref0
    # normal pointing left of ref tangent
    nx = -math.sin(yaw_ref0)
    ny =  math.cos(yaw_ref0)
    e_y = dx * nx + dy * ny
    e_psi = wrap_to_pi(yaw - yaw_ref0)

    # solve finite-horizon LQR on augmented state [e_y, e_psi, u_prev]
    # control is u = delta_err (rad), and we penalize du = u - u_prev
    u0 = self._solve_receding_horizon_lqr(e_y, e_psi, v, delta_ff[0])

    # compose actual steering: delta = delta_ff + delta_err
    delta = float(delta_ff[0] + u0)
    delta = float(np.clip(delta, -self.cfg.delta_max, self.cfg.delta_max))

    # update previous u (delta_err) for rate penalty
    self._u_prev = float(u0)

    # normalize steer to [-1, 1]
    steer = float(np.clip(delta / self.cfg.delta_max, -1.0, 1.0))

    # simple speed control -> throttle
    throttle = self.cfg.speed_kp * (self.cfg.v_ref - v)
    throttle = float(np.clip(throttle, self.cfg.throttle_min, self.cfg.throttle_max))

    return np.array([throttle, steer], dtype=np.float32)

  def _solve_receding_horizon_lqr(self, e_y: float, e_psi: float, v: float, delta_ff0: float) -> float:
    """
    Discrete-time linearized error dynamics around centerline with feedforward steering:
      e_y[k+1]   = e_y[k]   + v*dt * e_psi[k]
      e_psi[k+1] = e_psi[k] + v*dt/L * u[k]    where u = delta_err
    Augment with u_prev to penalize rate:
      x = [e_y, e_psi, u_prev]
      u = delta_err
      du = u - u_prev
    Cost:
      sum x'Qx + u'Ru + du'Rdu du
    We implement the du penalty by expanding stage cost in terms of x and u.
    """
    dt = self._dt
    Lw = self.cfg.wheelbase
    v_eff = max(0.5, float(abs(v)))  # avoid singular at low speed

    # system matrices for augmented state
    A = np.array([
        [1.0, v_eff * dt, 0.0],
        [0.0, 1.0,    0.0],
        [0.0, 0.0,    1.0],
    ], dtype=float)

    B = np.array([
        [0.0],
        [v_eff * dt / Lw],
        [1.0],
    ], dtype=float)

    # stage cost:
    # xQx + uRu + (u - u_prev) Rdu (u - u_prev)
    # with u_prev being x[2]
    q_ey = self.cfg.q_ey
    q_epsi = self.cfg.q_epsi
    r_u = self.cfg.r_u
    r_du = self.cfg.r_du

    Q = np.diag([q_ey, q_epsi, 0.0])

    # Expand (u - x2)^2 * r_du = r_du * (u^2 - 2u x2 + x2^2)
    # => add r_du to R, add r_du to Q[2,2], and add cross term N between x and u.
    R = np.array([[r_u + r_du]], dtype=float)
    Q_aug = Q.copy()
    Q_aug[2, 2] = Q_aug[2, 2] + r_du

    # cross term: 2 * x' N u in standard form
    # from -2*r_du*u*x2 => N[2,0] = -r_du
    N = np.zeros((3, 1), dtype=float)
    N[2, 0] = -r_du

    # finite-horizon Riccati backward pass (time-invariant here)
    P = Q_aug.copy()
    # terminal cost = Q_aug
    for _ in range(self.cfg.horizon - 1, -1, -1):
      # compute K = (R + B'PB)^-1 (B'PA + N')
      S = R + (B.T @ P @ B)
      # ensure invertible
      S_inv = np.linalg.inv(S)
      K = S_inv @ (B.T @ P @ A + N.T)
      P = Q_aug + (A.T @ P @ A) - (A.T @ P @ B + N) @ K

    # initial augmented state
    x0 = np.array([e_y, e_psi, self._u_prev], dtype=float).reshape(3, 1)

    # control law u = -K x
    u0 = float(-(K @ x0)[0, 0])

    # clamp delta_err so total delta stays within limit (keep some room for delta_ff)
    # delta = delta_ff0 + u0 within [-delta_max, delta_max]
    u_min = -self.cfg.delta_max - delta_ff0
    u_max =  self.cfg.delta_max - delta_ff0
    u0 = float(np.clip(u0, u_min, u_max))
    return u0
