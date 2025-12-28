from dataclasses import dataclass
import math
import numpy as np

def _wrap_to_pi(a: float) -> float:
  """wrap angle to [-pi, pi]"""
  a = (a + math.pi) % (2 * math.pi) - math.pi
  return a

@dataclass
class PurePursuitConfig:
  lookahead_m: float = 0.2          # lookahead distance in meters along arclength s
  steer_gain: float = 0.5           # proportional gain on heading error
  max_steer: float = 1.0            # action space expects [-1, 1]
  target_speed: float = 1.5         # m/s (rough)
  speed_kp: float = 0.9             # throttle P gain
  max_throttle: float = 1.0         # action space expects [-1, 1]
  min_throttle: float = -1.0

class PurePursuitAgent:
  def __init__(self, cfg: PurePursuitConfig):
    self.cfg = cfg

  def reset(self, env):
    tv = env.track_view() if hasattr(env, "track_view") else getattr(env, "_track_view", None)
    if tv is None:
      raise RuntimeError("env must provide track_view() or _track_view")
    self._tv = tv
    self._L = float(tv.L)

  def act(self, env, obs: np.ndarray, info: dict) -> np.ndarray:
    x, y, yaw, vx, vy, yaw_rate, progress = map(float, obs)

    # progress -> s
    s_now = float(progress) * self._L
    s_tgt = float(self._tv.wrap_s(s_now + self.cfg.lookahead_m))
    if getattr(env, "_track_view", None) is not None and getattr(env._track_view, "close", False):
        s_tgt = s_tgt % self._L
    else:
        s_tgt = min(s_tgt, self._L)

    # target point on centerline via TrackView API
    c = self._tv.sample_center(s_tgt)
    xt = float(np.asarray(c["x"]))
    yt = float(np.asarray(c["y"]))

    # heading error to target
    dx = xt - x
    dy = yt - y
    target_heading = math.atan2(dy, dx)
    alpha = _wrap_to_pi(target_heading - yaw)

    # steer: simple proportional on alpha (normalized to [-1, 1])
    steer = self.cfg.steer_gain * alpha
    steer = float(np.clip(steer, -self.cfg.max_steer, self.cfg.max_steer))

    # speed control: use longitudinal speed estimate from (vx, vy) in obs
    speed = math.hypot(vx, vy)
    throttle = self.cfg.speed_kp * (self.cfg.target_speed - speed)
    throttle = float(np.clip(throttle, self.cfg.min_throttle, self.cfg.max_throttle))

    return np.array([throttle, steer], dtype=np.float32)
