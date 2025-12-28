import json
import os
from typing import Optional
import time

import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces

from util.cubic_spline_path import CubicSplinePath
from util.sim_model import VehicleSimModel

def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

base_path = "./core/params/"
def load_param(file_name):
  with open(os.path.join(base_path, file_name + ".json")) as json_file:
    return json.load(json_file)

class TrackView:
  def __init__(self, path: CubicSplinePath, close: bool, width_left=0.0, width_right=0.0):
    self.path = path
    self.close = bool(close)
    self.L = float(path.length)
    self.width_left = width_left
    self.width_right = width_right

  def wrap_s(self, s):
    s = np.asarray(s)
    if self.close:
        return np.mod(s, self.L)
    # open track: clamp (or raise)
    return np.clip(s, 0.0, self.L)

  def sample_center(self, s_array):
    s = self.wrap_s(s_array)
    x = self.path.X(s)
    y = self.path.Y(s)
    yaw = self.path.calc_yaw(s)          # vectorized
    kappa = self.path.calc_curvature(s)  # vectorized

    tx = np.cos(yaw)
    ty = np.sin(yaw)
    nx = -ty
    ny = tx

    return {
        "s": s, "x": x, "y": y,
        "yaw": yaw, "kappa": kappa,
        "tx": tx, "ty": ty, "nx": nx, "ny": ny
    }

  def sample_boundaries(self, s_array):
    c = self.sample_center(s_array)
    x, y, nx, ny = c["x"], c["y"], c["nx"], c["ny"]

    # left positive along normal
    left = np.stack([x + self.width_left * nx, y + self.width_left * ny], axis=-1)
    right = np.stack([x - self.width_right * nx, y - self.width_right * ny], axis=-1)
    return left, right

  def project(self, x, y, s):
    # uses your optimizer; returns s* (scalar)
    ret = self.path.find_nearest_point(s, x, y)
    s_proj = float(ret[0][0])
    if self.close:
        s_proj = float(np.mod(s_proj, self.L))
    else:
        s_proj = float(np.clip(s_proj, 0.0, self.L))
    return s_proj

  def frenet_errors(self, x, y, yaw_vehicle, s):
    # nearest projection
    s_proj = self.project(x, y, s)

    x_ref = float(self.path.X(s_proj))
    y_ref = float(self.path.Y(s_proj))
    yaw_ref = float(self.path.calc_yaw(s_proj))
    kappa = float(self.path.calc_curvature(s_proj))

    # normal (left)
    nx = -np.sin(yaw_ref)
    ny =  np.cos(yaw_ref)

    # signed lateral error (meters)
    dx = x - x_ref
    dy = y - y_ref
    e_y = dx * nx + dy * ny  # positive = left of centerline

    # heading error
    e_psi = wrap_angle(yaw_vehicle - yaw_ref)

    return {
        "s": s_proj,
        "e_y": float(e_y),
        "e_psi": float(e_psi),
        "kappa": float(kappa),
        "yaw_ref": float(yaw_ref),
        "x_ref": x_ref,
        "y_ref": y_ref
    }

# Minimal, generic racing environment skeleton.
class RacingEnvBase(gym.Env):
  """Custom Environment that follows gymnasium interface"""
  metadata = {'render_modes': ['human', 'telemetry', 'rgb_array', 'matplotlib'],'render_fps': 25}

  def __init__(
    self,
    path: Optional[CubicSplinePath] = None,
    wL: float = 0.0,
    wR: float = 0.0,
    close: bool = True,

    dt: float = 0.05,
    enable_record: bool = True,
    streamer=None,
    render_backend: Optional[str] = None,
    max_episode_time: float = 420.0,
  ):
    super().__init__()

    if path is None:
      self.track = pd.read_json(os.path.join(base_path,"track.json"))
      self.track['S']=np.sqrt(self.track[['X', 'Y']].diff().pow(2).sum(axis=1)).cumsum()
      self.path = CubicSplinePath(self.track['X'], self.track['Y'], close=close)
    else:
      self.path = path

    self._track_view = TrackView(self.path, close=close, width_left=wL, width_right=wR)

    self.dt = float(dt)
    self.v = VehicleSimModel(scale=2, control_dt = dt, sim_dt= dt/10)

    # Simple 2d continuous action: [steer, throttle]
    self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

    # Observation: [x, y, vx, vy, progress]
    high = np.array([np.inf] * 5, dtype=np.float32)
    self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    # internal sim state
    self._rng = np.random.RandomState()
    self._state = None
    self._t = 0.0
    self._episode_start_time = None
    self._last_progress = 0.0
    self._last_s = 0.0
    self.max_episode_time = float(max_episode_time)

    # render & logger
    self._fig = None
    self.streamer = streamer
    self.render_backend = render_backend
    self.logbuf = []
    self.enable_record = bool(enable_record)

  # ---- Public API ----
  def track_view(self):
    return self._track_view

  def close(self):
    pass

  def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
    super().reset(seed=seed)

    ### Initial Vehicle state [X, Y, phi]
    phi0 = np.arctan2(self.path.dY(0), self.path.dX(0))
    self.v.x   = np.array([self.path.X(0), self.path.Y(0), phi0],
                    dtype=np.float64)
    ### Initial Vehicle state [vx, vy, r]
    self.v.dq  = np.array([1.0, 0., 0.],
                    dtype=np.float64)

    self._state = {
      "pos": np.array([self.v.x[0], self.v.x[1]], dtype=np.float32),
      "vel": np.array([self.v.dq[0], self.v.dq[1]], dtype=np.float32),
      "progress": 0.0,
      "lap": 0,
    }
    self._t = 0.0
    self._episode_start_time = time.time()
    self._last_progress = 0.0

    obs = self._make_obs(self._state)
    info = self._make_info(self._state)

    if self.enable_record:
      self._record_reset(obs, info)
    if self.streamer:
      self._stream_reset(obs, info)

    return obs, info

  def step(self, action):
    action = np.asarray(action, dtype=np.float32)
    action = np.clip(action, self.action_space.low, self.action_space.high)
    # use the VehicleSimModel self.v for dynamics
    throttle, steer = float(action[0]), float(action[1])
    u = np.array([throttle, steer], dtype=np.float64)

    self.v.sim_step(u)
    self._t += self.dt

    self._state["pos"] = np.array([self.v.x[0], self.v.x[1]], dtype=np.float32)
    self._state["vel"] = np.array([self.v.dq[0], self.v.dq[1]], dtype=np.float32)

    # compute derived quantities
    f = self._track_view.frenet_errors(self.v.x[0], self.v.x[1], self.v.x[2], self._last_s)
    self._state["progress"] = f["s"] / self._track_view.L
    self._state["e_y"] = f["e_y"]
    self._last_s = f["s"]

    reward, fail_event, terminated, truncated, lap_complete = self._make_reward_and_events(self._state, action)

    info = {
      "progress": float(self._state["progress"]),
      "track_error": float(self._state["e_y"]),
      "fail_event": fail_event,
      "lap_complete": lap_complete,
      "terminated": terminated,
      "truncated": truncated,
    }

    obs = self._make_obs(self._state)

    if self.enable_record:
      self._record_step(action, obs, float(reward), dict(info))
    if self.streamer:
      self._stream_step(action, obs, float(reward), dict(info))

    return obs, reward, terminated, truncated, info

  def render(self, mode="rgb_array", render_cfg=None):
    if mode == "rgb_array":
      frame = self._render_rgb(render_cfg)
      return frame

    if mode == "telemetry":
      return self._render_telemetry(render_cfg)

    if mode == "human":
      self._render_human(render_cfg)
      return None

    if mode == "matplotlib":
      return self._render_matplotlib(render_cfg)

    raise ValueError("unknown render mode")

  # ---- Internal helpers ----
  def _make_obs(self, state):
    pos = state["pos"]
    vel = state["vel"]
    progress = state.get("progress", 0.0)
    return np.array([pos[0], pos[1], vel[0], vel[1], progress], dtype=np.float32)

  def _make_info(self, state):
    return {
      "progress": float(state.get("progress", 0.0)),
      "lap": int(state.get("lap", 0)),
      "time": float(self._t),
    }

  def _make_reward_and_events(self, state, action=None):
    # reward is delta progress since last step, penalize going off track (big dist from path)
    progress = state["progress"]
    delta = progress - self._last_progress
    if self._track_view.close:
      delta = (delta + 1.0) % 1.0

    self._last_progress = progress
    reward = delta * 10.0  # scale

    fail_event = None
    lap_complete = False
    terminated = False
    truncated = False

    # Off-track detection: if path provided and distance is large
    if self.path is not None:
      min_d = float(np.abs(state.get("e_y", 0.0)))
      if min_d > 5.0:
        fail_event = "off_track"
        terminated = True
        reward -= 10.0

    # time limit
    if self._t >= self.max_episode_time:
      truncated = True

    # lap detection (progress wrapped around)
    if progress >= 0.999 and state.get("lap", 0) == 0:
      lap_complete = True
      state["lap"] = state.get("lap", 0) + 1
      reward += 100.0

    return float(reward), fail_event, bool(terminated), bool(truncated), bool(lap_complete)

  # ---- Recording / streaming ----
  def _record_reset(self, obs, info):
    self.logbuf.append({"type": "reset", "time": time.time(), "obs": obs.tolist(), "info": info})

  def _record_step(self, action, obs, reward, info):
    entry = {
      "type": "step",
      "time": time.time(),
      "action": np.asarray(action).tolist(),
      "obs": np.asarray(obs).tolist(),
      "reward": float(reward),
      "info": info,
    }
    self.logbuf.append(entry)

  def _stream_reset(self, obs, info):
    payload = {"event": "reset", "obs": obs.tolist(), "info": info}
    self._try_stream(payload)

  def _stream_step(self, action, obs, reward, info):
    payload = {"event": "step", "action": np.asarray(action).tolist(), "obs": obs.tolist(), "reward": float(reward), "info": info}
    self._try_stream(payload)

  def _try_stream(self, payload):
    if self.streamer is None:
      return
    try:
      # flexible calling convention: send_json / send / write
      if hasattr(self.streamer, "send_json"):
        self.streamer.send_json(payload)
      elif hasattr(self.streamer, "send"):
        self.streamer.send(payload)
      elif hasattr(self.streamer, "write"):
        self.streamer.write(payload)
    except Exception:
      # streaming is optional; swallow errors
      pass

  # ---- Rendering implementations (lightweight) ----
  def _render_rgb(self, render_cfg=None):
    # produce a small HxW x 3 uint8 array with a dot representing the agent
    H, W = 240, 320
    canvas = np.zeros((H, W, 3), dtype=np.uint8) + 30  # dark gray background
    pos = self._state["pos"]
    # map world coordinates to canvas: center at (0,0) maps to center
    scale = render_cfg.get("scale", 4.0) if isinstance(render_cfg, dict) else 4.0
    cx = int(W // 2 + pos[0] * scale)
    cy = int(H // 2 - pos[1] * scale)
    if 0 <= cx < W and 0 <= cy < H:
      rr = 4
      x0, x1 = max(0, cx - rr), min(W, cx + rr + 1)
      y0, y1 = max(0, cy - rr), min(H, cy + rr + 1)
      canvas[y0:y1, x0:x1, :] = np.array([255, 50, 50], dtype=np.uint8)
    return canvas

  def _render_telemetry(self, render_cfg=None):
    # lightweight data useful for telemetry
    return {
      "pos": self._state["pos"].tolist(),
      "vel": self._state["vel"].tolist(),
      "progress": float(self._state.get("progress", 0.0)),
      "time": float(self._t),
    }

  def _render_human(self, render_cfg=None):
    # Default: print simple telemetry to stdout (useful during local development)
    tel = self._render_telemetry(render_cfg)
    print(f"[RacingEnv] t={tel['time']:.2f} progress={tel['progress']:.3f} pos={tel['pos']} vel={tel['vel']}")

  def _render_matplotlib(self, render_cfg=None):
    import matplotlib.pyplot as plt

    if self._fig is None:
        plt.ion()
        self._fig, self._ax = plt.subplots()
        plt.show(block=False)
    ax = self._ax
    ax.cla()

    ax.plot(self.track['X'],  self.track['Y'], "r--" )
    ax.plot(self.track['X_i'],self.track['Y_i'], "k-")
    ax.plot(self.track['X_o'],self.track['Y_o'], "k-")

    carX, carY = self.v.shape.T
    ax.plot(carX,  carY, "b-")
    ax.plot(self.v.x[0], self.v.x[1], "x")
    ax.set_title("speed[m/s]:{:.2f}, deg:{:.2f}".format(self.v.dq[0], self.v.x[2]))
    #plt.pause(0.001)
    self._fig.canvas.draw()
    self._fig.canvas.flush_events()

if __name__ == "__main__":
  env = RacingEnvBase()

  obs, info = env.reset()
  for i in range(100):
    obs, reward, terminated, truncated, info = env.step([0.01, 0.0])
    env.render(mode="matplotlib")
    env.render(mode='human')

    if terminated or truncated:
      obs, info = env.reset()

