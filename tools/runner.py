from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import time

import numpy as np

@dataclass
class RunConfig:
  max_steps: int = 8000
  max_laps: int = 1
  render: str = "none"    # "none"|"human"|"matplotlib"|...
  sleep: float = 0.0
  seed: int = 0
  reset_on_done: bool = False  # True: continue after terminated/truncated until max_steps

def _get_lap_time_stamp(info: Dict[str, Any], env: Any) -> float:
  """
  Prefer sub-step interpolated lap boundary time if available.
  Fallback to info["time"], then env._t, then wall-clock.
  """
  t = info.get("lap_cross_time", None)
  if t is not None:
    return float(t)
  t = info.get("time", None)
  if t is not None:
    return float(t)
  return float(getattr(env, "_t", time.time()))

def run_episode(env: Any, agent: Any, cfg: RunConfig) -> Dict[str, Any]:
  """
  Minimal runner loop.
  - env: gymnasium-like env with reset/step/render/close
  - agent: provides reset(env) and act(env, obs, info)->action
  Returns metrics dict suitable for logging / Optuna.
  """
  obs, info = env.reset(seed=cfg.seed)
  agent.reset(env)

  lap_times: List[float] = []
  lap_idx0 = int(info.get("lap", 0))
  last_lap_t = _get_lap_time_stamp(info, env)

  total_reward = 0.0
  steps = 0
  terminated = False
  truncated = False

  for _ in range(int(cfg.max_steps)):
    action = agent.act(env, np.asarray(obs), info)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += float(reward)
    steps += 1

    if info.get("lap_complete", False):
      t_now = _get_lap_time_stamp(info, env)
      lap_times.append(float(t_now - last_lap_t))
      last_lap_t = t_now

      # stop once we collected enough laps
      if len(lap_times) >= int(cfg.max_laps):
        break

    if cfg.render != "none":
      env.render(mode=cfg.render)
    if cfg.sleep and cfg.sleep > 0:
      time.sleep(cfg.sleep)

    if terminated or truncated:
      if not cfg.reset_on_done:
        break
      obs, info = env.reset(seed=cfg.seed)
      agent.reset(env)
      lap_idx0 = int(info.get("lap", 0))
      last_lap_t = _get_lap_time_stamp(info, env)

  best_lap_time = min(lap_times) if lap_times else None

  # Track length if available
  track_length = None
  try:
    tv = env.track_view()
    track_length = float(tv.L)
  except Exception:
    track_length = None

  # Env spec (optional)
  env_spec = info.get("env_spec", None)
  physics_id = None
  if isinstance(env_spec, dict):
    physics_id = env_spec.get("physics_id", None)

  return {
    "lap_times": lap_times,
    "best_lap_time": best_lap_time,
    "laps": len(lap_times),
    "steps": steps,
    "total_reward": total_reward,
    "terminated": bool(terminated),
    "truncated": bool(truncated),
    "lap_start_index": lap_idx0,
    "track_length": track_length,
    "env_spec": env_spec,
    "physics_id": physics_id,
  }
