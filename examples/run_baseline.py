"""
Pure Pursuit baseline example (single entrypoint).

Run:
  python examples/run_baseline.py --render matplotlib
  python examples/run_baseline.py --render human
  python examples/run_baseline.py --steps 5000 --lookahead 2.5
  python examples/run_baseline.py --steps 5000 --target-speed 2.0
"""
import argparse
import math
import time
from dataclasses import dataclass
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

from envs.race_env import RacingEnvBase
from agents.pure_pursuit.agent import PurePursuitAgent, PurePursuitConfig

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--steps", type=int, default=3000)
  ap.add_argument("--render", type=str, default="matplotlib",
                  choices=["none", "human", "matplotlib", "rgb_array", "telemetry"])
  ap.add_argument("--sleep", type=float, default=0.0, help="wall-clock sleep per step (sec)")
  ap.add_argument("--lookahead", type=float, default=0.2)
  ap.add_argument("--target-speed", type=float, default=1.5)
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()

  env = RacingEnvBase()
  obs, info = env.reset(seed=args.seed)

  cfg = PurePursuitConfig(
      lookahead_m=float(args.lookahead),
      target_speed=float(args.target_speed),
  )
  agent = PurePursuitAgent(cfg)
  agent.reset(env)

  # track length (meters)
  L = float(env._track_view.L) if getattr(env, "_track_view", None) is not None else float(agent._L)
  print(f"track_length: {L:.3f} m")

  t0 = time.time()
  total_reward = 0.0

  lap_start_t = float(getattr(env, "_t", 0.0))
  lap_count = 0

  for k in range(args.steps):
    action = agent.act(env, obs, info)
    obs, reward, terminated, truncated, info = env.step(action)

    # lap timing (sim time)
    if info.get("lap_complete", False):
      now_t = float(getattr(env, "_t", 0.0))
      lap_time = now_t - lap_start_t
      lap_count += 1
      print(f"lap {lap_count}: {lap_time:.3f} s (sim time)")
      lap_start_t = now_t

    total_reward += float(reward)

    if args.render != "none":
      env.render(mode=args.render)

    if args.sleep > 0:
      time.sleep(args.sleep)

    if terminated or truncated:
      obs, info = env.reset(seed=args.seed)
      agent.reset(env)

  dt = time.time() - t0
  print(f"done: steps={args.steps} total_reward={total_reward:.3f} wall_time={dt:.2f}s")

  env.close()

if __name__ == "__main__":
  main()
