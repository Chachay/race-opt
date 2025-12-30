"""
Pure Pursuit baseline example (single entrypoint).

Run:
  python examples/run_baseline.py --render matplotlib
  python examples/run_baseline.py --render human
  python examples/run_baseline.py --steps 5000 --lookahead 2.5
  python examples/run_baseline.py --steps 5000 --target-speed 2.0
"""
import argparse
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

from envs.race_env import RacingEnvBase
from agents.pure_pursuit.agent import PurePursuitAgent, PurePursuitConfig
from tools.runner import run_episode, RunConfig

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--steps", type=int, default=3000)
  ap.add_argument("--laps", type=int, default=1)
  ap.add_argument("--render", type=str, default="matplotlib",
                  choices=["none", "human", "matplotlib", "rgb_array", "telemetry"])
  ap.add_argument("--sleep", type=float, default=0.0, help="wall-clock sleep per step (sec)")
  ap.add_argument("--lookahead", type=float, default=0.433)
  ap.add_argument("--steer-gain", type=float, default=1.19)
  ap.add_argument("--target-speed", type=float, default=1.85)
  ap.add_argument("--speed-kp", type=float, default=0.86)
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()

  env = RacingEnvBase()
  agent = PurePursuitAgent(PurePursuitConfig(
    lookahead_m=float(args.lookahead),
    steer_gain=float(args.steer_gain),
    target_speed=float(args.target_speed),
    speed_kp=float(args.speed_kp),
  ))

  metrics = run_episode(env, agent, RunConfig(
    max_steps=int(args.steps),
    max_laps=int(args.laps),
    render=str(args.render),
    sleep=float(args.sleep),
    seed=int(args.seed),
    reset_on_done=False,
  ))

  if metrics.get("track_length") is not None:
    print(f"track_length: {metrics['track_length']:.3f} m")
  if metrics["lap_times"]:
    for i, lt in enumerate(metrics["lap_times"], start=1):
      print(f"lap {i}: {lt:.3f} s (sim time)")
    print(f"best_lap: {metrics['best_lap_time']:.3f} s")
  else:
    print("no lap completed")
  print(f"steps={metrics['steps']} total_reward={metrics['total_reward']:.3f}")

  env.close()

if __name__ == "__main__":
  main()
