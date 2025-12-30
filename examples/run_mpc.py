import argparse
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

from envs.race_env import RacingEnvBase
from agents.mpc.agent import MPCAgent, MPCConfig
from tools.runner import run_episode, RunConfig

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--steps", type=int, default=3000)
  ap.add_argument("--laps", type=int, default=1)
  ap.add_argument("--render", type=str, default="matplotlib",
                  choices=["none", "human", "matplotlib", "rgb_array", "telemetry"])
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--vref", type=float, default=2.0)
  ap.add_argument("--horizon", type=int, default=15)
  ap.add_argument("--sleep", type=float, default=0.0)
  args = ap.parse_args()

  env = RacingEnvBase()
  obs, info = env.reset(seed=args.seed)

  cfg = MPCConfig(
    v_ref=float(args.vref),
    horizon=int(args.horizon)
  )
  agent = MPCAgent(cfg)
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
