import argparse
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

from envs.race_env import RacingEnvBase
from agents.mpc.agent import MPCAgent, MPCConfig

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--steps", type=int, default=8000)
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
  agent.reset(env, obs, info)

  # track length
  tv = env.track_view()
  print(f"track_length: {float(tv.L):.3f} m")

  # lap timing (prefer info["time"], fallback to env._t)
  lap_idx = int(info.get("lap", 0))
  lap_t0 = float(info.get("time", getattr(env, "_t", 0.0)))

  total_reward = 0.0
  t_wall0 = time.time()

  for k in range(args.steps):
    action = agent.act(env, obs, info)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += float(reward)

    if info.get("lap_complete", False):
      t_now = float(info.get("time", getattr(env, "_t", 0.0)))
      lap_time = t_now - lap_t0
      lap_idx = int(info.get("lap", lap_idx + 1))
      print(f"lap {lap_idx}: {lap_time:.3f} s (sim time)")
      lap_t0 = t_now

    if args.render != "none":
      env.render(mode=args.render)

    if args.sleep > 0:
      time.sleep(args.sleep)

    if terminated or truncated:
      obs, info = env.reset(seed=args.seed)
      agent.reset(env, obs, info)
      lap_idx = int(info.get("lap", 0))
      lap_t0 = float(info.get("time", getattr(env, "_t", 0.0)))

  dt_wall = time.time() - t_wall0
  print(f"done: steps={args.steps} total_reward={total_reward:.3f} wall_time={dt_wall:.2f}s")

  env.close()


if __name__ == "__main__":
  main()
