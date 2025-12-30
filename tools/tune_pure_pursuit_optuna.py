import os, sys
import optuna

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

from envs.race_env import RacingEnvBase
from agents.pure_pursuit.agent import PurePursuitAgent, PurePursuitConfig
from tools.runner import run_episode, RunConfig


def objective(trial: optuna.Trial) -> float:
  cfg = PurePursuitConfig(
    lookahead_m=trial.suggest_float("lookahead_m", 0.1, 1.0),
    steer_gain=trial.suggest_float("steer_gain", 0.2, 2.0),
    target_speed=trial.suggest_float("target_speed", 1.2, 5.0),
    speed_kp=trial.suggest_float("speed_kp", 0.1, 1.5),
  )
  env = RacingEnvBase()
  agent = PurePursuitAgent(cfg)
  m = run_episode(env, agent, RunConfig(max_steps=12000, max_laps=2, render="none", seed=0))
  env.close()
  if m["best_lap_time"] is None:
    return 1e6  # failed to complete a lap
  trial.set_user_attr("track_length", m.get("track_length"))
  trial.set_user_attr("env_spec", m.get("env_spec"))
  trial.set_user_attr("physics_id", m.get("physics_id"))
  return float(m["best_lap_time"])


def main():
  study = optuna.create_study(direction="minimize")
  study.optimize(objective, n_trials=50)
  print("best_value:", study.best_value)
  print("best_params:", study.best_params)


if __name__ == "__main__":
  main()
