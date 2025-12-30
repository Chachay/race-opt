# race-opt
## Overview
A research framework for autonomous racing focused on optimal control and reinforcement learning.
Built upon and significantly extending ideas from alexliniger/gym-racecar, with modern Gymnasium support, reproducible benchmarking, and real-time telemetry(planned).

![animation](assets/demo.gif)

## Quick start

### Run Samples
```
python examples/run_baseline.py --render matplotlib
python examples/run_mpc.py --render matplotlib
python tools/tune_pure_pursuit_optuna.py
```
## Dependencies
This repository currently includes:
- Environment (envs/) and Pure Pursuit baseline agent (agents/pure_pursuit/)
- Minimal LQR-based MPC agent (agents/mpc_lqr/)
- Optional MPC solvers (CasADi) (agents/mpc_casadi/)
- MPCC is not included yet

### Minimal requirements
- Python 3.10+ (recommended)
- NumPy
- Gymnasium
- Matplotlib (only needed for --render matplotlib)

### Optional
- CasADi (for nonlinear MPC if/when enabled)
- Optuna (for a parameter tuning)

## Implemented agents

| # | Algorithm | Lap time | Note |
| -|-|-| - |
| 1 | Pure pursuite | 11.75 s | agents/pure_pursuit |
| 2 | Minimal MPC(LQR) | 14.15 s | agents/mpc_lqr |
| 3 | CasADi MPC | 11.05 s | agents/mpc_casadi |
| 4 | CasADi MPCC | 10.23 s | agents/mpcc |
| * | GP-augmented MPCC | 9.61 s | (L. Hewig, et al. 2020) with a noisy sensor environment |

## Lineage and Acknowledgements
race-opt is heavily inspired by the pioneering work of Alexander Liniger(@alexliniger) to the gym-racecar project.

In particular, this repository builds upon:
- The formulation of autonomous racing as an optimal control problem
- The use of spline-based centerline representations

While race-opt is a ground-up re-architecture and not a direct fork, the conceptual foundations established in gym-racecar are essential to this work.
