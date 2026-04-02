# Inzozi Smart Glasses — RL Navigation Agent

> **Reinforcement Learning for Assistive Navigation in Kigali, Rwanda**
>
> An AI agent learns to guide a blind user safely through Kigali streets
> using smart glasses. Trained with DQN, PPO, A2C, and REINFORCE.

---

## Quick Start (3 commands)

```bash
git clone https://github.com/YOUR_USERNAME/Stecie_Niyonzima_rl_summative.git
cd Stecie_Niyonzima_rl_summative
pip install -r requirements.txt
python main.py
```

That's it. The best trained A2C model runs immediately with full Pygame visualization.

---

## Project Structure

```
Stecie_Niyonzima_rl_summative/
├── environment/
│   ├── custom_env.py        Custom Gymnasium environment (Kigali streets)
│   └── rendering.py         Pygame visualization — Kigali scene, HUD, sensors
├── training/
│   ├── dqn_training.py      DQN — 10 hyperparameter experiments
│   └── pg_training.py       PPO + A2C + REINFORCE — 10 experiments each
├── models/
│   ├── dqn/                 Saved DQN models (dqn_run_1.zip ... dqn_run_10.zip)
│   └── pg/
│       ├── ppo/             Saved PPO models
│       ├── a2c/             Saved A2C models
│       └── reinforce/       Saved REINFORCE models (.pt files)
├── data/
│   ├── at_data.csv          UN Assistive Technology dataset (Kaggle 2021)
│   └── at_dataset.py        Parses Rwanda AT coverage → weight 0.8275
├── results/                 Training curve plots + CSV hyperparameter tables
├── main.py                  Entry point — run best model with full GUI
├── requirements.txt         All dependencies
└── README.md
```

---

## Running the Simulation

## 🎥 Video Demonstration

[![Inzozi Smart Glasses RL Demo](https://img.youtube.com/vi/UNTYb9CW2g4/0.jpg)](https://youtu.be/UNTYb9CW2g4)

**Watch the full demonstration:**  
[https://youtu.be/UNTYb9CW2g4](https://youtu.be/UNTYb9CW2g4)

*The video shows both A2C and PPO agents navigating the Kigali environment with live terminal output and Pygame GUI. Camera on, full screen, problem statement, reward explanation, and performance commentary included.*

### Run best trained model (A2C — highest reward 466.9)
```bash
python main.py
```

### Run a specific algorithm
```bash
python main.py --algo ppo         # PPO best run (406.8)
python main.py --algo dqn         # DQN best run
python main.py --algo reinforce   # REINFORCE best run (191.2)
```

### Run random agent demo (no model needed — good for first demo)
```bash
python environment/rendering.py
```

### Run with more episodes
```bash
python main.py --algo a2c --episodes 10
```

---

## Training the Models

**Train DQN** (10 hyperparameter runs, ~20 min):
```bash
python training/dqn_training.py
```

**Train PPO + A2C + REINFORCE** (30 runs total, ~60 min):
```bash
python training/pg_training.py
```

Results are saved automatically to:
- `results/dqn_results.csv` — DQN hyperparameter table
- `results/ppo_results.csv` — PPO hyperparameter table
- `results/a2c_results.csv` — A2C hyperparameter table
- `results/reinforce_results.csv` — REINFORCE hyperparameter table
- `results/dqn_reward_curves.png` — DQN reward plots
- `results/ppo_reward_curves.png` — PPO reward plots
- `results/a2c_reward_curves.png` — A2C reward plots
- `results/reinforce_reward_curves.png` — REINFORCE reward plots

---

## Environment Details

| Property | Value |
|---|---|
| World size | 20 × 20 metres (one Kigali street block) |
| Action space | Discrete(10) — turn, move, warn, broadcast, assist, recenter |
| Observation space | Box(13,) — raycasts, density, slope, goal vector, AT weight |
| Max steps per episode | 500 |
| Obstacles | 6 motos + 10 pedestrians + 7 static kiosks |
| AT weight (Rwanda) | 0.8275 (UN Kaggle dataset — Partial coverage) |

**Terminal conditions:**
- Goal reached within 0.8m → +10 reward
- Collision with obstacle → -10 reward
- Edge fall → -15 reward
- Too many human assist requests → episode ends
- 500 steps without goal → -8 reward (timeout)

---

## Algorithm Results Summary

| Algorithm | Best Run | Mean Reward | Key Hyperparameters |
|---|---|---|---|
| A2C | Run 3 | **466.9** | lr=1e-3, n_steps=10, entropy=0.02 |
| PPO | Run 7 | 406.8 | lr=3e-4, clip=0.4, gae=0.80 |
| DQN | Run 5 | ~447.0 | lr=5e-4, gamma=0.95, buf=20000 |
| REINFORCE | Run 1 | 191.2 | lr=1e-3, baseline=True |

---

## Dataset

The **UN Assistive Technology dataset** (Kaggle 2021) is used to derive
Rwanda's AT coverage weight:

- Rwanda status: **Partial coverage**
- Vision training: **Yes**
- Mobility training: **Yes**
- Computed AT weight: **0.8275**

This weight is embedded in observation index [9] and modulates reward
penalties — areas with coverage < 0.7 apply a 1.2× penalty multiplier,
simulating harder navigation in low-support zones.

Dataset source:
https://www.kaggle.com/datasets/masterexecutioner/un-data-assistive-technology

---

## Requirements

- Python 3.10 or 3.13
- All dependencies installed via `pip install -r requirements.txt`
- No GPU required — runs on CPU

---

## Troubleshooting

**OMP thread error on Windows:**
Already handled in `main.py` — the environment variables are set automatically.

**Model not found:**
Run `python training/dqn_training.py` and `python training/pg_training.py` first.

**Pygame display error:**
Make sure you are not running over SSH without a display. Run locally.

**Module not found:**
Make sure you ran `pip install -r requirements.txt` from the project root.

---

*Inzozi means "dreams" in Kinyarwanda.*
