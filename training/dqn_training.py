import os, sys, csv, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from environment.custom_env import InzoziGlassesEnv

MODELS_DIR  = os.path.join(os.path.dirname(__file__), "..", "models", "dqn")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

TIMESTEPS = 60_000   

CONFIGS = [
    dict(run=1,  lr=1e-3,  gamma=0.99, eps_end=0.05, eps_frac=0.1,  buf=10_000, batch=64,  tau=1.0,  net=[64,64]),
    dict(run=2,  lr=5e-4,  gamma=0.99, eps_end=0.05, eps_frac=0.2,  buf=50_000, batch=64,  tau=1.0,  net=[64,64]),
    dict(run=3,  lr=1e-4,  gamma=0.99, eps_end=0.01, eps_frac=0.3,  buf=50_000, batch=128, tau=1.0,  net=[128,128]),
    dict(run=4,  lr=1e-3,  gamma=0.95, eps_end=0.10, eps_frac=0.2,  buf=10_000, batch=32,  tau=0.5,  net=[64,64]),
    dict(run=5,  lr=5e-4,  gamma=0.95, eps_end=0.05, eps_frac=0.4,  buf=20_000, batch=64,  tau=0.5,  net=[128,64]),
    dict(run=6,  lr=2e-4,  gamma=0.99, eps_end=0.02, eps_frac=0.1,  buf=100_000,batch=128, tau=1.0,  net=[256,256]),
    dict(run=7,  lr=1e-3,  gamma=0.90, eps_end=0.10, eps_frac=0.5,  buf=5_000,  batch=32,  tau=1.0,  net=[64,64]),
    dict(run=8,  lr=3e-4,  gamma=0.99, eps_end=0.05, eps_frac=0.2,  buf=50_000, batch=64,  tau=0.1,  net=[128,128]),
    dict(run=9,  lr=1e-4,  gamma=0.999,eps_end=0.01, eps_frac=0.3,  buf=100_000,batch=256, tau=0.01, net=[256,128]),
    dict(run=10, lr=5e-3,  gamma=0.99, eps_end=0.20, eps_frac=0.1,  buf=10_000, batch=64,  tau=1.0,  net=[32,32]),
]

class EpisodeRewardLog(BaseCallback):
    def __init__(self):
        super().__init__()
        self.ep_rewards = []
        self._cur = 0.0

    def _on_step(self) -> bool:
        self._cur += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.ep_rewards.append(self._cur)
            self._cur = 0.0
        return True

def train_one(cfg: dict, verbose=0) -> dict:
    env = Monitor(InzoziGlassesEnv())
    cb  = EpisodeRewardLog()

    model = DQN(
        "MlpPolicy", env,
        learning_rate          = cfg["lr"],
        gamma                  = cfg["gamma"],
        exploration_final_eps  = cfg["eps_end"],
        exploration_fraction   = cfg["eps_frac"],
        buffer_size            = cfg["buf"],
        batch_size             = cfg["batch"],
        tau                    = cfg["tau"],
        policy_kwargs          = {"net_arch": cfg["net"]},
        verbose=verbose, seed=cfg["run"] * 7,
    )
    model.learn(TIMESTEPS, callback=cb)

    rews   = cb.ep_rewards
    mean_r = float(np.mean(rews[-20:])) if len(rews) >= 20 else (float(np.mean(rews)) if rews else 0.0)

    path = os.path.join(MODELS_DIR, f"dqn_run_{cfg['run']}")
    model.save(path)
    env.close()

    return {**{k: v for k, v in cfg.items() if k != "net"},
            "net_arch":    str(cfg["net"]),
            "mean_reward": round(mean_r, 3),
            "n_episodes":  len(rews),
            "_rewards":    rews}

def save_csv(results, name):
    skip = {"_rewards"}
    keys = [k for k in results[0] if k not in skip]
    with open(os.path.join(RESULTS_DIR, f"{name}_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in keys})

def plot_curves(results, title, filename, color="#4DC898"):
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), constrained_layout=True)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.patch.set_facecolor("#141820")
    best_mean, best_run = -np.inf, 1

    for res, ax in zip(results, axes.flat):
        rews = res["_rewards"]
        if rews:
            w  = min(10, len(rews))
            sm = np.convolve(rews, np.ones(w)/w, mode="valid")
            ax.plot(sm, lw=1.2, color=color)
            ax.fill_between(range(len(sm)), sm, alpha=0.15, color=color)
        ax.set_title(
            f"Run {res['run']}  lr={res['lr']}\n"
            f"γ={res['gamma']}  ε={res['eps_end']}  buf={res['buf']}",
            fontsize=7.5)
        ax.set_xlabel("Episode", fontsize=7); ax.set_ylabel("Reward", fontsize=7)
        ax.tick_params(labelsize=6); ax.grid(True, alpha=0.22, lw=0.4)
        ax.set_facecolor("#1a2028")
        if res["mean_reward"] > best_mean:
            best_mean, best_run = res["mean_reward"], res["run"]

    for spine in axes.flat[best_run-1].spines.values():
        spine.set_edgecolor("#FFD700"); spine.set_linewidth(2)

    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}. Best run: {best_run} (mean={best_mean:.3f})")
    return best_run

def main():
    print("=" * 55)
    print("DQN — 10 hyperparameter experiments (Inzozi Glasses)")
    print("=" * 55)
    results = []
    for cfg in CONFIGS:
        print(f"\n  Run {cfg['run']}/10  lr={cfg['lr']}  gamma={cfg['gamma']}  "
              f"eps_end={cfg['eps_end']}  buf={cfg['buf']}")
        res = train_one(cfg)
        results.append(res)
        print(f"        mean_reward={res['mean_reward']:.3f}  episodes={res['n_episodes']}")

    save_csv(results, "dqn")
    best = plot_curves(results, "DQN hyperparameter experiments — Inzozi Kigali",
                       "dqn_reward_curves.png")
    json.dump({"best_run": best},
              open(os.path.join(MODELS_DIR, "best_run.json"), "w"), indent=2)
    print("\nDQN training complete.")


if __name__ == "__main__":
    main()