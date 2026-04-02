import os, sys, csv, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from environment.custom_env import InzoziGlassesEnv

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS_DIR  = os.path.join(os.path.dirname(__file__), "..", "models", "pg")
for sub in ("ppo", "a2c", "reinforce"):
    os.makedirs(os.path.join(MODELS_DIR, sub), exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

TIMESTEPS   = 60_000
RF_EPISODES = 500

PPO_CONFIGS = [
    dict(run=1,  lr=3e-4, gamma=0.99, gae=0.95, ent=0.01, clip=0.2,  steps=2048, batch=64,  net=[64,64]),
    dict(run=2,  lr=1e-4, gamma=0.99, gae=0.95, ent=0.00, clip=0.2,  steps=1024, batch=64,  net=[64,64]),
    dict(run=3,  lr=5e-4, gamma=0.99, gae=0.90, ent=0.05, clip=0.1,  steps=512,  batch=128, net=[128,128]),
    dict(run=4,  lr=3e-4, gamma=0.95, gae=0.95, ent=0.01, clip=0.3,  steps=2048, batch=64,  net=[128,64]),
    dict(run=5,  lr=1e-3, gamma=0.99, gae=0.98, ent=0.00, clip=0.2,  steps=4096, batch=256, net=[256,256]),
    dict(run=6,  lr=2e-4, gamma=0.99, gae=0.95, ent=0.02, clip=0.2,  steps=1024, batch=64,  net=[64,64]),
    dict(run=7,  lr=3e-4, gamma=0.90, gae=0.80, ent=0.10, clip=0.4,  steps=512,  batch=32,  net=[64,64]),
    dict(run=8,  lr=5e-4, gamma=0.99, gae=0.95, ent=0.01, clip=0.2,  steps=2048, batch=128, net=[128,128]),
    dict(run=9,  lr=1e-4, gamma=0.999,gae=0.99, ent=0.00, clip=0.15, steps=4096, batch=64,  net=[256,128]),
    dict(run=10, lr=1e-3, gamma=0.95, gae=0.90, ent=0.05, clip=0.3,  steps=512,  batch=64,  net=[32,32]),
]

A2C_CONFIGS = [
    dict(run=1,  lr=7e-4, gamma=0.99, ent=0.00, steps=5,   net=[64,64]),
    dict(run=2,  lr=3e-4, gamma=0.99, ent=0.01, steps=5,   net=[64,64]),
    dict(run=3,  lr=1e-3, gamma=0.95, ent=0.02, steps=10,  net=[128,128]),
    dict(run=4,  lr=5e-4, gamma=0.99, ent=0.05, steps=20,  net=[64,64]),
    dict(run=5,  lr=2e-4, gamma=0.99, ent=0.00, steps=5,   net=[128,64]),
    dict(run=6,  lr=7e-4, gamma=0.90, ent=0.01, steps=5,   net=[64,64]),
    dict(run=7,  lr=1e-3, gamma=0.99, ent=0.10, steps=50,  net=[256,256]),
    dict(run=8,  lr=3e-4, gamma=0.999,ent=0.00, steps=10,  net=[128,128]),
    dict(run=9,  lr=5e-4, gamma=0.95, ent=0.02, steps=20,  net=[64,64]),
    dict(run=10, lr=1e-4, gamma=0.99, ent=0.01, steps=100, net=[32,32]),
]

RF_CONFIGS = [
    dict(run=1,  lr=1e-3, gamma=0.99, baseline=True,  hidden=64),
    dict(run=2,  lr=5e-4, gamma=0.99, baseline=False, hidden=64),
    dict(run=3,  lr=2e-3, gamma=0.99, baseline=True,  hidden=128),
    dict(run=4,  lr=1e-3, gamma=0.95, baseline=True,  hidden=64),
    dict(run=5,  lr=1e-3, gamma=0.90, baseline=False, hidden=64),
    dict(run=6,  lr=5e-3, gamma=0.99, baseline=True,  hidden=64),
    dict(run=7,  lr=1e-4, gamma=0.99, baseline=False, hidden=128),
    dict(run=8,  lr=1e-3, gamma=0.999,baseline=True,  hidden=64),
    dict(run=9,  lr=2e-4, gamma=0.95, baseline=True,  hidden=64),
    dict(run=10, lr=1e-2, gamma=0.99, baseline=False, hidden=32),
]


class MetricsLog(BaseCallback):
    def __init__(self):
        super().__init__()
        self.ep_rewards = []
        self._cur = 0.0
    def _on_step(self) -> bool:
        self._cur += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.ep_rewards.append(self._cur); self._cur = 0.0
        return True

def train_ppo(cfg):
    env = Monitor(InzoziGlassesEnv())
    cb  = MetricsLog()
    model = PPO("MlpPolicy", env,
                learning_rate=cfg["lr"], gamma=cfg["gamma"],
                gae_lambda=cfg["gae"], ent_coef=cfg["ent"],
                clip_range=cfg["clip"], n_steps=cfg["steps"],
                batch_size=cfg["batch"],
                policy_kwargs={"net_arch": cfg["net"]},
                verbose=0, seed=cfg["run"]*13)
    model.learn(TIMESTEPS, callback=cb)
    rews   = cb.ep_rewards
    mean_r = float(np.mean(rews[-20:])) if len(rews)>=20 else (float(np.mean(rews)) if rews else 0.0)
    model.save(os.path.join(MODELS_DIR, "ppo", f"ppo_run_{cfg['run']}")); env.close()
    return {**{k:v for k,v in cfg.items() if k!="net"}, "net_arch": str(cfg["net"]),
            "mean_reward": round(mean_r,3), "n_episodes": len(rews), "_rewards": rews}

def train_a2c(cfg):
    env = Monitor(InzoziGlassesEnv())
    cb  = MetricsLog()
    model = A2C("MlpPolicy", env,
                learning_rate=cfg["lr"], gamma=cfg["gamma"],
                ent_coef=cfg["ent"], n_steps=cfg["steps"],
                policy_kwargs={"net_arch": cfg["net"]},
                verbose=0, seed=cfg["run"]*17)
    model.learn(TIMESTEPS, callback=cb)
    rews   = cb.ep_rewards
    mean_r = float(np.mean(rews[-20:])) if len(rews)>=20 else (float(np.mean(rews)) if rews else 0.0)
    model.save(os.path.join(MODELS_DIR, "a2c", f"a2c_run_{cfg['run']}")); env.close()
    return {**{k:v for k,v in cfg.items() if k!="net"}, "net_arch": str(cfg["net"]),
            "mean_reward": round(mean_r,3), "n_episodes": len(rews), "_rewards": rews}


class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.input_norm = nn.LayerNorm(obs_dim)          
        self.pi = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),       
            nn.Linear(hidden,  hidden), nn.Tanh(),
            nn.Linear(hidden,  act_dim),
        )
        self.vf = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        for m in self.modules():                        
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x      = self.input_norm(x)
        logits = torch.clamp(self.pi(x), -10.0, 10.0)   
        probs  = torch.softmax(logits, dim=-1)
        bad    = torch.isnan(probs).any(dim=-1)
        if bad.any():
            probs[bad] = torch.ones(probs.shape[-1]) / probs.shape[-1]
        return probs

    def value(self, x):
        return self.vf(self.input_norm(x)).squeeze(-1)


def train_reinforce(cfg):
    env     = InzoziGlassesEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    net     = PolicyNet(obs_dim, act_dim, cfg["hidden"])
    opt     = optim.Adam(net.parameters(), lr=cfg["lr"], eps=1e-5)
    rews_all = []

    for ep in range(RF_EPISODES):
        obs, _ = env.reset(seed=ep)
        lps, rs, obs_list = [], [], []
        done = False
        while not done:
            ot  = torch.nan_to_num(
                    torch.tensor(obs, dtype=torch.float32).unsqueeze(0),
                    nan=0.0, posinf=1.0, neginf=0.0)
            pr  = net(ot)
            if torch.isnan(pr).any() or pr.sum() < 0.5:
                pr = torch.ones(1, act_dim) / act_dim
            d = torch.distributions.Categorical(pr)
            a = d.sample()
            lps.append(d.log_prob(a)); obs_list.append(ot)
            obs, r, term, trunc, _ = env.step(a.item())
            rs.append(float(np.clip(r, -20.0, 20.0)))
            done = term or trunc

        rews_all.append(sum(rs))
        if len(rs) < 2: continue

        G, rets = 0.0, []
        for r in reversed(rs):
            G = r + cfg["gamma"] * G; rets.insert(0, G)
        R = torch.tensor(rets, dtype=torch.float32)
        R = (R - R.mean()) / (R.std() + 1e-8) if R.std() > 1e-6 else R - R.mean()

        if cfg["baseline"] and obs_list:
            obs_b = torch.nan_to_num(torch.cat(obs_list))
            vals  = net.value(obs_b)
            adv   = R - vals.detach()
            vloss = nn.MSELoss()(vals, R)
        else:
            adv = R; vloss = torch.tensor(0.0)

        if any(torch.isnan(lp) for lp in lps): continue
        ploss = torch.stack([-lp * a for lp, a in zip(lps, adv)]).sum()
        if torch.isnan(ploss): continue

        loss = ploss + 0.5 * vloss
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)  
        opt.step()

    torch.save(net.state_dict(),
               os.path.join(MODELS_DIR, "reinforce", f"rf_run_{cfg['run']}.pt"))
    env.close()
    mean_r = float(np.mean(rews_all[-20:])) if len(rews_all)>=20 else float(np.mean(rews_all))
    return {**cfg, "mean_reward": round(mean_r,3), "n_episodes": len(rews_all), "_rewards": rews_all}


def save_csv(results, name):
    keys = [k for k in results[0] if not k.startswith("_")]
    with open(os.path.join(RESULTS_DIR, f"{name}_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in results: w.writerow({k:r[k] for k in keys})


def plot_10(results, title, filename, color):
    fig, axes = plt.subplots(2, 5, figsize=(18,7), constrained_layout=True)
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
        ax.set_title(f"Run {res['run']}  mean={res['mean_reward']:.2f}", fontsize=8)
        ax.set_xlabel("Episode",fontsize=7); ax.set_ylabel("Reward",fontsize=7)
        ax.tick_params(labelsize=6); ax.grid(True,alpha=0.22,lw=0.4)
        ax.set_facecolor("#1a2028")
        if res["mean_reward"] > best_mean:
            best_mean, best_run = res["mean_reward"], res["run"]
    for sp in axes.flat[best_run-1].spines.values():
        sp.set_edgecolor("#FFD700"); sp.set_linewidth(2)
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}. Best: run {best_run} mean={best_mean:.3f}")
    return best_run


def main():
    best = {}

    print("\n" + "="*50 + "\nPPO — 10 runs\n" + "="*50)
    ppo_res = []
    for c in PPO_CONFIGS:
        print(f"  PPO run {c['run']}/10  lr={c['lr']}  clip={c['clip']}  gae={c['gae']}")
        r = train_ppo(c); ppo_res.append(r)
        print(f"        mean_reward={r['mean_reward']:.3f}")
    save_csv(ppo_res, "ppo")
    best["ppo"] = plot_10(ppo_res,"PPO — Inzozi Kigali (10 runs)","ppo_reward_curves.png","#5BC8E8")
    json.dump({"best_run": best["ppo"]}, open(os.path.join(MODELS_DIR,"ppo","best_run.json"),"w"), indent=2)

    print("\n" + "="*50 + "\nA2C — 10 runs\n" + "="*50)
    a2c_res = []
    for c in A2C_CONFIGS:
        print(f"  A2C run {c['run']}/10  lr={c['lr']}  steps={c['steps']}  ent={c['ent']}")
        r = train_a2c(c); a2c_res.append(r)
        print(f"        mean_reward={r['mean_reward']:.3f}")
    save_csv(a2c_res, "a2c")
    best["a2c"] = plot_10(a2c_res,"A2C — Inzozi Kigali (10 runs)","a2c_reward_curves.png","#E8A85B")
    json.dump({"best_run": best["a2c"]}, open(os.path.join(MODELS_DIR,"a2c","best_run.json"),"w"), indent=2)

    print("\n" + "="*50 + "\nREINFORCE — 10 runs\n" + "="*50)
    rf_res = []
    for c in RF_CONFIGS:
        print(f"  REINFORCE run {c['run']}/10  lr={c['lr']}  baseline={c['baseline']}")
        r = train_reinforce(c); rf_res.append(r)
        print(f"             mean_reward={r['mean_reward']:.3f}")
    save_csv(rf_res, "reinforce")
    best["reinforce"] = plot_10(rf_res,"REINFORCE — Inzozi Kigali (10 runs)","reinforce_reward_curves.png","#C86AE0")

    print("\n=== Best runs summary ===")
    for alg, run in best.items():
        print(f"  {alg.upper():<12} best run = {run}")
    json.dump(best, open(os.path.join(RESULTS_DIR,"best_runs.json"),"w"), indent=2)
    print("\nAll PG training complete.")

if __name__ == "__main__":
    main()