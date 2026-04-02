import os, sys, subprocess, json, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT       = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(ROOT, "models", "pg_unity")
RESULTS    = os.path.join(ROOT, "results")
CONFIG_DIR = os.path.join(ROOT, "config")
for sub in ("ppo", "a2c", "reinforce"):
    os.makedirs(os.path.join(MODELS_DIR, sub), exist_ok=True)
os.makedirs(RESULTS,    exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

PPO_TEMPLATE = """behaviors:
  InzoziAgent:
    trainer_type: ppo
    hyperparameters:
      batch_size: {batch}
      buffer_size: {steps}
      learning_rate: {lr}
      learning_rate_schedule: linear
      beta: {ent}
      epsilon: {clip}
      lambd: {gae}
      num_epoch: 3
    network_settings:
      normalize: true
      hidden_units: {hidden}
      num_layers: 2
    reward_signals:
      extrinsic:
        gamma: {gamma}
        strength: 1.0
    max_steps: 30000
    time_horizon: 128
    summary_freq: 1000
    keep_checkpoints: 1
"""

A2C_TEMPLATE = """behaviors:
  InzoziAgent:
    trainer_type: ppo
    hyperparameters:
      batch_size: {steps}
      buffer_size: {steps}
      learning_rate: {lr}
      learning_rate_schedule: linear
      beta: {ent}
      epsilon: 0.2
      lambd: 0.95
      num_epoch: 1
    network_settings:
      normalize: true
      hidden_units: {hidden}
      num_layers: 2
    reward_signals:
      extrinsic:
        gamma: {gamma}
        strength: 1.0
    max_steps: 30000
    time_horizon: {steps}
    summary_freq: 1000
    keep_checkpoints: 1
"""

PPO_CONFIGS = [
    dict(run=1,  lr="3.0e-4", gamma=0.99, gae=0.95, ent=0.01, clip=0.2, steps=2048, batch=64,  hidden=128),
    dict(run=2,  lr="1.0e-4", gamma=0.99, gae=0.95, ent=0.00, clip=0.2, steps=1024, batch=64,  hidden=64),
    dict(run=3,  lr="5.0e-4", gamma=0.99, gae=0.90, ent=0.05, clip=0.1, steps=512,  batch=128, hidden=128),
    dict(run=4,  lr="3.0e-4", gamma=0.95, gae=0.95, ent=0.01, clip=0.3, steps=2048, batch=64,  hidden=128),
    dict(run=5,  lr="1.0e-3", gamma=0.99, gae=0.98, ent=0.00, clip=0.2, steps=4096, batch=256, hidden=256),
    dict(run=6,  lr="2.0e-4", gamma=0.99, gae=0.95, ent=0.02, clip=0.2, steps=1024, batch=64,  hidden=64),
    dict(run=7,  lr="3.0e-4", gamma=0.90, gae=0.80, ent=0.10, clip=0.4, steps=512,  batch=32,  hidden=64),
    dict(run=8,  lr="5.0e-4", gamma=0.99, gae=0.95, ent=0.01, clip=0.2, steps=2048, batch=128, hidden=128),
    dict(run=9,  lr="1.0e-4", gamma=0.999,gae=0.99, ent=0.00, clip=0.15,steps=4096, batch=64,  hidden=256),
    dict(run=10, lr="1.0e-3", gamma=0.95, gae=0.90, ent=0.05, clip=0.3, steps=512,  batch=64,  hidden=32),
]

A2C_CONFIGS = [
    dict(run=1,  lr="7.0e-4", gamma=0.99, ent=0.00, steps=5,   hidden=64),
    dict(run=2,  lr="3.0e-4", gamma=0.99, ent=0.01, steps=5,   hidden=64),
    dict(run=3,  lr="1.0e-3", gamma=0.95, ent=0.02, steps=10,  hidden=128),
    dict(run=4,  lr="5.0e-4", gamma=0.99, ent=0.05, steps=20,  hidden=64),
    dict(run=5,  lr="2.0e-4", gamma=0.99, ent=0.00, steps=5,   hidden=64),
    dict(run=6,  lr="7.0e-4", gamma=0.90, ent=0.01, steps=5,   hidden=64),
    dict(run=7,  lr="1.0e-3", gamma=0.99, ent=0.10, steps=50,  hidden=256),
    dict(run=8,  lr="3.0e-4", gamma=0.999,ent=0.00, steps=10,  hidden=128),
    dict(run=9,  lr="5.0e-4", gamma=0.95, ent=0.02, steps=20,  hidden=64),
    dict(run=10, lr="1.0e-4", gamma=0.99, ent=0.01, steps=100, hidden=32),
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


def run_mlagents(template, cfg, algo, run_id):
    config_path = os.path.join(CONFIG_DIR, f"{algo}_config.yaml")
    results_dir = os.path.join(RESULTS, "mlagents_runs")
    with open(config_path, "w") as f:
        f.write(template.format(**cfg))

    print(f"  --> Press Play in Unity NOW")
    cmd = [
        sys.executable, "-m", "mlagents.trainers.learn",
        config_path, f"--run-id={run_id}", "--force",
        f"--results-dir={results_dir}",
    ]
    subprocess.run(cmd, text=True)
    print(f"  Run complete.")
    return {"run": cfg["run"], "lr": cfg["lr"], "gamma": cfg["gamma"],
            "ent": cfg.get("ent", 0), "mean_reward": 0.0}

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(obs_dim)
        self.pi   = nn.Sequential(nn.Linear(obs_dim,hidden), nn.Tanh(),
                                  nn.Linear(hidden,hidden),  nn.Tanh(),
                                  nn.Linear(hidden,act_dim))
        self.vf   = nn.Sequential(nn.Linear(obs_dim,hidden), nn.Tanh(),
                                  nn.Linear(hidden,1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 0.5); nn.init.zeros_(m.bias)
    def forward(self, x):
        x = self.norm(x)
        l = torch.clamp(self.pi(x), -10, 10)
        p = torch.softmax(l, dim=-1)
        bad = torch.isnan(p).any(dim=-1)
        if bad.any(): p[bad] = torch.ones(p.shape[-1])/p.shape[-1]
        return p
    def value(self, x): return self.vf(self.norm(x)).squeeze(-1)


def train_reinforce(cfg):
    from environment.custom_env import InzoziGlassesEnv
    env     = InzoziGlassesEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    net     = PolicyNet(obs_dim, act_dim, cfg["hidden"])
    opt     = optim.Adam(net.parameters(), lr=cfg["lr"], eps=1e-5)
    rewards = []

    for ep in range(300):
        obs, _ = env.reset(seed=ep)
        lps, rs, obs_list = [], [], []
        done = False
        while not done:
            ot = torch.nan_to_num(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            pr = net(ot)
            if torch.isnan(pr).any(): pr = torch.ones(1, act_dim)/act_dim
            d  = torch.distributions.Categorical(pr)
            a  = d.sample()
            lps.append(d.log_prob(a)); obs_list.append(ot)
            obs, r, term, trunc, _ = env.step(a.item())
            rs.append(float(np.clip(r, -20, 20))); done = term or trunc
        rewards.append(sum(rs))
        if len(rs) < 2: continue
        G, rets = 0.0, []
        for r in reversed(rs): G = r+cfg["gamma"]*G; rets.insert(0,G)
        R = torch.tensor(rets, dtype=torch.float32)
        R = (R-R.mean())/(R.std()+1e-8) if R.std()>1e-6 else R-R.mean()
        adv   = R - net.value(torch.nan_to_num(torch.cat(obs_list))).detach() if cfg["baseline"] else R
        vloss = nn.MSELoss()(net.value(torch.nan_to_num(torch.cat(obs_list))), R) if cfg["baseline"] else torch.tensor(0.)
        ploss = torch.stack([-lp*a for lp,a in zip(lps,adv)]).sum()
        if torch.isnan(ploss): continue
        loss = ploss + 0.5*vloss
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
    env.close()
    torch.save(net.state_dict(), os.path.join(MODELS_DIR,"reinforce",f"rf_{cfg['run']}.pt"))
    mean_r = float(np.mean(rewards[-20:])) if len(rewards)>=20 else float(np.mean(rewards))
    print(f"        mean_reward={mean_r:.3f}")
    return {"run":cfg["run"],"lr":cfg["lr"],"gamma":cfg["gamma"],
            "baseline":cfg["baseline"],"mean_reward":round(mean_r,3)}


def save_csv(results, name):
    keys = [k for k in results[0]]
    with open(os.path.join(RESULTS, f"{name}_unity_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in results: w.writerow(r)


def main():
    print("="*55)
    print("PG UNITY TRAINING — PPO + A2C + REINFORCE")
    print("="*55)

    # PPO
    print("\n" + "="*40 + "\nPPO — 10 runs\n" + "="*40)
    ppo_res = []
    for c in PPO_CONFIGS:
        print(f"\n  PPO run {c['run']}/10  lr={c['lr']}  clip={c['clip']}")
        r = run_mlagents(PPO_TEMPLATE, c, "ppo", f"PPO_run_{c['run']}")
        ppo_res.append(r)
        if c["run"] < 10:
            input("  Stop Play in Unity → press Enter for next run...")
    save_csv(ppo_res, "ppo")

    # A2C
    print("\n" + "="*40 + "\nA2C — 10 runs\n" + "="*40)
    a2c_res = []
    for c in A2C_CONFIGS:
        print(f"\n  A2C run {c['run']}/10  lr={c['lr']}  steps={c['steps']}")
        r = run_mlagents(A2C_TEMPLATE, c, "a2c", f"A2C_run_{c['run']}")
        a2c_res.append(r)
        if c["run"] < 10:
            input("  Stop Play in Unity → press Enter for next run...")
    save_csv(a2c_res, "a2c")

    # REINFORCE (uses Pygame env — no Unity needed)
    print("\n" + "="*40 + "\nREINFORCE — 10 runs (Pygame env)\n" + "="*40)
    print("  REINFORCE runs on Pygame env — you can stop Unity Play now")
    rf_res = []
    for c in RF_CONFIGS:
        print(f"\n  REINFORCE run {c['run']}/10  lr={c['lr']}  baseline={c['baseline']}")
        r = train_reinforce(c); rf_res.append(r)
    save_csv(rf_res, "reinforce")

    print("\n=== All PG Unity training complete ===")
    print(f"Results saved to: {RESULTS}")


if __name__ == "__main__":
    main()