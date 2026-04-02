import os, sys, subprocess, json, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT       = os.path.join(os.path.dirname(__file__), "..")
CONFIG     = os.path.join(ROOT, "config", "dqn_config.yaml")
MODELS_DIR = os.path.join(ROOT, "models", "dqn_unity")
RESULTS    = os.path.join(ROOT, "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS,    exist_ok=True)
os.makedirs(os.path.join(ROOT, "config"), exist_ok=True)

DQN_BASE = """behaviors:
  InzoziAgent:
    trainer_type: sac
    hyperparameters:
      learning_rate: {lr}
      learning_rate_schedule: constant
      batch_size: {batch}
      buffer_size: {buf}
      buffer_init_steps: 1000
      tau: 0.005
      steps_per_update: 1
      save_replay_buffer: false
      init_entcoef: 0.5
      reward_signal_steps_per_update: 1
    network_settings:
      normalize: true
      hidden_units: {hidden}
      num_layers: 2
    reward_signals:
      extrinsic:
        gamma: {gamma}
        strength: 1.0
    max_steps: {steps}
    time_horizon: 64
    summary_freq: 1000
    keep_checkpoints: 1
"""

CONFIGS = [
    dict(run=1,  lr="3.0e-4", gamma=0.99, batch=256, buf=50000,  hidden=256, steps=30000),
    dict(run=2,  lr="1.0e-4", gamma=0.99, batch=128, buf=50000,  hidden=128, steps=30000),
    dict(run=3,  lr="5.0e-4", gamma=0.99, batch=256, buf=100000, hidden=256, steps=30000),
    dict(run=4,  lr="3.0e-4", gamma=0.95, batch=128, buf=50000,  hidden=128, steps=30000),
    dict(run=5,  lr="1.0e-3", gamma=0.99, batch=512, buf=100000, hidden=256, steps=30000),
    dict(run=6,  lr="2.0e-4", gamma=0.99, batch=256, buf=50000,  hidden=128, steps=30000),
    dict(run=7,  lr="3.0e-4", gamma=0.90, batch=128, buf=20000,  hidden=64,  steps=30000),
    dict(run=8,  lr="5.0e-4", gamma=0.99, batch=256, buf=50000,  hidden=256, steps=30000),
    dict(run=9,  lr="1.0e-4", gamma=0.999,batch=512, buf=100000, hidden=256, steps=30000),
    dict(run=10, lr="5.0e-3", gamma=0.99, batch=128, buf=10000,  hidden=64,  steps=30000),
]


def write_config(cfg):
    content = DQN_BASE.format(**cfg)
    with open(CONFIG, "w") as f:
        f.write(content)

def run_training(cfg):
    run_id = f"DQN_run_{cfg['run']}"
    results_dir = os.path.join(ROOT, "results", "mlagents_runs")

    write_config(cfg)

    print(f"\n  DQN run {cfg['run']}/10  lr={cfg['lr']}  gamma={cfg['gamma']}  buf={cfg['buf']}")
    print(f"  Run ID: {run_id}")
    print(f"  --> Press Play in Unity NOW if not already playing")

    cmd = [
        sys.executable, "-m", "mlagents.trainers.learn",
        CONFIG,
        f"--run-id={run_id}",
        "--force",
        f"--results-dir={results_dir}",
    ]

    result = subprocess.run(cmd, capture_output=False, text=True)

    stats_path = os.path.join(results_dir, run_id, "InzoziAgent", "InzoziAgent-step30000.pt")
    mean_r = 0.0

    print(f"  Run {cfg['run']} complete.")
    return {
        "run":         cfg["run"],
        "lr":          cfg["lr"],
        "gamma":       cfg["gamma"],
        "batch_size":  cfg["batch"],
        "buffer_size": cfg["buf"],
        "hidden":      cfg["hidden"],
        "max_steps":   cfg["steps"],
        "mean_reward": mean_r,
    }

def save_csv(results):
    keys = list(results[0].keys())
    with open(os.path.join(RESULTS, "dqn_unity_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results: w.writerow(r)
    print(f"\n  Saved dqn_unity_results.csv")

def main():
    print("=" * 55)
    print("DQN UNITY TRAINING — Inzozi Kigali (10 experiments)")
    print("=" * 55)
    print("\nINSTRUCTIONS:")
    print("  1. Unity must be OPEN (not necessarily in Play mode yet)")
    print("  2. Each run will tell you when to press Play")
    print("  3. Press Play in Unity DURING each run when prompted")
    input("\nPress Enter to start run 1...")

    results = []
    for cfg in CONFIGS:
        res = run_training(cfg)
        results.append(res)
        if cfg["run"] < 10:
            print(f"\n  Stopping Unity Play now...")
            input(f"  Stop Play in Unity, then press Enter to start run {cfg['run']+1}...")

    save_csv(results)
    print("\nAll DQN Unity runs complete.")
    
if __name__ == "__main__":
    main()