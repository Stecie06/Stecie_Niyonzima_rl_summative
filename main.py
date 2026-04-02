import os
import sys
import argparse
import time
import pygame
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from environment.custom_env import InzoziGlassesEnv, ACTIONS
from stable_baselines3 import A2C, PPO
from environment.rendering import (
    SCENE_W, SCENE_H, PANEL_W, HUD_H, WIN_W, WIN_H,
    make_background, draw_goal, draw_raycasts, draw_obstacles,
    draw_agent, draw_panel, draw_hud, C_BG
)

PROJECT_ROOT = os.path.dirname(__file__)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

import environment.rendering as rendering
rendering.MAX_EPISODES = 1000

def load_a2c_model(run=3):
    model_path = os.path.join(MODELS_DIR, "pg", "a2c", f"a2c_run_{run}")
    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"A2C model not found: {model_path}.zip")
    return A2C.load(model_path)

def load_ppo_model(run=7):
    model_path = os.path.join(MODELS_DIR, "pg", "ppo", f"ppo_run_{run}")
    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"PPO model not found: {model_path}.zip")
    return PPO.load(model_path)

def run_continuous(model, algo_name, run_number, duration_seconds, screen, clock,
                   font, font_big, font_small, bg, fps=15):
    start_time = time.time()
    episode_count = 0
    total_steps = 0
    total_reward_all = 0.0

    print("\n" + "="*70)
    print(f"{algo_name} Run {run_number} - {duration_seconds} seconds")
    print("Press ESC/Q to quit early.")
    print("="*70)

    while (time.time() - start_time) < duration_seconds:
        episode_count += 1
        env = InzoziGlassesEnv(render_mode=None)
        obs, info = env.reset()
        episode_reward = 0.0
        step = 0
        done = False

        print(f"\n--- Episode {episode_count} started ---")

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit()
                    return

            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step += 1
            total_steps += 1
            total_reward_all += reward

            if step % 10 == 0 or done:
                action_name = ACTIONS[action]
                status = ""
                if terminated:
                    status = info.get("result", "TERMINATED")
                elif truncated:
                    status = "TIMEOUT"
                print(f"Ep{episode_count:3d} Step{step:4d} {action_name:<22} "
                      f"R:{reward:+6.2f} EpR:{episode_reward:+7.2f} {status}")

            scene = pygame.Surface((SCENE_W, SCENE_H))
            scene.blit(bg, (0, 0))
            draw_goal(scene, env.goal_x, env.goal_y)
            draw_raycasts(scene, env.agent_x, env.agent_y, env.heading)
            draw_obstacles(scene, env.obstacles)
            draw_agent(scene, env.agent_x, env.agent_y, env.heading)

            screen.fill(C_BG)
            screen.blit(scene, (0, 0))
            elapsed = time.time() - start_time
            remaining = max(0, duration_seconds - elapsed)
            info_panel = {"result": f"Episode {episode_count} | Time left: {remaining:.0f}s"}
            draw_panel(screen, obs, action, reward, episode_count, step,
                       episode_reward, font, font_big)
            draw_hud(screen, episode_count, step, episode_reward, info_panel,
                     font, font_big)
            pygame.display.flip()
            clock.tick(fps)

        print(f"--- Episode {episode_count} finished. Steps: {step}, Reward: {episode_reward:.2f} ---")
        env.close()
        time.sleep(0.5)

    print(f"\n{algo_name} finished after {duration_seconds} seconds.")
    print(f"Episodes: {episode_count}, Steps: {total_steps}, Total reward: {total_reward_all:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run A2C then PPO, total 3 min")
    parser.add_argument("--duration", type=int, default=90,
                        help="Duration in seconds for EACH algorithm (default: 90)")
    args = parser.parse_args()

    per_algo_duration = args.duration

    print("="*70)
    print(f"TOTAL DEMO: {per_algo_duration*2} seconds ({per_algo_duration}s A2C + {per_algo_duration}s PPO)")
    print("="*70)

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Inzozi Glasses - Continuous Demo")
    clock = pygame.time.Clock()

    try:
        font = pygame.font.SysFont("DejaVuSans", 12)
        font_big = pygame.font.SysFont("DejaVuSans", 14, bold=True)
        font_small = pygame.font.SysFont("DejaVuSans", 9)
    except:
        font = pygame.font.Font(None, 13)
        font_big = pygame.font.Font(None, 15)
        font_small = pygame.font.Font(None, 9)

    bg = make_background(font_small)

    print("\nLoading A2C run 3...")
    a2c_model = load_a2c_model(3)
    print(f"A2C loaded. Running for {per_algo_duration} seconds...")
    run_continuous(a2c_model, "A2C", 3, per_algo_duration, screen, clock,
                   font, font_big, font_small, bg, fps=15)

    print("\n" + "="*70)
    print("A2C finished. Starting PPO run 7 in 1 second...")
    print("="*70)
    time.sleep(1)

    print("\nLoading PPO run 7...")
    ppo_model = load_ppo_model(7)
    print(f"PPO loaded. Running for {per_algo_duration} seconds...")
    run_continuous(ppo_model, "PPO", 7, per_algo_duration, screen, clock,
                   font, font_big, font_small, bg, fps=15)

    print("\n" + "="*70)
    print("Both demonstrations completed.")
    print("="*70)

    pygame.quit()