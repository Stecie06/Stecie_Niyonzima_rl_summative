import sys
import os
import math
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pygame
from environment.custom_env import InzoziGlassesEnv, ACTIONS, HILL_BANDS, AT_WEIGHT

SCALE   = 32          
ENV_M   = 20          
SCENE_W = ENV_M * SCALE   
SCENE_H = ENV_M * SCALE   
PANEL_W = 250         
HUD_H   = 100        
WIN_W   = SCENE_W + PANEL_W
WIN_H   = SCENE_H + HUD_H
FPS     = 20

C_BG          = (18, 24, 32)
C_ROAD        = (36, 46, 54)
C_LANE        = (55, 68, 78)
C_HILL        = (140, 110, 55)
C_PAVEMENT    = (52, 62, 68)
C_ZEBRA       = (190, 190, 190)
C_AGENT       = (72, 210, 130)
C_AGENT_RING  = (255, 255, 255)
C_GOAL        = (255, 215, 0)
C_MOTO        = (215, 75, 55)
C_PED         = (95, 175, 220)
C_STATIC      = (120, 88, 62)
C_RAY_F       = (55, 215, 170)
C_RAY_L       = (215, 175, 55)
C_RAY_R       = (215, 55, 175)
C_PANEL_BG    = (22, 30, 40)
C_HUD_BG      = (14, 18, 26)
C_TEXT        = (185, 205, 195)
C_TITLE       = (95, 215, 170)
C_ACTION      = (72, 225, 155)
C_REWARD_POS  = (72, 225, 130)
C_REWARD_NEG  = (225, 72, 72)
C_BAR_BG      = (38, 52, 48)
C_BAR_OK      = (72, 200, 120)
C_BAR_MID     = (215, 135, 55)
C_BAR_HIGH    = (215, 65, 65)

SHOP_LABELS = ["Amata", "Inyama", "MTN", "Boutique", "Pharmacie",
               "Umuganda", "Akabanga", "Inzozi"]

def px(metres: float) -> int:
    return int(metres * SCALE)

def make_background(font_small) -> pygame.Surface:
    surf = pygame.Surface((SCENE_W, SCENE_H))
    surf.fill(C_ROAD)

    for i in range(0, SCENE_W, SCALE * 4):
        pygame.draw.line(surf, C_LANE, (i, 0), (i, SCENE_H), 1)
    for j in range(0, SCENE_H, SCALE * 4):
        pygame.draw.line(surf, C_LANE, (0, j), (SCENE_W, j), 1)

    for y0, y1, grad in HILL_BANDS:
        alpha = int(28 + grad * 220)
        s     = pygame.Surface((SCENE_W, px(y1 - y0)), pygame.SRCALPHA)
        s.fill((*C_HILL, alpha))
        surf.blit(s, (0, px(y0)))

    mid = px(ENV_M / 2)
    for i in range(5):
        zx = px(ENV_M / 2 - 1.2 + i * 0.5)
        pygame.draw.rect(surf, C_ZEBRA, (zx, mid - px(1), px(0.3), px(2)))

    pygame.draw.rect(surf, C_PAVEMENT, (0, 0, SCENE_W, 3))
    pygame.draw.rect(surf, C_PAVEMENT, (0, SCENE_H - 3, SCENE_W, 3))
    pygame.draw.rect(surf, C_PAVEMENT, (0, 0, 3, SCENE_H))
    pygame.draw.rect(surf, C_PAVEMENT, (SCENE_W - 3, 0, 3, SCENE_H))

    shop_positions = [
        (3, 3), (7, 7), (14, 5), (5, 14), (16, 16),
        (10, 3), (3, 16), (16, 10),
    ]
    for idx, (sx, sy) in enumerate(shop_positions):
        spx = px(sx)
        spy = px(sy)
        pygame.draw.rect(surf, C_STATIC, (spx - 14, spy - 12, 30, 22))
        pygame.draw.rect(surf, (150, 115, 80), (spx - 14, spy - 12, 30, 22), 1)
        label = font_small.render(
            SHOP_LABELS[idx % len(SHOP_LABELS)], True, (215, 195, 155)
        )
        surf.blit(label, (spx - 14, spy - 22))

    return surf

def draw_goal(surf, gx, gy):
    ppx, ppy = px(gx), px(gy)
    for r, a in [(22, 40), (15, 90), (9, 200)]:
        s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*C_GOAL, a), (r, r), r)
        surf.blit(s, (ppx - r, ppy - r))
    pygame.draw.circle(surf, C_GOAL, (ppx, ppy), 8)
    pygame.draw.circle(surf, (255, 255, 255), (ppx, ppy), 8, 2)

def draw_obstacles(surf, obstacles):
    for obs in obstacles:
        opx = px(obs.x)
        opy = px(obs.y)
        if obs.kind == "moto":
            pygame.draw.ellipse(surf, C_MOTO, (opx - 10, opy - 5, 20, 10))
            pygame.draw.ellipse(surf, (255, 130, 90),
                                (opx - 10, opy - 5, 20, 10), 1)
            pygame.draw.circle(surf, (255, 240, 160), (opx + 9, opy), 3)
        elif obs.kind == "pedestrian":
            pygame.draw.circle(surf, C_PED, (opx, opy - 2), 5)
            pygame.draw.line(surf, C_PED, (opx, opy + 3),  (opx, opy + 13), 2)
            pygame.draw.line(surf, C_PED, (opx, opy + 7),  (opx - 4, opy + 12), 1)
            pygame.draw.line(surf, C_PED, (opx, opy + 7),  (opx + 4, opy + 12), 1)
            pygame.draw.line(surf, C_PED, (opx, opy + 3),  (opx - 5, opy + 8), 1)
            pygame.draw.line(surf, C_PED, (opx, opy + 3),  (opx + 5, opy + 8), 1)

def draw_raycasts(surf, ax, ay, heading):
    apx, apy = px(ax), px(ay)
    for color, off in [(C_RAY_F, 0.0), (C_RAY_L, -math.pi/4), (C_RAY_R, math.pi/4)]:
        ang  = heading + off
        epx  = int((ax + math.cos(ang) * 5) * SCALE)
        epy  = int((ay + math.sin(ang) * 5) * SCALE)
        s    = pygame.Surface((SCENE_W, SCENE_H), pygame.SRCALPHA)
        pygame.draw.line(s, (*color, 70), (apx, apy), (epx, epy), 1)
        surf.blit(s, (0, 0))

def draw_agent(surf, ax, ay, heading):
    apx, apy = px(ax), px(ay)
    
    halo = pygame.Surface((36, 36), pygame.SRCALPHA)
    pygame.draw.circle(halo, (*C_AGENT, 55), (18, 18), 18)
    surf.blit(halo, (apx - 18, apy - 18))
    
    pygame.draw.circle(surf, C_AGENT, (apx, apy), 10)
    pygame.draw.circle(surf, C_AGENT_RING, (apx, apy), 10, 2)
    
    hx = int(apx + math.cos(heading) * 15)
    hy = int(apy + math.sin(heading) * 15)
    pygame.draw.line(surf, (255, 255, 255), (apx, apy), (hx, hy), 2)
    
    pygame.draw.circle(surf, (200, 230, 215), (apx - 4, apy), 3, 1)
    pygame.draw.circle(surf, (200, 230, 215), (apx + 4, apy), 3, 1)
    pygame.draw.line(surf, (200, 230, 215), (apx - 1, apy), (apx + 1, apy), 1)


def draw_panel(screen, obs, action, reward, episode, step,
               total_r, font, font_big):
    px0 = SCENE_W
    pygame.draw.rect(screen, C_PANEL_BG, (px0, 0, PANEL_W, SCENE_H))
    pygame.draw.line(screen, (50, 78, 65), (px0, 0), (px0, SCENE_H), 1)

    y = 14

    def put(text, color=C_TEXT, big=False):
        nonlocal y
        f  = font_big if big else font
        screen.blit(f.render(text, True, color), (px0 + 10, y))
        y += 21 if big else 16

    put("INZOZI GLASSES", C_TITLE, big=True)
    put("Rwanda nav · random agent", (115, 160, 145))
    y += 4
    put(f"Episode  {episode}")
    put(f"Step     {step}/500")
    put(f"Reward   {reward:+.2f}",
        C_REWARD_POS if reward >= 0 else C_REWARD_NEG)
    put(f"Total R  {total_r:.1f}")
    put(f"AT wt    {AT_WEIGHT:.2f}")
    y += 6

    ax_rect = pygame.Rect(px0 + 10, y, PANEL_W - 20, 28)
    pygame.draw.rect(screen, (28, 46, 40), ax_rect, border_radius=5)
    pygame.draw.rect(screen, (75, 155, 110), ax_rect, 1, border_radius=5)
    aname   = ACTIONS[action] if action is not None else "—"
    screen.blit(font_big.render(aname, True, C_ACTION), (px0 + 15, y + 5))
    y += 36

    put("Sensors", (155, 175, 165))
    labels = [
        ("Front",   obs[0]),
        ("Left",    obs[1]),
        ("Right",   obs[2]),
        ("Traffic", obs[3]),
        ("Crowd",   obs[4]),
        ("Slope",   obs[5]),
        ("Goal",    obs[7]),
        ("AT wt",   obs[9]),
    ]
    bar_total = PANEL_W - 94
    for label, val in labels:
        bw  = int(bar_total * float(val))
        bx  = px0 + 10
        clr = C_BAR_OK if val < 0.5 else C_BAR_MID if val < 0.8 else C_BAR_HIGH
        pygame.draw.rect(screen, C_BAR_BG, (bx + 70, y + 2, bar_total, 10))
        if bw > 0:
            pygame.draw.rect(screen, clr, (bx + 70, y + 2, bw, 10))
        screen.blit(font.render(label, True, (145, 160, 155)), (bx, y))
        y += 14

    y += 4
    put("Rwanda coverage:")
    put("Partial · Vision Yes · Mobility Yes", (115, 160, 140))

def draw_hud(screen, episode, step, total_r, info, font, font_big):
    hy = SCENE_H
    pygame.draw.rect(screen, C_HUD_BG, (0, hy, WIN_W, HUD_H))
    pygame.draw.line(screen, (45, 70, 58), (0, hy), (WIN_W, hy), 1)

    screen.blit(
        font_big.render(
            "INZOZI SMART GLASSES — KIGALI ASSISTIVE NAVIGATION RL",
            True, C_TITLE,
        ),
        (12, hy + 8),
    )

    items = [
        f"Episode: {episode}/{MAX_EPISODES}",
        f"Step: {step}",
        f"Total reward: {total_r:.1f}",
        f"AT weight (Rwanda): {AT_WEIGHT:.2f}",
        f"Result: {info.get('result', '...')}",
    ]
    for i, item in enumerate(items):
        screen.blit(font.render(item, True, C_TEXT), (12 + i * 148, hy + 34))

    screen.blit(
        font.render("Q / ESC to quit", True, (90, 115, 100)),
        (12, hy + 56),
    )

def main():
    pygame.init()
    pygame.display.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Inzozi Smart Glasses — Random Agent Demo")
    clock  = pygame.time.Clock()

    try:
        font     = pygame.font.SysFont("DejaVuSans", 12)
        font_big = pygame.font.SysFont("DejaVuSans", 14, bold=True)
    except Exception:
        font     = pygame.font.Font(None, 13)
        font_big = pygame.font.Font(None, 15)

    font_small = pygame.font.SysFont("DejaVuSans", 9) if pygame.font.get_init() else font

    env = InzoziGlassesEnv(render_mode=None, n_motos=6,
                           n_pedestrians=10, n_static=7)
    bg  = make_background(font_small)

    episode     = 0
    obs, info   = env.reset(seed=0)
    total_r     = 0.0
    last_action = 0
    last_reward = 0.0
    running     = True

    while running and episode < MAX_EPISODES:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        total_r    += reward
        last_action = action
        last_reward = reward

        scene = pygame.Surface((SCENE_W, SCENE_H))
        scene.blit(bg, (0, 0))
        draw_goal(scene, env.goal_x, env.goal_y)
        draw_raycasts(scene, env.agent_x, env.agent_y, env.heading)
        draw_obstacles(scene, env.obstacles)
        draw_agent(scene, env.agent_x, env.agent_y, env.heading)

        screen.fill(C_BG)
        screen.blit(scene, (0, 0))
        draw_panel(screen, obs, last_action, last_reward,
                   episode + 1, env.step_count, total_r, font, font_big)
        draw_hud(screen, episode + 1, env.step_count, total_r,
                 info, font, font_big)
        pygame.display.flip()
        clock.tick(FPS)

        if term or trunc:
            episode += 1
            total_r  = 0.0
            if episode < MAX_EPISODES:
                obs, info = env.reset()
            time.sleep(0.25)

    env.close()
    pygame.quit()
    print(f"Random agent demo finished. Episodes: {episode}")

if __name__ == "__main__":
    main()