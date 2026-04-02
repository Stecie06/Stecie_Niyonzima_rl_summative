import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, List

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from data.at_dataset import AT_WEIGHT
except ImportError:
    AT_WEIGHT = 0.8275   

WORLD_SIZE     = 20.0   
MAX_STEPS      = 500
AGENT_SPEED    = 0.25   
SLOW_SPEED     = 0.12   
MOTO_SPEED     = 0.35
PED_SPEED      = 0.15
COLLISION_R    = 0.6    
GOAL_R         = 0.8    
MAX_ASSISTS    = 3      

HILL_BANDS = [
    (4.0,  8.0,  0.12),
    (12.0, 16.0, 0.18),
]

ACTIONS = {
    0: "TURN_LEFT",
    1: "TURN_RIGHT",
    2: "FORWARD_SLOW",
    3: "FORWARD_NORMAL",
    4: "STOP",
    5: "WARN_OBSTACLE_LEFT",
    6: "WARN_OBSTACLE_RIGHT",
    7: "BROADCAST_MOTO_DANGER",
    8: "REQUEST_HUMAN_ASSIST",
    9: "RECENTER_TO_GOAL",
}
N_ACTIONS = len(ACTIONS)

class Obstacle:
    def __init__(self, x, y, vx=0, vy=0, kind="static", w=1.0, h=1.0):
        self.x    = x
        self.y    = y
        self.vx   = vx
        self.vy   = vy
        self.kind = kind   
        self.w    = w
        self.h    = h

    def step(self):
        self.x = (self.x + self.vx) % WORLD_SIZE
        self.y = (self.y + self.vy) % WORLD_SIZE

    def dist_to(self, ax, ay) -> float:
        return math.hypot(self.x - ax, self.y - ay)

class InzoziGlassesEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        n_motos: int = 5,
        n_pedestrians: int = 8,
        n_static: int = 6,
    ):
        super().__init__()
        self.render_mode   = render_mode
        self.n_motos       = n_motos
        self.n_pedestrians = n_pedestrians
        self.n_static      = n_static

        # Spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(13,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        self.agent_x     = 0.0
        self.agent_y     = 0.0
        self.heading     = 0.0
        self.speed       = 0.0
        self.goal_x      = 0.0
        self.goal_y      = 0.0
        self.obstacles:  List[Obstacle] = []
        self.step_count  = 0
        self.assist_used = 0
        self.last_5_acts: List[int] = []
        self._done       = False
        self.window = None
        self.clock  = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        self.agent_x = float(rng.uniform(1.5, 4.0))
        self.agent_y = float(rng.uniform(1.5, 4.0))
        self.heading = float(rng.uniform(0, 2 * math.pi))
        self.speed   = 0.0

        self.goal_x  = float(rng.uniform(WORLD_SIZE - 4.5, WORLD_SIZE - 1.5))
        self.goal_y  = float(rng.uniform(WORLD_SIZE - 4.5, WORLD_SIZE - 1.5))

        self.obstacles = []
        for _ in range(self.n_motos):
            a  = rng.uniform(0, 2 * math.pi)
            sp = MOTO_SPEED * rng.uniform(0.7, 1.4)
            self.obstacles.append(Obstacle(
                x=float(rng.uniform(0, WORLD_SIZE)),
                y=float(rng.uniform(0, WORLD_SIZE)),
                vx=math.cos(a) * sp,
                vy=math.sin(a) * sp,
                kind="moto", w=0.9, h=0.5,
            ))
        for _ in range(self.n_pedestrians):
            a  = rng.uniform(0, 2 * math.pi)
            sp = PED_SPEED * rng.uniform(0.5, 1.5)
            self.obstacles.append(Obstacle(
                x=float(rng.uniform(0, WORLD_SIZE)),
                y=float(rng.uniform(0, WORLD_SIZE)),
                vx=math.cos(a) * sp,
                vy=math.sin(a) * sp,
                kind="pedestrian", w=0.5, h=0.5,
            ))
        for _ in range(self.n_static):
            self.obstacles.append(Obstacle(
                x=float(rng.uniform(3, WORLD_SIZE - 3)),
                y=float(rng.uniform(3, WORLD_SIZE - 3)),
                kind="static", w=1.2, h=1.2,
            ))

        self.step_count  = 0
        self.assist_used = 0
        self.last_5_acts = []
        self._done       = False

        return self._get_obs(), {"at_weight": AT_WEIGHT}

    def step(self, action: int):
        assert not self._done, "Call reset() first — episode already ended."
        self.step_count += 1
        action = int(action)
        self.last_5_acts.append(action)
        if len(self.last_5_acts) > 5:
            self.last_5_acts.pop(0)

        reward     = 0.0
        terminated = False
        truncated  = False

        prev_dist = math.hypot(self.agent_x - self.goal_x,
                               self.agent_y - self.goal_y)

        name = ACTIONS[action]

        if name == "TURN_LEFT":
            self.heading -= math.pi / 8

        elif name == "TURN_RIGHT":
            self.heading += math.pi / 8

        elif name == "FORWARD_SLOW":
            self.speed = SLOW_SPEED
            self.agent_x += math.cos(self.heading) * self.speed
            self.agent_y += math.sin(self.heading) * self.speed

        elif name == "FORWARD_NORMAL":
            self.speed = AGENT_SPEED
            self.agent_x += math.cos(self.heading) * self.speed
            self.agent_y += math.sin(self.heading) * self.speed

        elif name == "STOP":
            self.speed = 0.0

        elif name == "WARN_OBSTACLE_LEFT":
            reward += self._warn_reward("left")

        elif name == "WARN_OBSTACLE_RIGHT":
            reward += self._warn_reward("right")

        elif name == "BROADCAST_MOTO_DANGER":
            moto_close = any(
                o.kind == "moto" and o.dist_to(self.agent_x, self.agent_y) < 2.0
                for o in self.obstacles
            )
            reward += 5.0 if moto_close else -3.0

        elif name == "REQUEST_HUMAN_ASSIST":
            self.assist_used += 1
            reward += 0.5 if self.assist_used <= MAX_ASSISTS else -5.0

        elif name == "RECENTER_TO_GOAL":
            self.heading = math.atan2(
                self.goal_y - self.agent_y,
                self.goal_x - self.agent_x,
            )
            reward += 0.8

        self.agent_x = float(np.clip(self.agent_x, 0.05, WORLD_SIZE - 0.05))
        self.agent_y = float(np.clip(self.agent_y, 0.05, WORLD_SIZE - 0.05))

        for obs in self.obstacles:
            obs.step()

        new_dist = math.hypot(self.agent_x - self.goal_x,
                              self.agent_y - self.goal_y)
        min_d    = self._min_obs_dist()
        slope    = self._slope_at(self.agent_y)

        if name in ("FORWARD_SLOW", "FORWARD_NORMAL"):
            reward += 0.4 if new_dist < prev_dist else -0.15

        if name in ("FORWARD_SLOW", "FORWARD_NORMAL"):
            if min_d > 1.5:
                reward += 1.0
            elif min_d > 0.9:
                reward += 0.3

        td = self._traffic_density()
        cd = self._crowd_density()
        if name == "FORWARD_NORMAL" and td < 0.3 and cd < 0.3:
            reward += 0.5

        if slope > 0.1 and name in ("TURN_LEFT", "TURN_RIGHT", "FORWARD_SLOW"):
            reward += 3.0

        turns = sum(1 for a in self.last_5_acts
                    if ACTIONS[a] in ("TURN_LEFT", "TURN_RIGHT"))
        if turns >= 4:
            reward -= 1.0

        if name == "STOP" and min_d > 3.0:
            reward -= 0.5

        moto_very_close = any(
            o.kind == "moto" and o.dist_to(self.agent_x, self.agent_y) < 1.5
            for o in self.obstacles
        )
        if moto_very_close and name not in (
            "STOP", "WARN_OBSTACLE_LEFT", "WARN_OBSTACLE_RIGHT",
            "BROADCAST_MOTO_DANGER", "FORWARD_SLOW",
        ):
            reward -= 2.0

        if AT_WEIGHT < 0.7 and reward < 0:
            reward *= 1.2

        result = None

        if new_dist < GOAL_R:
            reward += 10.0
            terminated = True
            result = "goal_reached"

        elif min_d < COLLISION_R:
            reward -= 10.0
            terminated = True
            result = "collision"

        elif (self.agent_x <= 0.06 or self.agent_x >= WORLD_SIZE - 0.06 or
              self.agent_y <= 0.06 or self.agent_y >= WORLD_SIZE - 0.06):
            reward -= 15.0
            terminated = True
            result = "fell_off_edge"

        elif self.assist_used > MAX_ASSISTS + 2:
            terminated = True
            result = "too_many_assists"

        elif self.step_count >= MAX_STEPS:
            reward -= 8.0
            truncated = True
            result = "timeout"

        self._done = terminated or truncated
        obs = self._get_obs()
        info = {
            "step":          self.step_count,
            "dist_to_goal":  new_dist,
            "action_name":   ACTIONS[action],
            "at_weight":     AT_WEIGHT,
            "result":        result,
        }
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        rf, rl, rr = self._raycasts()
        td    = self._traffic_density()
        cd    = self._crowd_density()
        slope = self._slope_at(self.agent_y)

        dx = self.goal_x - self.agent_x
        dy = self.goal_y - self.agent_y
        goal_angle = ((math.atan2(dy, dx) - self.heading) % (2 * math.pi)) / (2 * math.pi)
        goal_dist  = math.hypot(dx, dy) / (WORLD_SIZE * math.sqrt(2))

        return np.array([
            rf,
            rl,
            rr,
            td,
            cd,
            float(np.clip(slope / 0.3, 0, 1)),
            goal_angle,
            float(np.clip(goal_dist, 0, 1)),
            float(np.clip(self.speed / AGENT_SPEED, 0, 1)),
            float(AT_WEIGHT),
            (self.heading % (2 * math.pi)) / (2 * math.pi),
            self.step_count / MAX_STEPS,
            self.assist_used / (MAX_ASSISTS + 3),
        ], dtype=np.float32)

    def _raycasts(self, length=5.0):
        offsets = {"front": 0.0, "left": -math.pi/4, "right": math.pi/4}
        results = {}
        for name, off in offsets.items():
            ang   = self.heading + off
            dx    = math.cos(ang)
            dy    = math.sin(ang)
            min_t = length
            for obs in self.obstacles:
                for t in np.linspace(0.1, length, 25):
                    rx = self.agent_x + dx * t
                    ry = self.agent_y + dy * t
                    if abs(rx - obs.x) < obs.w/2 and abs(ry - obs.y) < obs.h/2:
                        min_t = min(min_t, t)
                        break
            results[name] = 1.0 - float(np.clip(min_t / length, 0, 1))
        return results["front"], results["left"], results["right"]

    def _traffic_density(self, r=4.0) -> float:
        n = sum(1 for o in self.obstacles
                if o.kind == "moto" and o.dist_to(self.agent_x, self.agent_y) < r)
        return float(np.clip(n / max(self.n_motos, 1), 0, 1))

    def _crowd_density(self, r=4.0) -> float:
        n = sum(1 for o in self.obstacles
                if o.kind == "pedestrian" and o.dist_to(self.agent_x, self.agent_y) < r)
        return float(np.clip(n / max(self.n_pedestrians, 1), 0, 1))

    def _min_obs_dist(self) -> float:
        if not self.obstacles:
            return 999.0
        return min(o.dist_to(self.agent_x, self.agent_y) for o in self.obstacles)

    def _slope_at(self, y: float) -> float:
        for y0, y1, g in HILL_BANDS:
            if y0 <= y <= y1:
                return g
        return 0.0

    def _warn_reward(self, side: str) -> float:
        off = -math.pi/4 if side == "left" else math.pi/4
        ang = self.heading + off
        dx, dy = math.cos(ang), math.sin(ang)
        for t in np.linspace(0.1, 3.0, 18):
            rx = self.agent_x + dx * t
            ry = self.agent_y + dy * t
            for obs in self.obstacles:
                if abs(rx - obs.x) < obs.w and abs(ry - obs.y) < obs.h:
                    return 5.0   
        return -3.0              

    def render(self):
        pass

    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.window = None


if __name__ == "__main__":
    env = InzoziGlassesEnv()
    obs, info = env.reset(seed=42)
    print("Obs space:", env.observation_space)
    print("Act space:", env.action_space)
    print("AT weight:", AT_WEIGHT)
    print("First obs:", obs)
    for i in range(6):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        print(f"  step {i+1}: {ACTIONS[a]:<26} r={r:+.2f}  done={term or trunc}")
        if term or trunc:
            obs, info = env.reset()
    print("custom_env.py — OK")