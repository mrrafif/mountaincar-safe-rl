"""Environment wrappers for MountainCar experiments.

Provides two wrappers:
- ShapedMountainCar: reward shaping + completion bonus (used by DQN and PPO).
- ConstrainedMountainCar: same shaping + a proportional soft safety cost
  emitted via info['cost'], used for Safe PPO (CMDP) experiments.
"""
from __future__ import annotations

import gymnasium as gym


class ShapedMountainCar(gym.Wrapper):
    """MountainCar with potential-style reward shaping and a completion bonus.

    The shaping uses absolute deviation from the valley bottom (-0.5) scaled by
    1.5; reaching the flag adds a 100.0 bonus. No safety constraints.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        height_bonus = abs(obs[0] - (-0.5)) * 1.5
        shaped_reward = reward + height_bonus
        if terminated:
            shaped_reward += 100.0
        return obs, shaped_reward, terminated, truncated, info


class ConstrainedMountainCar(gym.Wrapper):
    """CMDP version: ShapedMountainCar plus a proportional speeding cost.

    The cost is emitted in info['cost'] and is zero unless |velocity| exceeds
    ``max_speed``, in which case it grows linearly with the excess (scale 100).
    """

    def __init__(self, env: gym.Env, max_speed: float = 0.04):
        super().__init__(env)
        self.max_speed = max_speed

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        velocity = abs(obs[1])

        if velocity > self.max_speed:
            cost = (velocity - self.max_speed) * 100.0
        else:
            cost = 0.0

        height_bonus = abs(obs[0] - (-0.5)) * 1.5
        shaped_reward = reward + height_bonus
        if terminated:
            shaped_reward += 100.0

        info["cost"] = cost
        return obs, shaped_reward, terminated, truncated, info
