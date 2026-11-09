"""Headless driving environment suitable for training and testing."""

from __future__ import annotations

import numpy as np

from .physics import Car, rectangles_overlap
from .sensors import raycast_observation
from .traffic import TrafficManager


class DrivingEnvironment:
    def __init__(self, config, seed: int | None = None) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.traffic = TrafficManager(config, self.rng)
        self.player: Car
        self.elapsed = 0.0
        self.distance = 0.0
        self.done = False

    def reset(self) -> np.ndarray:
        self.player = Car(
            self.config.screen_width / 2.0, self.config.screen_height * 0.75,
            self.config.car_width, self.config.car_length,
        )
        self.traffic.reset(self.player.y)
        self.elapsed = self.distance = 0.0
        self.done = False
        return self.observation()

    def observation(self) -> np.ndarray:
        return raycast_observation(self.player, self.traffic.cars, self.config)

    def step(self, action: np.ndarray, dt: float = 1.0 / 30.0) -> tuple[np.ndarray, float, bool]:
        if self.done:
            raise RuntimeError("Cannot step a terminated episode; call reset first.")
        previous_y = self.player.y
        self.player.update(action[0], action[1], dt, self.config)
        self.traffic.update(dt, self.player.y)
        self.elapsed += dt
        self.distance += previous_y - self.player.y
        collision = any(rectangles_overlap(self.player, npc) for npc in self.traffic.cars)
        outside = not (self.config.road_left <= self.player.x <= self.config.road_right)
        self.done = collision or outside or self.elapsed >= self.config.episode_seconds
        reward = (previous_y - self.player.y) * 0.05
        if collision:
            reward -= 100.0
        if outside:
            reward -= 25.0
        return self.observation(), float(reward), self.done
