"""Kinematic car state and axis-aligned collision geometry."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Car:
    x: float
    y: float
    width: float
    length: float
    speed: float = 0.0
    heading: float = -np.pi / 2.0

    def update(self, steering: float, acceleration: float, dt: float, config) -> None:
        acceleration = float(np.clip(acceleration, -1.0, 1.0))
        steering = float(np.clip(steering, -1.0, 1.0))
        left_limit = config.road_left + config.boundary_margin
        right_limit = config.road_right - config.boundary_margin
        if self.x < left_limit:
            steering = max(steering, config.boundary_steering_strength)
        elif self.x > right_limit:
            steering = min(steering, -config.boundary_steering_strength)
        if acceleration >= 0.0:
            self.speed += acceleration * config.acceleration_rate * dt
        else:
            self.speed += acceleration * config.braking_rate * dt
        self.speed = float(np.clip(self.speed * config.drag ** (dt * 60.0), 0.0, config.max_speed))
        self.heading += steering * config.steering_rate * dt
        self.heading = float(np.clip(self.heading, -0.55, 0.55))
        self.x += np.sin(self.heading) * self.speed * dt
        self.y -= np.cos(self.heading) * self.speed * dt
        self.x = float(np.clip(self.x, config.road_left + self.width / 2.0,
                               config.road_right - self.width / 2.0))

    def bounds(self) -> tuple[float, float, float, float]:
        half_w, half_l = self.width / 2.0, self.length / 2.0
        return (self.x - half_w, self.y - half_l, self.x + half_w, self.y + half_l)


def rectangles_overlap(first: Car, second: Car) -> bool:
    left_a, top_a, right_a, bottom_a = first.bounds()
    left_b, top_b, right_b, bottom_b = second.bounds()
    return left_a < right_b and right_a > left_b and top_a < bottom_b and bottom_a > top_b
