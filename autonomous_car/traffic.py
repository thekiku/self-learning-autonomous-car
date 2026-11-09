"""Domain-randomized NPC traffic."""

from dataclasses import dataclass

import numpy as np

from .physics import Car


@dataclass
class TrafficCar(Car):
    lane: int = 0
    target_lane: int = 0
    lane_change_timer: float = 0.0


class TrafficManager:
    def __init__(self, config, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng
        self.cars: list[TrafficCar] = []

    def reset(self, player_y: float) -> None:
        self.cars = []
        for index in range(self.config.traffic_count):
            lane = int(self.rng.integers(self.config.lane_count))
            y = player_y - 260.0 - index * 120.0 - self.rng.uniform(0.0, 600.0)
            self.cars.append(self._new_car(lane, y))

    def _lane_x(self, lane: int) -> float:
        return self.config.road_left + (lane + 0.5) * self.config.lane_width

    def _new_car(self, lane: int, y: float) -> TrafficCar:
        return TrafficCar(
            self._lane_x(lane), y, self.config.car_width, self.config.car_length,
            speed=float(self.rng.uniform(self.config.traffic_speed_min, self.config.traffic_speed_max)),
            lane=lane, target_lane=lane,
            lane_change_timer=float(self.rng.uniform(1.0, 4.0)),
        )

    def update(self, dt: float, player_y: float) -> None:
        for car in self.cars:
            car.y -= car.speed * dt
            car.lane_change_timer -= dt
            if car.lane_change_timer <= 0.0:
                car.lane_change_timer = float(self.rng.uniform(1.5, 5.0))
                if self.rng.random() < self.config.lane_change_probability:
                    options = [lane for lane in (car.lane - 1, car.lane + 1)
                               if 0 <= lane < self.config.lane_count]
                    if options:
                        car.target_lane = int(self.rng.choice(options))
            target_x = self._lane_x(car.target_lane)
            car.x += np.clip(target_x - car.x, -75.0 * dt, 75.0 * dt)
            if car.y < player_y - 800.0:
                lane = int(self.rng.integers(self.config.lane_count))
                car.x = self._lane_x(lane)
                car.y = player_y + 500.0 + self.rng.uniform(0.0, 500.0)
                car.lane = car.target_lane = lane
