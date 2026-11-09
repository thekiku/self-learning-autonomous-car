"""Configuration values for the driving environment and evolutionary trainer."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class SimulationConfig:
    screen_width: int = 960
    screen_height: int = 640
    road_width: float = 420.0
    lane_count: int = 3
    car_width: float = 24.0
    car_length: float = 42.0
    max_speed: float = 260.0
    acceleration_rate: float = 180.0
    braking_rate: float = 260.0
    drag: float = 0.985
    steering_rate: float = 2.4
    sensor_range: float = 260.0
    sensor_angles: Tuple[float, ...] = (-0.85, -0.42, 0.0, 0.42, 0.85)
    episode_seconds: float = 25.0
    traffic_count: int = 12
    traffic_speed_min: float = 70.0
    traffic_speed_max: float = 190.0
    lane_change_probability: float = 0.12
    population_size: int = 80
    elite_count: int = 8
    tournament_size: int = 4
    mutation_rate: float = 0.08
    mutation_scale: float = 0.12
    hidden_layers: Tuple[int, ...] = field(default_factory=lambda: (16, 12))

    @property
    def lane_width(self) -> float:
        return self.road_width / self.lane_count

    @property
    def road_left(self) -> float:
        return (self.screen_width - self.road_width) / 2.0

    @property
    def road_right(self) -> float:
        return self.road_left + self.road_width
