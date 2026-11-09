"""Raycast sensors in the vehicle's local forward frame."""

from __future__ import annotations

import math

import numpy as np


def raycast_observation(player, traffic, config) -> np.ndarray:
    """Return normalized nearest-obstacle distances for configured rays."""
    distances = []
    for angle in config.sensor_angles:
        direction_x = math.sin(player.heading + angle)
        direction_y = -math.cos(player.heading + angle)
        nearest = config.sensor_range
        candidates = [(config.road_left, None), (config.road_right, None)]
        for npc in traffic:
            relative_x = npc.x - player.x
            relative_y = npc.y - player.y
            forward = relative_x * direction_x + relative_y * direction_y
            lateral = abs(relative_x * direction_y - relative_y * direction_x)
            if forward > 0.0 and lateral < npc.width * 0.75:
                candidates.append((forward, npc))
        for distance, obstacle in candidates:
            if obstacle is None:
                # The boundary approximation uses the ray's horizontal travel.
                if abs(direction_x) > 1e-6:
                    boundary_distance = (distance - player.x) / direction_x
                    if boundary_distance > 0.0:
                        nearest = min(nearest, boundary_distance)
            elif distance > 0.0:
                nearest = min(nearest, distance)
        distances.append(np.clip(nearest / config.sensor_range, 0.0, 1.0))
    speed_ratio = np.clip(player.speed / config.max_speed, 0.0, 1.0)
    return np.asarray([*distances, speed_ratio], dtype=np.float64)
