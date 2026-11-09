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

        # Intersect the ray with each vertical road edge. The ray parameter
        # t is distance along a unit direction vector, so t is directly
        # comparable with the configured sensor range.
        if abs(direction_x) > 1e-8:
            for boundary_x in (config.road_left, config.road_right):
                boundary_distance = (boundary_x - player.x) / direction_x
                if 0.0 < boundary_distance < nearest:
                    nearest = boundary_distance

        for npc in traffic:
            relative_x = npc.x - player.x
            relative_y = npc.y - player.y
            forward = relative_x * direction_x + relative_y * direction_y
            lateral = abs(relative_x * direction_y - relative_y * direction_x)
            # This conservative capsule approximation avoids expensive
            # per-pixel collision tests while accounting for vehicle width.
            obstacle_radius = max(npc.width, npc.length) * 0.5
            if 0.0 < forward < nearest and lateral < obstacle_radius:
                nearest = forward
        distances.append(np.clip(nearest / config.sensor_range, 0.0, 1.0))
    speed_ratio = np.clip(player.speed / config.max_speed, 0.0, 1.0)
    return np.asarray([*distances, speed_ratio], dtype=np.float64)
