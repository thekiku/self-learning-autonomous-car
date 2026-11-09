"""Pygame visualization for a trained or manually controlled episode."""

from __future__ import annotations

import numpy as np


class PygameRenderer:
    def __init__(self, config) -> None:
        import pygame

        pygame.init()
        self.pygame = pygame
        self.config = config
        self.screen = pygame.display.set_mode((config.screen_width, config.screen_height))
        pygame.display.set_caption("Neuroevolution Autonomous Car")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)

    def draw(self, environment) -> None:
        pygame = self.pygame
        screen = self.screen
        screen.fill((38, 125, 62))
        road = pygame.Rect(
            int(self.config.road_left), 0,
            int(self.config.road_width), self.config.screen_height,
        )
        pygame.draw.rect(screen, (55, 55, 58), road)
        for lane in range(1, self.config.lane_count):
            x = int(self.config.road_left + lane * self.config.lane_width)
            pygame.draw.line(screen, (210, 210, 210), (x, 0), (x, self.config.screen_height), 2)
        self._draw_sensor_rays(environment)
        self._draw_car(environment.player, (40, 170, 255), environment.player.y)
        for npc in environment.traffic.cars:
            self._draw_car(npc, (220, 65, 60), environment.player.y)
        self._draw_hud(environment)
        pygame.display.flip()

    def _draw_sensor_rays(self, environment) -> None:
        player = environment.player
        observation = environment.observation()
        for angle, normalized_distance in zip(self.config.sensor_angles, observation[:-1]):
            distance = normalized_distance * self.config.sensor_range
            start = (int(player.x), int(self.config.screen_height * 0.75))
            end_x = player.x + np.sin(player.heading + angle) * distance
            end_y = self.config.screen_height * 0.75 - np.cos(player.heading + angle) * distance
            self.pygame.draw.line(
                self.screen, (255, 215, 70), start, (int(end_x), int(end_y)), 1
            )

    def _draw_hud(self, environment) -> None:
        pygame = self.pygame
        panel = pygame.Surface((245, 104), pygame.SRCALPHA)
        panel.fill((12, 18, 28, 220))
        lines = (
            "NEUROEVOLUTION DRIVER",
            f"Speed: {environment.player.speed:5.1f} px/s",
            f"Distance: {environment.distance:6.1f} px",
            f"Time: {environment.elapsed:5.1f} s",
        )
        for index, text in enumerate(lines):
            font = self.font if index == 0 else self.small_font
            color = (100, 220, 255) if index == 0 else (235, 240, 245)
            panel.blit(font.render(text, True, color), (12, 8 + index * 23))
        self.screen.blit(panel, (16, 16))

    def _draw_car(self, car, color, camera_y: float) -> None:
        pygame = self.pygame
        screen_y = self.config.screen_height * 0.75 + (car.y - camera_y)
        rect = pygame.Rect(
            int(car.x - car.width / 2), int(screen_y - car.length / 2),
            int(car.width), int(car.length),
        )
        pygame.draw.rect(self.screen, color, rect, border_radius=4)

    def close(self) -> None:
        self.pygame.quit()
