"""Evolutionary training loop and optional Pygame renderer."""

from __future__ import annotations

import numpy as np

from .environment import DrivingEnvironment
from .genetic_algorithm import GeneticAlgorithm
from .neural_network import NeuralNetwork


class EvolutionTrainer:
    def __init__(self, config, seed: int | None = None) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.environment = DrivingEnvironment(config, seed=seed)
        self.best_genome: np.ndarray | None = None

        def network_factory() -> NeuralNetwork:
            return NeuralNetwork(len(config.sensor_angles) + 1, config.hidden_layers, 2, self.rng)

        self.ga = GeneticAlgorithm(
            config.population_size, config.elite_count, config.tournament_size,
            config.mutation_rate, config.mutation_scale, network_factory, self.rng,
        )
        self.ga.seed_safe_individual()

    def evaluate(self, individual) -> float:
        network = NeuralNetwork(
            len(self.config.sensor_angles) + 1, self.config.hidden_layers, 2,
            self.rng, genome=individual.genome,
        )
        observation = self.environment.reset()
        total = 0.0
        while not self.environment.done:
            observation, reward, _ = self.environment.step(network.predict(observation))
            total += reward
        return total

    def train(self, generations: int, progress: bool = True) -> list[float]:
        best_scores = []
        for generation in range(generations):
            for individual in self.ga.population:
                individual.fitness = self.evaluate(individual)
            best = self.ga.ranked()[0]
            self.best_genome = best.genome.copy()
            best_scores.append(best.fitness)
            if progress:
                print(f"generation={generation + 1} best_fitness={best.fitness:.2f}")
            self.ga.evolve()
        return best_scores

    def watch(self, genome: np.ndarray) -> None:
        """Render one episode using a supplied policy genome."""
        from .renderer import PygameRenderer

        renderer = PygameRenderer(self.config)
        network = NeuralNetwork(
            len(self.config.sensor_angles) + 1, self.config.hidden_layers, 2,
            self.rng, genome=genome,
        )
        try:
            while True:
                observation = self.environment.reset()
                while not self.environment.done:
                    for event in renderer.pygame.event.get():
                        if event.type == renderer.pygame.QUIT:
                            return
                    observation, _, _ = self.environment.step(network.predict(observation))
                    renderer.draw(self.environment)
                    renderer.clock.tick(60)
        finally:
            renderer.close()
