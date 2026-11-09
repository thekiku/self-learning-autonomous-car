"""Genetic operators for optimizing neural-network policy genomes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .neural_network import NeuralNetwork


@dataclass
class Individual:
    genome: np.ndarray
    fitness: float = 0.0


class GeneticAlgorithm:
    def __init__(
        self,
        population_size: int,
        elite_count: int,
        tournament_size: int,
        mutation_rate: float,
        mutation_scale: float,
        network_factory,
        rng: np.random.Generator,
    ) -> None:
        self.population_size = population_size
        self.elite_count = elite_count
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.network_factory = network_factory
        self.rng = rng
        template = network_factory()
        self.population = [Individual(template.genome().copy())]
        self.population.extend(
            Individual(network_factory().genome()) for _ in range(population_size - 1)
        )

    def ranked(self) -> list[Individual]:
        """Return individuals ordered from highest to lowest measured fitness."""
        return sorted(self.population, key=lambda item: item.fitness, reverse=True)

    def _select_parent(self) -> Individual:
        candidates = self.rng.choice(self.population, self.tournament_size, replace=False)
        return max(candidates, key=lambda item: item.fitness)

    def _crossover(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        # Uniform crossover preserves independent parameter material from both
        # parents without assuming that neighboring genes are correlated.
        mask = self.rng.random(first.size) < 0.5
        return np.where(mask, first, second).copy()

    def _mutate(self, genome: np.ndarray) -> np.ndarray:
        mask = self.rng.random(genome.size) < self.mutation_rate
        # Independent zero-mean Gaussian perturbations provide local search
        # while the rate controls how many parameters are changed.
        genome[mask] += self.rng.normal(0.0, self.mutation_scale, mask.sum())
        return genome

    def seed_safe_individual(self) -> None:
        """Insert a straight, accelerating policy to anchor early evolution."""
        network = self.network_factory()
        for layer in network.layers:
            layer.weights.fill(0.0)
            layer.bias.fill(0.0)
        network.layers[-1].bias[1] = 1.5
        self.population[0] = Individual(network.genome())

    def evolve(self) -> Individual:
        """Create the next generation while preserving the current champion."""
        ranked = self.ranked()
        # Elitism prevents the best policy from being lost by stochastic
        # crossover or mutation.
        next_population = [Individual(item.genome.copy()) for item in ranked[: self.elite_count]]
        while len(next_population) < self.population_size:
            first = self._select_parent().genome
            second = self._select_parent().genome
            child = self._mutate(self._crossover(first, second))
            next_population.append(Individual(child))
        self.population = next_population
        return ranked[0]
