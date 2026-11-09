"""Command-line entry point for training the autonomous car."""

import argparse

from .config import SimulationConfig
from .simulation import EvolutionTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a neuroevolution highway driver.")
    parser.add_argument("--population", type=int, default=80)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--headless", action="store_true", help="Train without opening the frontend.")
    parser.add_argument("--watch", action="store_true", help="Render the best policy after training.")
    args = parser.parse_args()
    config = SimulationConfig(population_size=args.population)
    trainer = EvolutionTrainer(config, seed=args.seed)
    trainer.train(args.generations)
    if args.watch and not args.headless and trainer.best_genome is not None:
        trainer.watch(trainer.best_genome)


if __name__ == "__main__":
    main()
