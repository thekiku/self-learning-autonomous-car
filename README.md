# Neuroevolution Autonomous Car

A from-scratch 2D highway simulator using NumPy for the multilayer perceptron and genetic algorithm, and Pygame for optional rendering.

## Run

```powershell
python -m pip install -r requirements.txt
python -m autonomous_car.main --population 80 --generations 100
```

Run the trained policy in the animated Pygame frontend:

```powershell
python -m autonomous_car.main --population 80 --generations 100 --watch
```

The window displays the highway, randomized traffic, the learned car, raycast
sensors, speed, distance, and episode time. Close the window to stop playback.

For a faster headless smoke test:

```powershell
python -m autonomous_car.main --population 20 --generations 2 --headless
```

The policy receives normalized raycast distances and normalized speed. It outputs tanh-bounded steering and acceleration commands. Each generation uses a new randomized traffic seed, while elite preservation and tournament selection retain useful policies.

## Architecture

- `config.py`: immutable simulation and evolution parameters.
- `neural_network.py`: NumPy-only MLP with flat genome serialization.
- `genetic_algorithm.py`: elitism, tournament selection, uniform crossover, and Gaussian mutation.
- `physics.py`: car state integration and collision geometry.
- `sensors.py`: local-frame raycasts against road boundaries and traffic.
- `traffic.py`: randomized NPC traffic and lane changes.
- `environment.py`: episode lifecycle, observations, rewards, and termination.
- `simulation.py`: generation training loop and optional rendering.
- `main.py`: command-line entry point.
