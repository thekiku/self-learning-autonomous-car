# Neuroevolution Autonomous Car

A from-scratch 2D highway simulator using NumPy for the multilayer perceptron and genetic algorithm, and Pygame for optional rendering.

## Run on Windows

Open PowerShell and go to the project folder:

```powershell
cd "C:\Users\pasha\Documents\projects\Self-Learning Autonomous Car"
```

Confirm Python is available:

```powershell
python --version
```

Install NumPy and Pygame:

```powershell
python -m pip install -r requirements.txt
```

### 1. Train without opening a window

This runs the backend simulation and genetic algorithm in the terminal:

```powershell
python -m autonomous_car.main --population 80 --generations 100 --headless
```

- `--population 80` evaluates 80 neural-network cars per generation.
- `--generations 100` repeats selection, crossover, and mutation 100 times.
- `--headless` prevents Pygame from opening a window.

### 2. Train and watch the best car

This trains first, then opens the animated frontend:

```powershell
python -m autonomous_car.main --population 80 --generations 100 --watch
```

Use this command by itself; do not add `--headless`. The terminal prints each
generation's fitness first. After training completes, it prints
`Training complete. Opening Pygame frontend` and opens a separate window. The
Pygame window shows the highway,
randomized traffic, learned car, raycast sensors, speed, distance, and episode
time. The playback loops through new randomized episodes until the window is
closed.

### 3. Watch quickly while developing

Use a small run to open the frontend sooner:

```powershell
python -m autonomous_car.main --population 20 --generations 10 --watch
```

This is useful for checking the animation, but larger populations and more
generations generally produce better driving behavior.

For an immediate frontend test, use one population and one generation:

```powershell
python -m autonomous_car.main --population 2 --generations 1 --watch
```

### 4. Stop the frontend

Close the Pygame window normally. If the terminal process is still running,
press `Ctrl+C` in PowerShell.

### 5. Run a quick installation smoke test

Use this to verify the backend without waiting for a full training session:

```powershell
python -m autonomous_car.main --population 20 --generations 2 --headless
```

Expected output resembles:

```text
generation=1 best_fitness=...
generation=2 best_fitness=...
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
