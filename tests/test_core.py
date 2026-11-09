import numpy as np

from autonomous_car.config import SimulationConfig
from autonomous_car.environment import DrivingEnvironment
from autonomous_car.neural_network import NeuralNetwork
from autonomous_car.physics import Car


def test_network_output_is_bounded():
    network = NeuralNetwork(6, (8, 6), 2, np.random.default_rng(1))
    output = network.predict(np.ones(6))
    assert output.shape == (2,)
    assert np.all(output >= -1.0) and np.all(output <= 1.0)


def test_environment_step_returns_valid_observation():
    environment = DrivingEnvironment(SimulationConfig(episode_seconds=1.0), seed=2)
    observation = environment.reset()
    next_observation, reward, done = environment.step(np.zeros(2))
    assert observation.shape == (6,)
    assert next_observation.shape == (6,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)


def test_straight_action_does_not_oscillate():
    config = SimulationConfig(traffic_count=0)
    car = Car(config.screen_width / 2, 480, config.car_width, config.car_length)
    positions = []
    for _ in range(120):
        car.update(0.0, 1.0, 1.0 / 30.0, config)
        positions.append(car.x)
    assert max(positions) - min(positions) < 1e-6
