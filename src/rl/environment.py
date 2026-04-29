import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FLEnvironment(gym.Env):
    """
    RL Environment for Federated Learning

    State:
        [avg_battery, avg_latency, avg_reliability, accuracy]

    Action:
        Weight vector for clients (simplified)

    Reward:
        Accuracy - energy penalty + fairness
    """

    def __init__(self, num_clients=5):
        super(FLEnvironment, self).__init__()

        self.num_clients = num_clients

        # State space: 4 features
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

        # Action space: weights per client
        self.action_space = spaces.Box(low=0, high=1, shape=(num_clients,), dtype=np.float32)

        self.state = None

    def reset(self, seed=None, options=None):
        self.state = np.random.rand(4)
        return self.state, {}

    def step(self, action):
        # Normalize action (weights)
        action = action / (np.sum(action) + 1e-8)

        # Simulate environment response
        accuracy = np.random.uniform(0.5, 0.9)
        energy_cost = np.mean(action) * 0.5
        fairness = 1.0 - np.std(action)

        reward = accuracy - energy_cost + fairness

        self.state = np.array([
            np.random.rand(),  # battery
            np.random.rand(),  # latency
            np.random.rand(),  # reliability
            accuracy
        ])

        done = False
        return self.state, reward, done, False, {}