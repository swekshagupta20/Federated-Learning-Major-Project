from stable_baselines3 import PPO
from src.rl.environment import FLEnvironment


class RLAgent:
    def __init__(self, num_clients=5):
        self.env = FLEnvironment(num_clients=num_clients)

        self.model = PPO(
            "MlpPolicy",
            self.env,
            verbose=1,
        )

    def train(self, timesteps=5000):
        print("[RL] Training PPO agent...")
        self.model.learn(total_timesteps=timesteps)
        print("[RL] Training complete.")

    def predict_weights(self, state):
        action, _ = self.model.predict(state)
        return action