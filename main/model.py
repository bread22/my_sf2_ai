import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym
import retro


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, feature_dim: int = 128):
        super().__init__(observation_space, feature_dim)
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.Flatten()
        )

        # Calculate the output shape of the CNN by doing a forward pass with a dummy input
        with torch.no_grad():
            dummy_input = torch.zeros(1, *observation_space.shape)
            cnn_output_shape = self.cnn(dummy_input).shape[-1]

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(cnn_output_shape, feature_dim),
            torch.nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        features = self.cnn(observations)
        return self.linear(features)

