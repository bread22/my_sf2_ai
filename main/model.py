import torch
import torch.nn as nn
import numpy as np
import random

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class StreetFighterAgent(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(StreetFighterAgent, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

    def act(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randrange(self.fc[-1].out_features)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                q_values = self.forward(state)
                return torch.argmax(q_values).item()