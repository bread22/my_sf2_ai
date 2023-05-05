import torch
import torch.nn as nn
import numpy as np
import random
import collections
import torch.nn.functional as F


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

    def update(self, batch, optimizer, target_network, device, gamma=0.99):
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert the lists of NumPy arrays to single NumPy arrays before creating tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(actions), dtype=torch.long).to(device).view(-1, 1)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        dones = torch.tensor(np.array(dones, dtype=np.uint8), dtype=torch.float32).to(device)

        current_q_values = self(states).gather(1, actions)
        next_q_values = target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + gamma * next_q_values * (1 - dones)

        loss = F.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)