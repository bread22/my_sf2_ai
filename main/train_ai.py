import retro
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from model import DQN
import random
import numpy as np


def train():
    # Environment
    game = 'StreetFighterIISpecialChampionEdition-Genesis'
    state = 'Champion.Level1.RyuVsGuile'
    env = retro.make(game, state, record='.')
    env.reset()

    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    input_shape = env.observation_space.shape
    num_actions = env.action_space.n
    policy_net = DQN(input_shape, num_actions).to(device)
    target_net = DQN(input_shape, num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Training
    optimizer = optim.Adam(policy_net.parameters(), lr=0.00025)
    memory = deque(maxlen=10000)
    batch_size = 32
    gamma = 0.95
    eps_start = 1
    eps_end = 0.1
    eps_decay = 1000000
    target_update = 10
    num_episodes = 10
    for i_episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        while not done:
            epsilon = eps_end + (eps_start - eps_end) * \
                np.exp(-1. * len(memory) / eps_decay)
            action = env.action_space.sample() if np.random.uniform(
                0, 1) < epsilon else policy_net(
                torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)).argmax(dim=1).item()

            next_obs, reward, done, info = env.step(action)
            total_reward += reward
            memory.append((obs, action, next_obs, reward, done))

            obs = next_obs

            if len(memory) < batch_size:
                continue

            batch = random.sample(memory, batch_size)
            obs_batch, action_batch, next_obs_batch, reward_batch, done_batch = map(
                np.array, zip(*batch))
            obs_batch = torch.tensor(obs_batch, dtype=torch.float32).to(device)
            action_batch = torch.tensor(
                action_batch, dtype=torch.long).unsqueeze(-1).to(device)
            next_obs_batch = torch.tensor(
                next_obs_batch, dtype=torch.float32).to(device)
            reward_batch = torch.tensor(
                reward_batch, dtype=torch.float32).unsqueeze(-1).to(device)
            done_batch = torch.tensor(
                done_batch, dtype=torch.float32).unsqueeze(-1).to(device)

            current_q = policy_net(obs_batch).gather(
                1, action_batch).squeeze(-1)
            next_q = target_net(
                next_obs_batch).max(dim=1, keepdim=True)[0].detach()
            target_q = reward_batch + gamma * next_q * (1 - done_batch)

            loss = F.mse_loss(current_q, target_q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if env.frame_count % (target_update * 1000) == 0:
                target_net.load_state_dict(policy_net.state_dict())

        print('Episode: %d, total_reward: %d' % (i_episode, total_reward))

    # Save model
    torch.save(policy_net.state_dict(), 'policy_net.pt')


if __name__ == '__main__':
    train()
