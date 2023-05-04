import random
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import DQN, ReplayMemory

BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.02
EPS_DECAY = 1000000
TARGET_UPDATE = 1000
MEMORY_SIZE = 100000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

    q_values = policy_net(states).gather(1, actions)
    next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
    expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)

    loss = nn.SmoothL1Loss()(q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def select_action(state, policy_net, eps):
    if random.random() < eps:
        return random.randrange(policy_net.fc3.out_features)
    else:
        with torch.no_grad():
            state = torch.tensor([state], dtype=torch.float32).to(device)
            q_values = policy_net(state)
            return q_values.argmax().item()

def train(num_episodes=1000):
    env = gym.make('StreetFighterIISpecialChampionEdition-v0')
    obs = env.reset()
    input_shape = obs.shape[0] * obs.shape[1] * obs.shape[2]
    num_actions = env.action_space.n
    policy_net = DQN(input_shape, num_actions).to(device)
    target_net = DQN(input_shape, num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    memory = ReplayMemory(MEMORY_SIZE)

    episode_rewards = []
    for episode in range(num_episodes):
        obs = env.reset()
        state = np.array(obs).flatten()
        eps = EPS_END + (EPS_START - EPS_END) * np.exp(-episode / EPS_DECAY)
        episode_reward = 0

        while True:
            action = select_action(state, policy_net, eps)
            obs, reward, done, info = env.step(action)
            next_state = np.array(obs).flatten()
            episode_reward += reward
            reward = np.sign(reward)
            memory.push(state, action, reward, next_state, done)
            state = next_state

            optimize_model(memory, policy_net, target_net, optimizer)

            if done:
                episode_rewards.append(episode_reward)
                print(f"Episode {episode}, Reward: {episode_reward}")
                break

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    return policy_net, episode_rewards
