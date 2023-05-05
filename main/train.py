import torch
import torch.optim as optim
import retro
import time
from model import StreetFighterAgent
import cv2
import numpy as np
import collections
import random
from copy import deepcopy


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def preprocess_state(state):
    gray_state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    resized_state = cv2.resize(gray_state, (84, 84))
    return resized_state


def one_hot_encode_action(action, num_actions):
    encoded_action = np.zeros(num_actions)
    encoded_action[action] = 1
    return encoded_action


# timer
start_time = time.time()

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create environment
env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', state='Champion.Level1.RyuVsGuile')
state = env.reset()
preprocessed_state = preprocess_state(state)
observation_shape = (1, *preprocessed_state.shape)  # Add the channel dimension
n_actions = env.action_space.n

# Initialize agent
agent = StreetFighterAgent(observation_shape, n_actions).to(device)
batch_size = 64
replay_buffer = ReplayBuffer(10000)
optimizer = torch.optim.Adam(agent.parameters(), lr=0.00025)
target_network = deepcopy(agent)

# Training parameters
num_episodes = 1000
max_timesteps = 10000
save_every = 20

for episode in range(num_episodes):
    state = preprocess_state(env.reset())
    state = np.expand_dims(state, axis=0)
    done = False

    while not done:
        action = agent.act(state)
        encoded_action = one_hot_encode_action(action, n_actions)
        next_state, reward, done, _ = env.step(encoded_action)
        env.render()  # Add this line

        next_state = preprocess_state(next_state)
        next_state = np.expand_dims(next_state, axis=0)

        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state

        if len(replay_buffer) > batch_size:
            batch = replay_buffer.sample(batch_size)
            agent.update(batch, optimizer, target_network, device)

    print(f'episode: {episode}, elapse time: {int(time.time() - start_time)}')
    if episode % save_every == 0:
        torch.save(agent.state_dict(), f"models/agent_{episode}.pt")
        print(f"Saving models/agent_{episode}.pt")
