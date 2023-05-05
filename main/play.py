import torch
import retro
import cv2
import numpy as np
from model import StreetFighterAgent


def preprocess_state(state):
    gray_state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    resized_state = cv2.resize(gray_state, (84, 84))
    return resized_state


def one_hot_encode_action(action, num_actions):
    encoded_action = np.zeros(num_actions)
    encoded_action[action] = 1
    return encoded_action


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create environment
env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', state='Champion.Level1.RyuVsGuile')
state = env.reset()
preprocessed_state = preprocess_state(state)
observation_shape = (1, *preprocessed_state.shape)  # Add the channel dimension
n_actions = env.action_space.n

# Initialize agent
agent = StreetFighterAgent(observation_shape, n_actions).to(device)

# Load trained model
agent.load_state_dict(torch.load("models/final_agent.pt"))

# Play
num_episodes = 10

for episode in range(num_episodes):
    state = preprocess_state(env.reset())
    state = np.expand_dims(state, axis=0)
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)
        encoded_action = one_hot_encode_action(action, n_actions)
        next_state, reward, done, _ = env.step(encoded_action)
        env.render()

        next_state = preprocess_state(next_state)
        next_state = np.expand_dims(next_state, axis=0)
        state = next_state
        total_reward += reward

    print(f'Episode {episode + 1}, Total reward: {total_reward}')

env.close()
