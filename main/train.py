import torch
import torch.optim as optim
import gym
import time
from model import StreetFighterAgent

# timer
start_time = int(time.time())

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create environment
env = gym.make('StreetFighterIISpecialChampionEdition-Genesis')
observation_shape = env.observation_space.shape
n_actions = env.action_space.n

# Initialize agent
agent = StreetFighterAgent(observation_shape, n_actions).to(device)
optimizer = optim.Adam(agent.parameters())

# Training parameters
num_episodes = 1000
max_timesteps = 10000
save_every = 20

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    for t in range(max_timesteps):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        # Update agent
        agent.update(next_state, reward, done)

        state = next_state

        if done:
            break

    print(f"Episode: {episode}, Total Reward: {total_reward}, elap_time: {int(time.time() - start_time)}")

    if episode % save_every == 0:
        torch.save(agent.state_dict(), f"models/model_{episode}.pth")
