import gym
import numpy as np
import torch

from model import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def select_action(state, policy_net):
    with torch.no_grad():
        state = torch.tensor([state], dtype=torch.float32).to(device)
        q_values = policy_net(state)
        return q_values.argmax().item()

def test(load_path='policy_net.pth', num_episodes=10):
    env = gym.make('StreetFighterIISpecialChampionEdition-v0')
    obs = env.reset()
    input_shape = obs.shape[0] * obs.shape[1] * obs.shape[2]
    num_actions = env.action_space.n

    policy_net = DQN(input_shape, num_actions).to(device)
    policy_net.load_state_dict(torch.load(load_path))
    policy_net.eval()

    episode_rewards = []
    for episode in range(num_episodes):
        obs = env.reset()
        state = np.array(obs).flatten()
        episode_reward = 0

        while True:
            action = select_action(state, policy_net)
            obs, reward, done, info = env.step(action)
            state = np.array(obs).flatten()
            episode_reward += reward

            if done:
                episode_rewards.append(episode_reward)
                print(f"Episode {episode}, Reward: {episode_reward}")
                break

    return episode_rewards
