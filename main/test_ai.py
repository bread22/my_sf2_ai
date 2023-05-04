import gym
import retro
import torch
import torch.nn.functional as F
from model import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(model_path, game, state, visualize=True):
    env = retro.make(game, state)
    obs = env.reset()
    done = False
    total_reward = 0.0
    model = DQN(env.observation_space.shape, env.action_space.n)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    while not done:
        if visualize:
            env.render()
        obs_v = torch.tensor(obs).unsqueeze(0)
        q_vals_v = model(obs_v).squeeze()
        action = torch.argmax(q_vals_v).item()
        obs, reward, done, _ = env.step(action)
        total_reward += reward

    env.close()
    return total_reward
