import torch
import retro
import time
from model import StreetFighterAgent, ReplayBuffer
import cv2
import numpy as np
from copy import deepcopy
import torch.multiprocessing as mp


def preprocess_state(state):
    gray_state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    resized_state = cv2.resize(gray_state, (84, 84))
    return resized_state


def one_hot_encode_action(action, num_actions):
    encoded_action = np.zeros(num_actions)
    encoded_action[action] = 1
    return encoded_action


def train_process(process_index, device, num_episodes, save_every, init_time):
    # Set up the environment
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

    for episode in range(num_episodes):
        state = preprocess_state(env.reset())
        state = np.expand_dims(state, axis=0)
        done = False
        start_time = time.time()

        while not done:
            action = agent.act(state)
            encoded_action = one_hot_encode_action(action, n_actions)
            next_state, reward, done, _ = env.step(encoded_action)
            # env.render()

            next_state = preprocess_state(next_state)
            next_state = np.expand_dims(next_state, axis=0)

            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state

            if len(replay_buffer) > batch_size:
                batch = replay_buffer.sample(batch_size)
                agent.update(batch, optimizer, target_network, device)

        print(f'process: {process_index}, episode: {episode}, time elapsed: {int(time.time() - start_time)}, total time: {int(time.time() - init_time)}')
        if (episode + 1) % save_every == 0:
            torch.save(agent.state_dict(), f"models/agent_{process_index}_{episode + 1}.pt")
            print(f"Saving models/agent_{process_index}_{episode}.pt")


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_processes = 3
    num_episodes = 999
    save_every = 20
    init_time = time.time()

    # Use a pool of processes to parallelize
    with mp.Pool(processes=num_processes) as pool:
        pool.starmap(train_process, [(process_index, device, num_episodes, save_every, init_time) for process_index in range(num_processes)])