import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import retro


# Define the Q-Network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        return self.net(x)


# Preprocessing function for game screen
def preprocess_screen(screen):
    screen = np.dot(screen[..., :3], [0.299, 0.587, 0.114])
    screen = screen / 255.0
    screen = screen[::2, ::2]  # Downsample the screen by a factor of 2
    return torch.tensor(screen.copy(), dtype=torch.float32).unsqueeze(0)


def action_to_array(action, num_buttons):
    action_array = np.zeros(num_buttons, dtype=np.uint8)
    action_array[action] = 1
    return action_array


# Define the environment and model
env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', state='Champion.Level1.RyuVsGuile')
screen_height, screen_width = preprocess_screen(env.reset()).shape[1:]
input_dim = screen_height * screen_width
#
# input_dim = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]
output_dim = env.action_space.n

model = DQN(input_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Train the model
num_episodes = 500
epsilon = 0.9
gamma = 0.99

for episode in range(num_episodes):
    state = preprocess_screen(env.reset())
    done = False

    while not done:
        env.render()

        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = model(state)
            action = torch.argmax(q_values).item()

        # Convert action to an array of buttons
        action_array = action_to_array(action, env.action_space.n)
        # Take a step in the environment
        next_state, reward, done, _ = env.step(action_array)
        next_state = preprocess_screen(next_state)

        # Update the model
        target = reward + gamma * torch.max(model(next_state))
        current_q = model(state)[0, action]

        loss = loss_fn(current_q, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
    epsilon *= 0.999

    print(f"Episode: {episode}, Epsilon: {epsilon}")

env.close()


# Save the model
torch.save(model.state_dict(), 'sf2_ai.pth')


