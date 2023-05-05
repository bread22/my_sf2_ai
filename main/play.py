import torch
import gym
from model import StreetFighterAgent
import cv2


def preprocess_state(state):
    gray_state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    resized_state = cv2.resize(gray_state, (84, 84))
    return resized_state


# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create environment
env = gym.make('StreetFighterIISpecialChampionEdition-Genesis')
observation_shape = env.observation_space.shape
n_actions = env.action_space.n

# Initialize agent
agent = StreetFighterAgent(observation_shape, n_actions).to(device)

# Load trained model
model_path = "models/model.pth"  # Replace with the appropriate model filename
agent.load_state_dict(torch.load(model_path, map_location=device))
agent.eval()

state = env.reset()
done = False

while not done:
    env.render()

    state = preprocess_state(state)
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        action = torch.argmax(agent(state)).item()

    next_state, reward, done, _ = env.step(action)
    state = next_state

env.close()
