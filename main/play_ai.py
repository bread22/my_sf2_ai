# Load the model
loaded_model = DQN(input_dim, output_dim)
loaded_model.load_state_dict(torch.load('sf2_ai.pth'))
loaded_model.eval()

# Play the game using the trained model
num_test_episodes = 10

# Play the game using the trained model
for episode in range(num_test_episodes):
    state = preprocess_screen(env.reset())
    done = False

    while not done:
        env.render()
        with torch.no_grad():
            q_values = loaded_model(state)
            action = torch.argmax(q_values).item()

        # Convert action to an array of buttons
        action_array = action_to_array(action, env.action_space.n)

        next_state, reward, done, _ = env.step(action_array)
        state = preprocess_screen(next_state)

    print(f"Test Episode: {episode}")

env.close()