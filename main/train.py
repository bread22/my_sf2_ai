from model import CustomCNN, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from custom_sf2_env import CustomSF2Env
from stable_baselines3.common.callbacks import BaseCallback


class SaveEveryNIterationsCallback(BaseCallback):
    def __init__(self, save_freq: int, model_path: str):
        super(SaveEveryNIterationsCallback, self).__init__()
        self.save_freq = save_freq
        self.model_path = model_path
        self.episode_counter = 0
        self.saved_last_iteration = False

    def _on_step(self) -> bool:
        if self.training_env.get_attr('game_over')[0]:
            self.episode_counter += 1
            print(f'Episode: {self.episode_counter}')
            self.saved_last_iteration = False

        if self.episode_counter % self.save_freq == 0 and not self.saved_last_iteration:
            print(f'Saving model at {self.episode_counter // self.training_env.num_envs} iterations')
            self.model.save(f"{self.model_path}_iter{self.episode_counter // self.training_env.num_envs}")
            self.saved_last_iteration = True

        return True



def render_callback(local_vars, global_vars):
    local_vars['self'].env.render()


def make_env(env_id, state):
    def _init():
        env = CustomSF2Env(game=env_id, state=state)
        return env
    return _init


def main():
    env_id = "StreetFighterIISpecialChampionEdition-Genesis"
    # state = "Champion.Level1.RyuVsGuile"
    state = "Champion.Level12.RyuVsBison"

    model_path = 'models/sf2_model'
    save_freq = 10
    num_envs = 4  # Number of environments to run in parallel

    # Create a vectorized environment with multiple parallel environments
    env = SubprocVecEnv([make_env(env_id, state) for _ in range(num_envs)])

    policy_kwargs = {
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs": {"feature_dim": 128},
    }

    # Load the saved model
    model = PPO.load(model_path, env)
    save_every_n_iterations_callback = SaveEveryNIterationsCallback(save_freq=save_freq, model_path=model_path)
    model.learn(total_timesteps=int(5e6), callback=save_every_n_iterations_callback)

    model.save(model_path)


if __name__ == "__main__":
    main()
