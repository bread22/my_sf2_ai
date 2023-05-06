from model import CustomCNN, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from custom_sf2_env import CustomSF2Env
from stable_baselines3.common.callbacks import BaseCallback


class RenderEveryTenEpisodesCallback(BaseCallback):
    def __init__(self, check_freq: int):
        super(RenderEveryTenEpisodesCallback, self).__init__()
        self.check_freq = check_freq
        self.episode_counter = 0
        self.should_render = False

    def _on_step(self) -> bool:
        game_over = self.training_env.get_attr('game_over')[0]
        if game_over:
            self.episode_counter += 1
            self.training_env.env_method('reset', indices=[0])
            if self.episode_counter % self.check_freq == 0:
                self.should_render = True
            else:
                self.should_render = False

        if self.should_render:
            self.training_env.env_method('render', indices=[0])

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

    num_envs = 4  # Number of environments to run in parallel

    # Create a vectorized environment with multiple parallel environments
    env = SubprocVecEnv([make_env(env_id, state) for _ in range(num_envs)])

    policy_kwargs = {
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs": {"feature_dim": 128},
    }

    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, device='cuda')
    # model.learn(total_timesteps=int(1e5))
    # callback = RenderEveryTenEpisodesCallback(check_freq=5)
    model.learn(total_timesteps=int(1e6))

    model.save(model_path)


if __name__ == "__main__":
    main()
