import os
from model import CustomCNN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import retro

def render_callback(local_vars, global_vars):
    local_vars['self'].env.render()

def make_env(env_id, state):
    def _init():
        env = retro.make(game=env_id, state=state)
        return env
    return _init

def main():
    env_id = "StreetFighterIISpecialChampionEdition-Genesis"
    state = "Champion.Level1.RyuVsGuile"
    model_path = 'models/sf2_model'

    num_envs = 4  # Number of environments to run in parallel

    # Create a vectorized environment with multiple parallel environments
    env = SubprocVecEnv([make_env(env_id, state) for _ in range(num_envs)])

    policy_kwargs = {
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs": {"feature_dim": 128},
    }

    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, device='cuda')
    model.learn(total_timesteps=int(1e6))

    model.save(model_path)

if __name__ == "__main__":
    main()
