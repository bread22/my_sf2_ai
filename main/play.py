from model import CustomCNN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import retro


def main():
    env_id = "StreetFighterIISpecialChampionEdition-Genesis"
    state = "Champion.Level1.RyuVsGuile"
    model_path = 'models/sf2_model'

    env = retro.make(game=env_id, state=state)
    env = DummyVecEnv([lambda: env])

    policy_kwargs = {
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs": {"feature_dim": 128},
    }

    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, device='cuda')
    model.load(model_path)

    obs = env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()


if __name__ == "__main__":
    main()
