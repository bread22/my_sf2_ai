from model import CustomCNN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import retro


def main():
    env_id = "StreetFighterIISpecialChampionEdition-Genesis"
    state = "Champion.Level12.RyuVsBison"
    model_path = '../models/sf2_model_iter70.zip'
    test_rounds = 30
    won = 0

    env = retro.make(game=env_id, state=state)
    env = DummyVecEnv([lambda: env])

    policy_kwargs = {
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs": {"feature_dim": 128},
    }

    model = PPO.load(model_path, env)

    for i in range(1, test_rounds + 1):
        obs = env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            if info[0]['matches_won'] == 2:
                print(f'{i}: Won')
                won += 1
                break
            if info[0]['enemy_matches_won'] == 2:
                print(f'{i}: Lost')
                break
            # env.render()
    print('Winning rate {:.2%}'.format(won/test_rounds))
    env.close()


if __name__ == "__main__":
    main()
