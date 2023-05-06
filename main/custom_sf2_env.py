import retro
import numpy as np
from pprint import pprint
from datetime import datetime


class CustomSF2Env(retro.RetroEnv):
    def __init__(self, game, state, scenario=None, info=None, use_restricted_actions=retro.Actions.FILTERED, *args, **kwargs):
        super(CustomSF2Env, self).__init__(game, state, scenario, info, use_restricted_actions, *args, **kwargs)
        self.printed = False
        self.info = None
        self.info_prev = None
        self.done = False

    def step(self, action):
        obs, reward, done, info = super(CustomSF2Env, self).step(action)
        # if not self.printed:
        #     pprint(f'info: {info}')

        if not self.info_prev:
            self.info, self.info_prev = info, info
        else:
            self.info_prev, self.info = self.info, info
        # Check if Bison has been defeated
        if self.is_bison_defeated(info):
            done = True
        self.done = done

        shaped_reward = self.shape_reward(reward)
        return obs, shaped_reward, done, info

    def shape_reward(self, reward):
        shaped_reward = reward

        # Add your reward shaping logic here
        # Example: Increase reward if the opponent's health decreases
        shaped_reward += (self.info_prev["enemy_health"] - self.info["enemy_health"])
        if self.info_prev["health"] == self.info["health"]:
            shaped_reward += 1

        return shaped_reward

    @property
    def game_over(self):
        return self.done

    def reset(self):
        # Load the Bison level state
        self.em.set_state(self.initial_state)

        # Call the parent class's reset method to set up the environment properly
        return super().reset()

    @staticmethod
    def is_bison_defeated(info):
        """ info format:
            {
                'enemy_matches_won': 251658240,
                'score': 1086411,
                'matches_won': 0,
                'continuetimer': 0,
                'enemy_health': 176,
                'health': 176
            }
        """
        # Implement your logic to detect if Bison has been defeated using the 'info' dictionary
        # if info['matches_won'] != 0:
        #     print(f'matches_won: {info["matches_won"]}')
        if info['matches_won'] == 2:
            with open('win_records.txt', mode='a') as fp:
                fp.write(str(datetime.now()) + '\n')

        return info['matches_won'] == 2
