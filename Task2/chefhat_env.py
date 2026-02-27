# chefhat_env.py

import numpy as np
from reward_utils import dense_reward, potential

class ChefHatEnvWrapper:
    def __init__(self, env, gamma=0.99, use_shaping=True):
        self.env = env
        self.gamma = gamma
        self.use_shaping = use_shaping
        self.prev_obs = None

    def reset(self):
        obs = self.env.reset()
        self.prev_obs = self._process_obs(obs)
        return self.prev_obs

    def step(self, action):
        obs, env_reward, done, info = self.env.step(action)
        obs = self._process_obs(obs)

        reward = env_reward  # sparse baseline

        if self.use_shaping and self.prev_obs is not None:
            r_dense = dense_reward(info, self.prev_obs, obs)
            r_potential = self.gamma * potential(obs) - potential(self.prev_obs)
            reward += r_dense + r_potential

        self.prev_obs = obs
        return obs, reward, done, info

    def _process_obs(self, raw_obs):
        """
        Minimal state representation
        """
        return {
            "hand_size": len(raw_obs["hand"]),
            "hand": np.array(raw_obs["hand"]) / 13.0,
            "board": np.array(raw_obs["board"]) / 13.0,
            "possible_actions": raw_obs["possible_actions"],
        }