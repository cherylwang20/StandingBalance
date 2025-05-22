import gym
from myosuite.myosuite.utils import gym
import numpy as np
from gym.spaces import Box
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

class ActionSpaceWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.syn_action_shape = 26 + 80  # 26 reduced + 80 direct mappings
        self.action_space = gym.spaces.Box(low=-1., high=1., shape=(self.syn_action_shape,), dtype=np.float32)
        
        # Define the mapping from reduced to original action space for the first 210 muscles
        self.action_mapping = {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            1: [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            2: [22],
            3: [23],
            4: [24, 25, 26, 27, 32, 33, 34, 35, 36, 37, 38, 39],
            5: [28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47],
            6: [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
            7: [60, 61, 62, 63, 64, 65, 66, 67, 68],
            8: [69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80],
            9: [81, 82, 83, 84, 85, 86, 87, 88, 89],
            10: [90, 91, 92, 93, 94],
            11: [95, 96, 97, 98, 99],
            12: [100, 101, 102, 103, 104, 105, 106],
            13: [107, 108, 109, 110, 111, 112, 113],
            14: [114, 115, 116, 117, 118],
            15: [119, 120, 121, 122, 123],
            16: [124, 125, 126, 127, 128, 129],
            17: [130, 131, 132, 133, 134, 135],
            18: [136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155],
            19: [156, 157, 158, 159, 160],
            20: [161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180],
            21: [181, 182, 183, 184, 185],
            22: [186, 187, 188, 189, 190, 191],
            23: [192, 193, 194, 195, 196, 197],
            24: [198, 199, 200, 201, 202, 203],
            25: [204, 205, 206, 207, 208, 209]
        }
        
        # Add the direct mapping for the next 80 muscles (210 to 289)
        for i in range(26, 106):
            self.action_mapping[i] = [i + 184]  # Mapping 210 to 290 (offset by 184)

    def action(self, action):
        # Map the reduced action space to the full action vector
        assert len(action) == self.syn_action_shape

        full_action = np.zeros(self.env.action_space.shape)
        for i, indices in self.action_mapping.items():
            full_action[indices] = action[i]
        return full_action


class MyMultiAgentEnv(MultiAgentEnv):
    def __init__(self, env_config):
        self.env = ActionSpaceWrapper(gym.make('myoTorsoReachFixed-v0'))
        # Split the action space for two agents with different policies
        self.agents = {
            "agent_1": Box(low=-1, high=1, shape=(26,), dtype=np.float32),  # Actions 0-25
            "agent_2": Box(low=-1, high=1, shape=(81,), dtype=np.float32)   # Actions 26-106
        }

    def reset(self):
        obs = self.env.reset()
        return {agent: obs for agent in self.agents}

    def step(self, action_dict):
        combined_action = np.concatenate([action_dict["agent_1"], action_dict["agent_2"]])
        obs, reward, done, info = self.env.step(combined_action)
        return {agent: obs for agent in self.agents}, {agent: reward for agent in self.agents}, {agent: done for agent in self.agents}, {agent: info for agent in self.agents}


ray.init()
env_name = 'myoTorsoReachFixed-v0'
gym.make(env_name)
register_env('myoTorsoReachFixed-v1', lambda config: MyMultiAgentEnv(config))

policy_1 = (None, Box(low=-1, high=1, shape=(26,), dtype=np.float32), Box(low=-1, high=1, shape=(26,), dtype=np.float32), {})
policy_2 = (None, Box(low=-1, high=1, shape=(81,), dtype=np.float32), Box(low=-1, high=1, shape=(81,), dtype=np.float32), {})

policy_mapping_fn = lambda agent_id: "policy_1" if agent_id == "agent_1" else "policy_2"

trainer_config = {
    "env": env_name,
    "multiagent": {
        "policies": {
            "policy_1": policy_1,
            "policy_2": policy_2
        },
        "policy_mapping_fn": policy_mapping_fn,
    },
}

trainer = PPO(config=trainer_config)
for i in range(100):  # Train for 100 iterations
    print(trainer.train())