import gym
from myosuite.myosuite.utils import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from datetime import datetime
import torch
import time
import argparse
parser = argparse.ArgumentParser(description="Main script to train an agent")

parser.add_argument("--seed", type=int, default=0, help="Seed for random number generator")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--env_name", type=str, default=1, help="environment name")
parser.add_argument("--group", type=str, default='testing', help="group name")
parser.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate for the optimizer")
parser.add_argument("--clip_range", type=float, default=0.2, help="Clip range for the policy gradient update")
parser.add_argument("--algo", type=str, default='PPO', help="algorithm for training")

args = parser.parse_args()

step = False
sarco = False

class ActionSpaceWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.syn_action_shape = 24 + 80  # 24 reduced + 80 direct mappings
        self.action_space = gym.spaces.Box(low=-1., high=1., shape=(self.syn_action_shape,), dtype=np.float32)
        
        # Define the mapping from reduced to original action space for the first 210 muscles
        self.action_mapping = {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], #psoas major right
            1: [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],  #psoas major left
            2: [22], # RA, right
            3: [23], #RA left
            4: [24, 25, 26, 27], #ILpL right
            5: [28, 29, 30, 31], #ILpL left
            6: [32, 33, 34, 35, 36, 37, 38, 39],  #ILpT right
            7: [40, 41, 42, 43, 44, 45, 46, 47], #ILpT left
            8: [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68], #LTpT right
            9: [69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89], #LTpT left
            10: [90, 91, 92, 93, 94], #LTpL right
            11: [95, 96, 97, 98, 99], #LTpL left
            12: [100, 101, 102, 103, 104, 105, 106], #QL_post right
            13: [107, 108, 109, 110, 111, 112, 113],  #QL_post left
            14: [114, 115, 116, 117, 118],  #QL_mid right
            15: [119, 120, 121, 122, 123],  #QL_mid left
            16: [124, 125, 126, 127, 128, 129 ], #QL_ant right
            17: [130, 131, 132, 133, 134, 135], #QL_ant left
            18: [136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160], #MF right
            19: [161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185], #MF left
            20: [186, 187, 188, 189, 190, 191], #EO right
            21: [192, 193, 194, 195, 196, 197], #IO right
            22: [198, 199, 200, 201, 202, 203], #EO left
            23: [204, 205, 206, 207, 208, 209] #IO left
        }
        
        # Add the direct mapping for the next 80 muscles (210 to 289)
        for i in range(24, 104):
            self.action_mapping[i] = [i + 184]  # Mapping 210 to 290 (offset by 184)

    def action(self, action):
        # Map the reduced action space to the full action vector
        assert len(action) == self.syn_action_shape

        full_action = np.zeros(self.env.action_space.shape)
        for i, indices in self.action_mapping.items():
            full_action[indices] = action[i]
        return full_action

def make_env(env_name, idx, seed=0):
    def _init():
        env = ActionSpaceWrapper(gym.make(env_name))
        env.seed(seed + idx)
        return env
    return _init

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

class TensorboardCallback(BaseCallback):
	"""
	Custom callback for plotting additional values in tensorboard.
	"""
	def __init__(self, verbose=0):
	    super(TensorboardCallback, self).__init__(verbose)
	def _on_step(self) -> bool:
	    # Log scalar value (here a random variable)
	    value = self.training_env.get_obs_vec()
	    self.logger.record("obs", value)
	
	    return True
	
def main():
    dof_env = ['myoStandingBack-v0']


    training_steps = 8000000
    for env_name in dof_env:
        print('Begin training')
        ENTROPY = 0.01
        start_time = time.time()
        time_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        time_now = time_now + str(args.seed) + args.algo
        print(time_now + '\n\n')
        LR = linear_schedule(args.learning_rate)
        CR = linear_schedule(args.clip_range)

        IS_WnB_enabled = False

        loaded_model = '2025_01_21_16_23_510SAC'
        try:
            import wandb
            from wandb.integration.sb3 import WandbCallback
            IS_WnB_enabled = True
            config = {
                "policy_type": 'PPO',
                'name': time_now,
                "total_timesteps": training_steps,
                "env_name": env_name,
                "dense_units": 512,
                "activation": "relu",
                "max_episode_steps": 200,
                "seed": args.seed,
                "entropy": ENTROPY,
                "lr": args.learning_rate,
                "CR": args.clip_range,
                "num_envs": args.num_envs,
                "loaded_model": loaded_model,
            }
            #config = {**config, **envs.rwd_keys_wt}
            run = wandb.init(project="MyoBack_Train",
                            group=args.group,
                            settings=wandb.Settings(start_method="thread"),
                            config=config,
                            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                            monitor_gym=True,  # auto-upload the videos of agents playing the game
                            save_code=True,  # optional
                            )
        except ImportError as e:
            pass 
        env_name = args.env_name
        log_path = './standingBalance/policy_best_model/'+ env_name + '/' + time_now +'/'
        num_cpu = args.num_envs
        env = SubprocVecEnv([make_env(env_name, i, seed=args.seed) for i in range(num_cpu)])
        envs = VecMonitor(env)

        eval_callback = EvalCallback(envs, best_model_save_path=log_path, log_path=log_path, eval_freq=2000, deterministic=True, render=False)


        policy_kwargs = {
            'activation_fn': torch.nn.modules.activation.ReLU,
            'net_arch': {'pi': [512, 512], 'vf': [512, 512]}
            }
        #policy_kwargs = dict(activation_fn=torch.nn.Sigmoid, net_arch=(dict(pi=[64, 64], vf=[64, 64])))
        #model = PPO.load('standingBalance/policy_best_model/myoLegReachFixed-v2/2023_11_16_16_11_00/best_model',  env, verbose=0, policy_kwargs=policy_kwargs, tensorboard_log="./standingBalance/temp_env_tensorboard/"+env_name)
        if args.algo == 'PPO':
            model = PPO('MlpPolicy', envs, ent_coef=0.01, learning_rate=LR, clip_range=CR, verbose=0, policy_kwargs =policy_kwargs, tensorboard_log=f"runs/{time_now}")
            #model = PPO.load('standingBalance/policy_best_model/' + 'myoSarcTorsoReachFixed-v1' + '/' + loaded_model +'/best_model',  envs, ent_coef=0.001, learning_rate=LR, clip_range=CR, verbose=0, policy_kwargs=policy_kwargs, tensorboard_log="./standingBalance/temp_env_tensorboard/"+env_name)
        elif args.algo == 'SAC':
            net_shape = [400, 300]
            policy_kwargs = dict(net_arch=dict(pi=net_shape, qf=net_shape))
            #model = SAC('MlpPolicy', envs, buffer_size=100000, policy_kwargs=policy_kwargs, learning_rate=LR, verbose=0,  tensorboard_log=f"runs/{time_now}")
            model = SAC.load('standingBalance/policy_best_model/' + 'myoTorsoReachFixed-v1' + '/' + loaded_model +'/best_model', envs,  buffer_size=100000, learning_rate=LR, verbose=0, tensorboard_log=f"runs/{time_now}")
        
        obs_callback = TensorboardCallback()
        callback = CallbackList([eval_callback, WandbCallback(gradient_save_freq=100)])#, obs_callback])

        model.learn(total_timesteps= training_steps, tb_log_name=env_name+"_" + time_now, callback=callback)
        model.save('ep_train_results')
        elapsed_time = time.time() - start_time

        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)

        print(time_now)
        print(f"Elapsed time: {hours} hours, {minutes} minutes, {seconds} seconds.")
        if IS_WnB_enabled:
            run.finish()


if __name__ == "__main__":
    # TRAIN
    main()