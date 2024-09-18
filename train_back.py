import gym
from myosuite.myosuite.utils import gym
from gym import spaces
import neptune
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
#import helper_callback

from datetime import datetime
import torch
import time
import argparse
parser = argparse.ArgumentParser(description="Main script to train an agent")

parser.add_argument("--seed", type=int, default=0, help="Seed for random number generator")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--env_name", type=str, default=1, help="environment name")
parser.add_argument("--group", type=str, default='testing', help="environment name")
parser.add_argument("--learning_rate", type=float, default=0.0003, help="Learning rate for the optimizer")
parser.add_argument("--clip_range", type=float, default=0.2, help="Clip range for the policy gradient update")

args = parser.parse_args()

step = False
sarco = False

def make_env(env_name, idx, seed=0):
    def _init():
        env = gym.make(env_name)
        env.seed(seed + idx)
        return env
    return _init

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
    dof_env = ['myoStandingBack-v1']

    training_steps = 10000000
    for env_name in dof_env:
        print('Begin training')
        ENTROPY = 0.01
        start_time = time.time()
        time_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        time_now = time_now + str(args.seed)
        print(time_now + '\n\n')

        IS_WnB_enabled = False

        loaded_model = 'N/A'
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
                            settings=wandb.Settings(start_method="fork"),
                            config=config,
                            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                            monitor_gym=True,  # auto-upload the videos of agents playing the game
                            save_code=True,  # optional
                            )
        except ImportError as e:
            pass 
        env_name = 'myoStandingBack-v1'
        log_path = './standingBalance/policy_best_model/'+ env_name + '/' + time_now +'/'
        num_cpu = 4
        env = SubprocVecEnv([make_env(env_name, i, seed=args.seed) for i in range(num_cpu)])
        envs = VecMonitor(env)
        print(env_name)
        eval_callback = EvalCallback(env, best_model_save_path=log_path, log_path=log_path, eval_freq=10000, deterministic=True, render=False)


        policy_kwargs = {
            'activation_fn': torch.nn.modules.activation.ReLU,
            'net_arch': {'pi': [512, 512], 'vf': [512, 512]}
            }
        #policy_kwargs = dict(activation_fn=torch.nn.Sigmoid, net_arch=(dict(pi=[64, 64], vf=[64, 64])))
        #model = PPO.load('standingBalance/policy_best_model/myoLegReachFixed-v2/2023_11_16_16_11_00/best_model',  env, verbose=0, policy_kwargs=policy_kwargs, tensorboard_log="./standingBalance/temp_env_tensorboard/"+env_name)
    
        model = PPO('MlpPolicy', env, verbose=0, policy_kwargs =policy_kwargs, tensorboard_log=f"runs/{time_now}")

        obs_callback = TensorboardCallback()
        callback = CallbackList([eval_callback, WandbCallback(gradient_save_freq=100,
                    model_save_freq=10000,
                    model_save_path=f"models/{time_now}")])#, obs_callback])

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