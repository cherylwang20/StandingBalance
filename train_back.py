import gym
from myosuite.myosuite.utils import gym
from gym import spaces
import neptune
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor
#import helper_callback

from datetime import datetime
import torch
import time

step = False
sarco = False

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
	
class CustomNeptuneCallback(BaseCallback):
    def __init__(self, run):
        super(CustomNeptuneCallback, self).__init__(verbose=1)
        self.run = run
        # You might want to add more parameters here if needed

    def _on_step(self) -> bool:
        # Check if an episode has ended
        if 'episode' in self.locals["infos"][0]:
            episode_info = self.locals["infos"][0]['episode']
            # Log episodic information to Neptune
            self.run["metrics/episode_reward"].append(episode_info['r'])
            self.run["metrics/episode_length"].append(episode_info['l'])

        return True

dof_env = ['myoStandingBack-v1']

training_steps = 1000
for env_name in dof_env:
    print('Begin training')

    start_time = time.time()
    time_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    print(time_now + '\n\n')


    env_name = 'myoStandingBack-v1'
    log_path = './standingBalance/policy_best_model/'+ env_name + '/' + time_now +'/'
    env = gym.make(env_name)
    print(type(env.action_space))
    env = Monitor(env)
    print(env_name)
    print(env.rwd_keys_wt) 
    print(env.obs_dict.keys())
    eval_callback = EvalCallback(env, best_model_save_path=log_path, log_path=log_path, eval_freq=10000, deterministic=True, render=False)

    loaded_model = "2024_06_18_11_28_38"

    parameter = {
    "dense_units": 512,
    "activation": "relu",
    "training_steps": training_steps,
    "loaded_model": 'N/A',
    }

    parameters = {**parameter, **env.rwd_keys_wt}


    policy_kwargs = {
        'activation_fn': torch.nn.modules.activation.ReLU,
        'net_arch': {'pi': [512, 512], 'vf': [512, 512]}
        }
    #policy_kwargs = dict(activation_fn=torch.nn.Sigmoid, net_arch=(dict(pi=[64, 64], vf=[64, 64])))
    #model = PPO.load('standingBalance/policy_best_model/myoLegReachFixed-v2/2023_11_16_16_11_00/best_model',  env, verbose=0, policy_kwargs=policy_kwargs, tensorboard_log="./standingBalance/temp_env_tensorboard/"+env_name)

    model = PPO('MlpPolicy', env, verbose=0, policy_kwargs =policy_kwargs)

    obs_callback = TensorboardCallback()
    callback = CallbackList([eval_callback])

    model.learn(total_timesteps= training_steps, tb_log_name=env_name+"_" + time_now, callback=callback)
    model.save('ep_train_results')
    elapsed_time = time.time() - start_time

    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)


    print(time_now)
    print(f"Elapsed time: {hours} hours, {minutes} minutes, {seconds} seconds.")