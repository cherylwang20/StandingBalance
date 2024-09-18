import gym
from myosuite.myosuite.utils import gym
import numpy as np
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import skvideo
import skvideo.io
import os
import cv2
import random
from tqdm.auto import tqdm

nb_seed = 1

torso = False
movie = True
path = './'

env_name = 'myoStandingBack-v0'
model_num =  '2024_09_17_10_36_35' #'2024_04_23_17_17_53' 
model = PPO.load(path+'/standingBalance/policy_best_model'+ '/'+ env_name + '/' + model_num +
                 r'/best_model')



#model = PPO.load('ep_train_results')
env = gym.make(env_name)
s, m, t = [], [], []

env.reset()

random.seed() 

frames = []
view = 'front'
m_act = []
for _ in tqdm(range(2)):
    ep_rewards = []
    done = False
    obs = env.reset()
    step = 0
    muscle_act = []
    for _ in tqdm(range(300)):
          obs = env.obsdict2obsvec(env.obs_dict, env.obs_keys)[1]
          #obs = env.get_obs_dict()
          
          action, _ = model.predict(obs, deterministic=True)
          #env.sim.data.ctrl[:] = action
          obs, reward, done, info, _ = env.step(action)
          #t.append(env.obs_dict['reach_err']) #s.append(env.sim.data.qpos[joint_interest_id])
          m.append(action)
          if movie:
                  geom_1_indices = np.where(env.sim.model.geom_group == 1)
                  env.sim.model.geom_rgba[geom_1_indices, 3] = 0
                  frame = env.sim.renderer.render_offscreen(width= 640, height=480,camera_id=f'{view}_camera')
                  
                  frame = (frame).astype(np.uint8)
                  frame = np.flipud(frame)
            # if slow see https://github.com/facebookresearch/myosuite/blob/main/setup/README.md
                  frames.append(frame[::-1,:,:])
                  #env.sim.mj_render(mode='window') # GUI
          step += 1
    m_act.append(muscle_act)


'''
# evaluate policy
all_rewards = []
meta_list = []
succ = 20
for _ in tqdm(range(20)): # 20 random targets
  ep_rewards = []
  done = False
  obs = env.reset()
  step = 0
  meta = 0
  while (not done) and (step < 500):
      # get the next action from the policy
      obs = env.obsdict2obsvec(env.obs_dict, env.obs_keys)[1]
      #env.mj_render()
      action, _ = model.predict(obs, deterministic= True)
      metabolicCost = np.sum(np.square(env.sim.data.act[:].copy()))/env.sim.model.na 
      meta += metabolicCost
      # take an action based on the current observation
      obs, reward, done, info = env.step(action)
      ep_rewards.append(reward)
      step += 1
  if step < 500:
      succ -= 1
      print(step)
  all_rewards.append(np.sum(ep_rewards))
  meta_list.append(meta)
env.close()
print(f"Average reward: {np.mean(all_rewards)} over 20 episodes")
print(f"Successful Rate: {succ/20}")
print(f"Metabolic Cost {np.mean(meta_list)}")
'''

if movie:
    os.makedirs(path+'/videos' +'/' + env_name, exist_ok=True)
    skvideo.io.vwrite(path+'/videos'  +'/' + env_name + '/' + model_num + f'{view}_video.mp4', np.asarray(frames), inputdict = {'-r':'100'} , outputdict={"-pix_fmt": "yuv420p"})
	
