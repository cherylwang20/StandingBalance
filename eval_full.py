import gym
from myosuite.myosuite.utils import gym
import numpy as np
from stable_baselines3 import PPO, SAC
import matplotlib.pyplot as plt
import skvideo
import skvideo.io
import os
import random
from tqdm.auto import tqdm
import warnings

# Ignore specific warning
warnings.filterwarnings("ignore", message=".*tostring.*is deprecated.*")


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

nb_seed = 1

torso = False
movie = True
path = './'

env_name = 'myoTorsoReachFixed-v0'
#env_name = 'myoStandingBack-v1'

model_num ='2024_11_30_22_03_240SAC' #'2024_09_17_10_36_35'
model = SAC.load(path+'/standingBalance/policy_best_model'+ '/'+ env_name + '/' + model_num +
                 r'/best_model')



#model = PPO.load('ep_train_results')
env = ActionSpaceWrapper(gym.make(env_name))
s, m, t = [], [], []

env.reset()

random.seed() 

leg_action = np.loadtxt('muscle_activation.txt', dtype=np.float32)
back_action = np.loadtxt('back_activation.txt', dtype=np.float32)

frames = []
view = 'front'
m_act = []
all_rewards = []
for _ in tqdm(range(1)):
    ep_rewards = []
    done = False
    obs = env.reset()
    step = 0
    muscle_act = []
    while (not done) and (step < 400):
          obs = env.obsdict2obsvec(env.obs_dict, env.obs_keys)[1]  
          action, _ = model.predict(obs, deterministic= True)
          obs, reward, done, info, _ = env.step(action)
          ep_rewards.append(reward)
          m.append(action)
          if movie:
                  geom_1_indices = np.where(env.sim.model.geom_group == 1)
                  env.sim.model.geom_rgba[geom_1_indices, 3] = 0
                  frame = env.sim.renderer.render_offscreen(width= 440, height=380,camera_id=f'{view}_view')
                  
                  #frame = (frame).astype(np.uint8)
                  frame = np.flipud(frame)
            # if slow see https://github.com/facebookresearch/myosuite/blob/main/setup/README.md
                  frames.append(frame[::-1,:,:])
                  #env.sim.mj_render(mode='window') # GUI
          step += 1
    all_rewards.append(np.sum(ep_rewards))
    m_act.append(muscle_act)
print(f"Average reward: {np.mean(all_rewards)}")


if movie:
    os.makedirs(path+'/videos' +'/' + env_name, exist_ok=True)
    skvideo.io.vwrite(path+'/videos'  +'/' + env_name + '/' + model_num + f'{view}_video.mp4', np.asarray(frames), inputdict = {'-r':'200'} , outputdict={"-pix_fmt": "yuv420p"})
	
