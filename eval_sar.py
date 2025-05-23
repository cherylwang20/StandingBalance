import mujoco
import sys
sys.path.append('./../')
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
import joblib
import argparse
parser = argparse.ArgumentParser(description="Main script to train an agent")

parser.add_argument("--env_name", type=str, default='myoTorsoBalanceExo-v0', help="environment name")
parser.add_argument("--policy", type=str, default='Healthy', help="policy name")

args = parser.parse_args()

# Ignore specific warning
warnings.filterwarnings("ignore", message=".*tostring.*is deprecated.*")


class ActionSpaceWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.syn_action_shape = 24 + 80  
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

class SynergyWrapper(gym.ActionWrapper):
    """
    gym.ActionWrapper that reformulates the action space as the synergy space and inverse transforms
    synergy-exploiting actions back into the original muscle activation space.
    """
    def __init__(self, env, ica, pca, phi):
        super().__init__(env)
        self.ica = ica
        self.pca = pca
        self.scaler = phi
        
        self.action_space = gym.spaces.Box(low=-1., high=1., shape=(self.pca.components_.shape[0],),dtype=np.float32)
    
    def action(self, act):
        action = self.pca.inverse_transform(self.ica.inverse_transform(self.scaler.inverse_transform([act])))
        return action[0]

def load_locomotion_SAR():
    """
    Loads the trained SAR model for locomotion tasks.

    Returns
    -------
    tuple
        The trained ICA model, PCA model, and scaler for locomotion tasks.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(current_dir, './ICAPCA')
    dim = 35

    ica = joblib.load(os.path.join(root_dir, f'ica_{dim}.pkl'))
    pca = joblib.load(os.path.join(root_dir, f'pca_{dim}.pkl'))
    normalizer = joblib.load(os.path.join(root_dir, f'scaler_{dim}.pkl'))

    return ica, pca, normalizer

nb_seed = 1

torso = False
movie = True
path = os.getcwd()
ica,pca,normalizer = load_locomotion_SAR()
env_name = args.env_name 

model_num = args.policy 
model = SAC.load(path+'/pretrained_policies/' + model_num +
                 r'/best_model')

env = SynergyWrapper(ActionSpaceWrapper(gym.make(env_name)), ica, pca, normalizer)
m = []

env.reset()

options = mujoco.MjvOption()
mujoco.mjv_defaultOption(options)
options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

# Tweak scales of contact visualization elements
env.sim.model.vis.scale.contactwidth = 0.1
env.sim.model.vis.scale.contactheight = 0.03
env.sim.model.vis.scale.forcewidth = 0.05
env.sim.model.vis.map.force = 0.05

random.seed() 

frames = []
view = 'front'
m_act = []
all_rewards = []
ankle_torque  = []
ankle_torque_2 = []
ankle_torque_3 = []
ank_muscle = []
for _ in tqdm(range(1)):
    ep_rewards = []
    done = False
    obs = env.reset()
    step = 0
    muscle_act = []
    acceleration = []
    while (not done) and (step < 700):
          obs = env.obsdict2obsvec(env.obs_dict, env.obs_keys)[1]  
          action, _ = model.predict(obs, deterministic= True)
          obs, reward, done, info, _ = env.step(action)
          ep_rewards.append(reward)
          m.append(action)
          if movie:
                  geom_1_indices = np.where(env.sim.model.geom_group == 1)
                  geom_2_indices = np.where(env.sim.model.geom_group == 2)
                  env.sim.model.geom_rgba[geom_1_indices, 3] = 0
                  env.sim.model.geom_rgba[geom_2_indices, 3] = 0
                  #env.sim.renderer.render_to_window()
                  frame = env.sim.renderer.render_offscreen(width= 640, height=480,camera_id='side_view')
                  ank_muscle.append(env.sim.data.act[env.sim.model.actuator_name2id('tibant_r')].copy())
                  acceleration.append(np.abs(env.sim.data.joint('slide_joint').qacc.copy()  ))
                  ankle_torque.append(env.sim.data.joint('hip_flexion_r').qfrc_actuator.copy())
                  ankle_torque_2.append(env.get_limitfrc('hip_flexion_l').copy() + env.sim.data.joint('hip_flexion_l').qfrc_actuator.copy()) 
                  ankle_torque_3.append(env.sim.data.joint('hip_flexion_l').qpos.copy()*180/np.pi)
                  #print(env.sim.data.joint('hip_flexion_r').qpos.copy()*180/np.pi)
                  frame = (frame).astype(np.uint8)
                  frames.append(frame)

          step += 1
    all_rewards.append(np.sum(ep_rewards))
    m_act.append(muscle_act)
print(f"Average reward: {np.mean(all_rewards)}")
plt.plot(acceleration)
plt.show()

plt.plot(ank_muscle)
plt.show()


if movie:
    os.makedirs(path+'/videos' +'/' + env_name, exist_ok=True)
    skvideo.io.vwrite(path+'/videos'  +'/' + env_name + '/' + model_num + f'{view}_video.mp4', np.asarray(frames), inputdict = {'-r':'100'} , outputdict={"-pix_fmt": "yuv420p"})
	
