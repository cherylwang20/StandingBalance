""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
SCRIPT CREATED TO TRY DIFFERENT REWARDS ON THE WALK_V0 ENVIRONMENT. 
================================================= """
 
import collections
import random
import gym
import numpy as np
from myosuite.envs.myo.base_v0 import BaseV0
import matplotlib.path as mplPath
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import time
from myosuite.utils.quat_math import mat2euler, euler2quat

class ReachEnvV0(BaseV0):

    DEFAULT_OBS_KEYS = ['qpos', 'qvel', 'tip_pos', 'reach_err']
    # Weights should be positive, unless the contribution of the components of the reward shuld be changed. 
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        #"positionError":    .5,
        #"metabolicCost":    1,
        "pose":             1,
        #"pelvis_rot_err": .5, 
        #'hip_flex':               1,
        #'knee_angle':             1, 
        'centerOfMass':        1, 
        #'feet_height':          1,
        #"com_error":            1,
        #"com_height_error":      1, 
        #"bonus":                1, 
        "done":                 -10.
    } 

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed)
        self.cpt = 0
        self.perturbation_time = -1
        self.perturbation_duration = 50
        self.perturbation_magnitude = 0
        eval_range = kwargs['eval_range'] if 'eval_range' in kwargs else [0, 0]
        self.force_range = [8000, 8000] #150N --> 1m/s/s ; 5m/s/s --> 700 N #low: 150, 2500 #high: 5500 - 8100
        self._setup(**kwargs)

    def _setup(self,
            target_reach_range:dict,
            target_jnt_range:dict = None,   # joint ranges as tuples {name:(min, max)}_nq
            target_jnt_value:list = None,   # desired joint vector [des_qpos]_nq
            far_th = .35,
            pose_thd = 0.35,
            obs_keys:list = DEFAULT_OBS_KEYS,
            weighted_reward_keys:dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
            **kwargs,
        ):
        self.far_th = far_th
        self.pose_thd = pose_thd
        self.target_reach_range = target_reach_range
        # resolve joint demands
        if target_jnt_range:
            self.target_jnt_ids = []
            self.target_jnt_range = []
            for jnt_name, jnt_range in target_jnt_range.items():
                self.target_jnt_ids.append(self.sim.model.joint_name2id(jnt_name))
                self.target_jnt_range.append(jnt_range)
            self.target_jnt_range = np.array(self.target_jnt_range)
            self.target_jnt_value = np.mean(self.target_jnt_range, axis=1)  # pseudo targets for init
        else:
            self.target_jnt_value = target_jnt_value
        super()._setup(obs_keys=obs_keys,
                weighted_reward_keys=weighted_reward_keys,
                sites=self.target_reach_range.keys(),
                **kwargs,
                )     
        self.init_qpos = self.sim.model.key_qpos[0]
    

    def step(self, a):
        if self.perturbation_time <= self.time < self.perturbation_time + 1/2*self.perturbation_duration*self.dt  : 
            self.sim.data.xfrc_applied[self.sim.model.body_name2id('plate'), :] = self.perturbation_magnitude
        elif self.perturbation_time + 1/2*self.perturbation_duration*self.dt <= self.time < self.perturbation_time + self.perturbation_duration*self.dt  : 
            self.sim.data.xfrc_applied[self.sim.model.body_name2id('plate'), :] = -self.perturbation_magnitude
        else: self.sim.data.xfrc_applied[self.sim.model.body_name2id('plate'), :] = np.zeros((1, 6))
        # rest of the code for performing a regular environment step
        a = np.clip(a, self.action_space.low, self.action_space.high )
        self.last_ctrl = self.robot.step(ctrl_desired=a,
                                          ctrl_normalized=self.normalize_act,
                                          step_duration=self.dt,
                                          realTimeSim=self.mujoco_render_frames,
                                          render_cbk=self.mj_render if self.mujoco_render_frames else None)
        return super().forward()

    def get_obs_vec(self):
        self.obs_dict['time'] = np.array([self.sim.data.time])
        self.obs_dict['qpos'] = self.sim.data.qpos[:56].copy()
        self.obs_dict['qvel'] = self.sim.data.qvel[:55].copy()*self.dt
        if self.sim.model.na>0:
            self.obs_dict['act'] = self.sim.data.act[:].copy()
        # reach error
        self.obs_dict['tip_pos'] = np.array([])
        self.obs_dict['target_pos'] = np.array([])
        for isite in range(len(self.tip_sids)):
            self.obs_dict['tip_pos'] = np.append(self.obs_dict['tip_pos'], self.sim.data.site_xpos[self.tip_sids[isite]].copy())
            self.obs_dict['target_pos'] = np.append(self.obs_dict['target_pos'], self.sim.data.site_xpos[self.target_sids[isite]].copy())
        self.obs_dict['reach_err'] = np.array(self.obs_dict['target_pos'])-np.array(self.obs_dict['tip_pos'])

        # center of mass and base of support
        xpos = {}
        body_names = ['calcn_l', 'calcn_r', 'femur_l', 'femur_r', 'patella_l', 'patella_r', 'pelvis', 
                      'root', 'talus_l', 'talus_r', 'tibia_l', 'tibia_r', 'toes_l', 'toes_r', 'world']
        for names in body_names: xpos[names] = self.sim.data.xipos[self.sim.model.body_name2id(names)].copy() # store x and y position of the com of the bodies
        # Bodies relevant for hte base of support: 
        labels = ['calcn_r', 'calcn_l', 'toes_l', 'toes_r']
        x, y = [], [] # Storing position of the foot
        for label in labels:
            x.append(xpos[label][0]) # storing x position
            y.append(xpos[label][1]) # storing y position
        # CoM is considered to be the center of mass of the pelvis (for now)
        pos = self.sim.data.xipos.copy()
        vel = self.sim.data.cvel.copy()
        mass = self.sim.model.body_mass
        print(np.sum(mass))
        com_v = np.sum(vel *  mass.reshape((-1, 1)), axis=0) / np.sum(mass)
        self.obs_dict['com_v'] = com_v[-3:]
        com = np.sum(pos * mass.reshape((-1, 1)), axis=0) / np.sum(mass)
        self.obs_dict['com'] = com[:2]
        self.obs_dict['com_height'] = com[-1:]
        self.obs_dict['hip_flex_r'] = np.asarray(self.sim.data.joint('hip_flexion_r').qpos.copy())
        self.obs_dict['cal_l'] = np.array(self.sim.data.xipos[self.sim.model.body_name2id('calcn_l')].copy()[1])
        # Storing base of support - x and y position of right and left calcaneus and toes
        self.obs_dict['base_support'] =  [x, y]
        #self.obs_dict['ver_sep'] = np.array(max(y), min(y))
        # print('Ordered keys: {}'.format(self.obs_keys))
        self.obs_dict['err_cal'] = np.array(0.31 - self.obs_dict['cal_l'] )
        self.obs_dict['knee_angle'] = np.array(np.mean(self.sim.data.qpos[self.sim.model.joint_name2id('knee_angle_l')].copy() + self.sim.data.qpos[self.sim.model.joint_name2id('knee_angle_r')].copy()))
        self.obs_dict['pose_err'] = self.sim.model.key_qpos[0] - self.sim.data.qpos.copy()
        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs

    def get_obs_dict(self, sim):
        obs_dict = {}

        obs_dict['time'] = np.array([sim.data.time])      
        obs_dict['qpos'] = sim.data.qpos[:56].copy()
        obs_dict['qvel'] = sim.data.qvel[:55].copy()*self.dt
        obs_dict['pose_err'] = sim.model.key_qpos[0] - sim.data.qpos.copy()
        #np.array([-0.3, 0, 0]) - np.array([sim.data.joint('flex_extension').qpos[0].copy(),sim.data.joint('lat_bending').qpos[0].copy(), sim.data.joint('axial_rotation').qpos[0].copy()])
        #print([self.sim.data.joint('flex_extension').qpos.copy(),self.sim.data.joint('lat_bending').qpos.copy(), self.sim.data.joint('axial_rotation').qpos.copy()])
        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()
        # reach error
        obs_dict['tip_pos'] = np.array([])
        obs_dict['target_pos'] = np.array([])

        ### we append the target position of the two feet
        

        for isite in range(len(self.tip_sids)):
            obs_dict['tip_pos'] = np.append(obs_dict['tip_pos'], sim.data.site_xpos[self.tip_sids[isite]].copy())
            obs_dict['target_pos'] = np.append(obs_dict['target_pos'], sim.data.site_xpos[self.target_sids[isite]].copy())
        obs_dict['reach_err'] = np.array(obs_dict['target_pos'])-np.array(obs_dict['tip_pos'])

        obs_dict['feet_heights'] = self._get_feet_heights().copy()
        a = (self.sim.data.joint('hip_adduction_r').qpos.copy()+self.sim.data.joint('hip_adduction_l').qpos.copy())/2
        obs_dict['hip_add'] = np.asarray([a])
        b = (self.sim.data.joint('knee_angle_r').qpos.copy()+self.sim.data.joint('knee_angle_l').qpos.copy())/2
        obs_dict['knee_angle'] = np.asarray([b])
        c = (self.sim.data.joint('hip_flexion_r').qpos.copy()+self.sim.data.joint('hip_flexion_l').qpos.copy())/2
        obs_dict['hip_flex'] = np.asarray([c])
        obs_dict['hip_flex_r'] = np.asarray(self.sim.data.joint('hip_flexion_r').qpos.copy())

        # center of mass and base of support
        x, y = np.array([]), np.array([])
        for label in ['calcn_r', 'calcn_l', 'toes_l', 'toes_r']:
            xpos = np.array(sim.data.xipos[sim.model.body_name2id(label)].copy())[:2] # select x and y position of the current body
            x = np.append(x, xpos[0])
            y = np.append(y, xpos[1])
        #obs_dict['cal_l'] = np.array(sim.data.xipos[sim.model.body_name2id('calcn_l')].copy())
        obs_dict['base_support'] = np.append(x, y)
        #obs_dict['ver_sep'] = np.array(max(y), min(y))
        # CoM is considered to be the center of mass of the pelvis (for now) 
        pos = sim.data.xipos.copy()
        vel = sim.data.cvel.copy()
        obs_dict['feet_v'] = sim.data.cvel[sim.model.body_name2id('patella_r')].copy()
        #3*sim.data.cvel[sim.model.body_name2id('pelvis')].copy() - sim.data.cvel[sim.model.body_name2id('toes_r')].copy()
        mass = sim.model.body_mass
        #print(np.sum(mass))
        com = np.sum(pos * mass.reshape((-1, 1)), axis=0) / np.sum(mass)
        com_v = np.sum(vel *  mass.reshape((-1, 1)), axis=0) / np.sum(mass)
        obs_dict['com_v'] = com_v[-3:]
        obs_dict['com'] = com[:2]
        obs_dict['com_height'] = com[-1:]# self.sim.data.body('pelvis').xipos.copy()
        #print('com_height', obs_dict['com_height'])
        baseSupport = obs_dict['base_support'].reshape(2,4)
        #areaofbase = Polygon(zip(baseSupport[0], baseSupport[1])).area
        obs_dict['centroid'] = np.array(Polygon(zip(baseSupport[0], baseSupport[1])).centroid.coords)
        pelvis_com = np.array(sim.data.xipos[sim.model.body_name2id('pelvis')].copy())
        obs_dict['pelvis_com'] = pelvis_com[:2]
        obs_dict['err_com'] = np.array(obs_dict['centroid']- obs_dict['com'])
        #obs_dict['err_com'] = np.array(obs_dict['centroid']- obs_dict['pelvis_com']) #change since 2023/12/08/ 15:52
        return obs_dict


    def get_reward_dict(self, obs_dict):
        pose_dist = np.linalg.norm(obs_dict['pose_err'], axis=-1)
        #print(pose_dist)
        #print('hip flexion',self.sim.data.joint('hip_flexion_r').qpos.copy())
        hip_fle = self.obs_dict['hip_flex']
        hip_flex_r = self.obs_dict['hip_flex_r'].reshape(-1)[0]
        hip_add = self.obs_dict['hip_add']
        knee_angle = self.obs_dict['knee_angle']
        self.obs_dict['pelvis_target_rot'] = [-2.6389, -np.pi/2 , 2.012]
        self.obs_dict['pelvis_rot'] = mat2euler(np.reshape(self.sim.data.site_xmat[self.sim.model.site_name2id("pelvis")], (3, 3)))
        
        #print('rotation', self.obs_dict['pelvis_rot'])
        pelvis_rot_err = np.abs(np.linalg.norm(self.obs_dict['pelvis_rot'] - self.obs_dict['pelvis_target_rot'] , axis=-1))
        #print(self.obs_dict['pelvis_rot'])
        #print(-self.obs_dict['pelvis_rot'][0]+self.sim.data.joint('hip_flexion_r').qpos.copy() )
        positionError = np.linalg.norm(obs_dict['reach_err'], axis=-1)
        feet_v = np.linalg.norm(obs_dict['feet_v'][-3:], axis = -1) 
        com_vel = np.linalg.norm(obs_dict['com_v'], axis = -1) # want to minimize translational velocity
        comError = np.linalg.norm(obs_dict['err_com'], axis=-1)
        self.com_err = comError
        timeStanding = np.linalg.norm(obs_dict['time'], axis=-1)
        metabolicCost = np.sum(np.square(obs_dict['act']))/self.sim.model.na
        # act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)/self.sim.model.na if self.sim.model.na !=0 else 0
        # Within: center of mass in between toes and calcaneous and rihgt foot left foot
        baseSupport = obs_dict['base_support'].reshape(2,4)
        centerMass = np.squeeze(obs_dict['com']) #.reshape(1,2)
        bos = mplPath.Path(baseSupport.T)
        within = bos.contains_point(centerMass)
        #print(within)
        
        feet_width, vertical_sep = self.feet_width()
        feet_height = np.linalg.norm(obs_dict['feet_heights'])
        com_height = obs_dict['com_height'][0]
        com_height_error = np.linalg.norm(obs_dict['com_height'][0]-0.855)
        com_bos = 1 if within else -1 # Reward is 100 if com is in bos.
        farThresh = self.far_th*len(self.tip_sids) if np.squeeze(obs_dict['time'])>2*self.dt else np.inf # farThresh = 0.5
        nearThresh = len(self.tip_sids)*.050 # nearThresh = 0.05
        # Rewards are defined ni the dictionary with the appropiate sign
        comError = comError.reshape(-1)[0]
        #print(within,comError)
        positionError = positionError.reshape(-1)[0]
        pose_dist = pose_dist.reshape(-1)[0]
        #print(positionError, pose_dist)
        com_height_error = com_height_error.reshape(-1)[0]
        feet_v = feet_v.reshape(-1)[0]
        timeStanding = timeStanding.reshape(-1)[0]
        com_height = com_height.reshape(-1)[0]
        hip_add = hip_add.reshape(-1)[0]
        hip_fle = np.abs(hip_fle.reshape(-1)[0] - 0.2)
        knee_angle = knee_angle.reshape(-1)[0]
        com_vel = com_vel.reshape(-1)[0]
        #print(pose_dist, hip_fle)
        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('positionError',        -1*positionError ),#-10.*vel_dist
            ('pose',    -1.*pose_dist),
            ('bonus',   1.*(pose_dist<self.pose_thd) + 1.*(pose_dist<1.5*self.pose_thd)),
            #('smallErrorBonus',     1.*(positionError<2*nearThresh) + 1.*(positionError<nearThresh)),
            #('timeStanding',        1.*timeStanding), 
            ('metabolicCost',       -1.*metabolicCost),
            #('highError',           -1.*(positionError>farThresh)),
            ('centerOfMass',        1.*(com_bos)),
            ('com_error',             np.exp(-5.*np.abs(comError))),
            ('com_height_error',     np.exp(-5*np.abs(com_height_error))),
            ('feet_height',         np.exp(-1*feet_height)),
            ('feet_width',            5*np.clip(feet_width, 0.3, 0.5)),
            ('pelvis_rot_err',        -1 * pelvis_rot_err),
            ('com_v',                  3*np.exp(-5*np.abs(com_vel))), #3*(com_bos - np.tanh(feet_v))**2), #penalize when COM_v is high
            ('hip_add',               2*np.clip(hip_add, -0.3, -0.2)),
            ('knee_angle',             10*np.clip(knee_angle, 0.4, 0.6)),
            ('hip_flex',               np.exp(-hip_fle)),#10*np.clip(hip_fle, 0.4, 0.7)),
            ('hip_flex_r',             5*np.exp(-.5*np.abs(hip_flex_r - 1))),
            # Must keys
            ('bonus',                1),
            ('sparse',              -1.*positionError),
            ('solved',              1.*hip_flex_r>1),  # standing task succesful
            ('done',                False), # model has failed to complete the task 
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        #print([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()])
        return rwd_dict
    
    def _get_feet_heights(self):
        """
        Get the height of both feet.
        """
        foot_id_l = self.sim.model.body_name2id('calcn_l') 
        foot_id_ll = self.sim.model.body_name2id('toes_l')
        foot_id_r = self.sim.model.body_name2id('calcn_r')
        foot_id_rr = self.sim.model.body_name2id('toes_r')
        return np.array([self.sim.data.body_xpos[foot_id_l][2], self.sim.data.body_xpos[foot_id_ll][2], self.sim.data.body_xpos[foot_id_r][2], self.sim.data.body_xpos[foot_id_rr][2]])
    
    def allocate_randomly(self, perturbation_magnitude): #allocate the perturbation randomly in one of the six directions
        array = np.zeros(6)
        random_index = random.choice([0]) # 0: ML, 1: AP fall back, 3: AP fall forward
        array[random_index] = perturbation_magnitude
        return array
    # generate a perturbation
    
    def generate_perturbation(self):
        M = self.sim.model.body_mass.sum()
        g = np.abs(self.sim.model.opt.gravity.sum())
        self.perturbation_time = np.random.uniform(self.dt*(0.35*self.horizon), self.dt*(0.36*self.horizon)) # between 10 and 20 percent, was between 0.1 and 0.4
        # perturbation_magnitude = np.random.uniform(0.08*M*g, 0.14*M*g)
        ran = self.force_range
    
        perturbation_magnitude = np.random.uniform(ran[0], ran[1])
        if random.random() > 1:
            perturbation_magnitude = - perturbation_magnitude
        self.perturbation_magnitude = self.allocate_randomly(perturbation_magnitude)#[0,0,0, perturbation_magnitude, 0, 0] # front and back

        return
        # generate a valid target

    def generate_targets(self):
        for site, span in self.target_reach_range.items():
            sid = self.sim.model.site_name2id(site)
            sid_target = self.sim.model.site_name2id(site+'_target')
            self.sim.model.site_pos[sid] = self.sim.data.site_xpos[sid].copy() + self.np_random.uniform(low=span[0], high=span[1])
        self.sim.forward()

    def feet_width(self):
        #'calcn_r', 'calcn_l', 'toes_l', 'toes_r'
        a = self.obs_dict['base_support'].reshape(2, 4)
        x, y = a[0], a[1]
        width = np.abs((x[0]-x[1])**2)
        #width = np.sqrt((np.mean([x[0], x[3]])+np.mean([x[1], x[2]]))**2 +(np.mean([y[0], y[3]])+np.mean([y[1], y[2]]))**2)#np.abs(np.mean([x[0], x[3]]) - np.mean([x[1], x[2]]))
        step = np.abs(np.mean([y[0], y[3]]) - np.mean([y[1], y[2]]))
        return width, step

    def reset(self, **kwargs):
        self.generate_perturbation()
        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super().reset(**kwargs)
        return obs
    
    def get_limitfrc(self, joint_name):
        non_joint_limit_efc_idxs = np.where(self.sim.data.efc_type != self.sim.lib.mjtConstraint.mjCNSTR_LIMIT_JOINT)[0]
        only_jnt_lim_efc_force = self.sim.data.efc_force.copy()
        only_jnt_lim_efc_force[non_joint_limit_efc_idxs] = 0.0
        joint_force = np.zeros((self.sim.model.nv,))
        self.sim.lib.mj_mulJacTVec(self.sim.model._model, self.sim.data._data, joint_force, only_jnt_lim_efc_force)
        return joint_force[self.sim.model.joint(joint_name).dofadr]