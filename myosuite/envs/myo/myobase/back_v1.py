""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

import collections
from myosuite.utils import gym
import numpy as np
from shapely.geometry import Polygon

from myosuite.envs.myo.base_v0 import BaseV0
import mujoco as mj


class BackEnvV0(BaseV0):

    DEFAULT_OBS_KEYS = ['qpos', 'qvel', 'pose_err']
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "pose": 1.0,
        "bonus": 4.0,
        "act_reg": 1.0,
        "penalty": 50,
        "done": 0,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):

        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        self.cpt = 0
        self.perturbation_time = -1
        self.perturbation_duration = 0
        self.force_range = [50, 100]
        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.MYO_CREDIT)

        self._setup(**kwargs)

    def _setup(self,
            target_reach_range: dict,
            viz_site_targets:tuple = None,  # site to use for targets visualization []
            target_jnt_range:dict = None,   # joint ranges as tuples {name:(min, max)}_nq
            target_jnt_value:list = None,   # desired joint vector [des_qpos]_nq

            reset_type = "init",            # none; init; random
            target_type = "generate",       # generate; switch; fixed
            obs_keys:list = DEFAULT_OBS_KEYS,
            weighted_reward_keys:dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
            pose_thd = 0.35,
            weight_bodyname = None,
            weight_range = None,
            **kwargs,
        ):
        self.target_reach_range = target_reach_range
        self.reset_type = reset_type
        self.target_type = target_type
        self.pose_thd = pose_thd
        self.weight_bodyname = weight_bodyname
        self.weight_range = weight_range

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

    def get_obs_vec(self):
        self.obs_dict['time'] = np.array([self.sim.data.time])
        self.obs_dict['qpos'] = self.sim.data.qpos[:].copy()
        self.obs_dict['qvel'] = self.sim.data.qvel[:].copy()*self.dt
        if self.sim.model.na>0:
            self.obs_dict['act'] = self.sim.data.act[:].copy()
        
        self.obs_dict['tip_pos'] = np.array([])
        self.obs_dict['target_pos'] = np.array([])
        for isite in range(len(self.tip_sids)):
            self.obs_dict['tip_pos'] = np.append(self.obs_dict['tip_pos'], self.sim.data.site_xpos[self.tip_sids[isite]].copy())
            self.obs_dict['target_pos'] = np.append(self.obs_dict['target_pos'], self.sim.data.site_xpos[self.target_sids[isite]].copy())
        self.obs_dict['reach_err'] = np.array(self.obs_dict['target_pos'])-np.array(self.obs_dict['tip_pos'])
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
        com_v = np.sum(vel *  mass.reshape((-1, 1)), axis=0) / np.sum(mass)
        self.obs_dict['com_v'] = com_v[-3:]
        com = np.sum(pos * mass.reshape((-1, 1)), axis=0) / np.sum(mass)
        self.obs_dict['com'] = com[:2]
        self.obs_dict['base_support'] =  [x, y]
        self.obs_dict['pose_err'] = self.target_jnt_value - self.obs_dict['qpos']

        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time])
        obs_dict['qpos'] = sim.data.qpos[:].copy()
        """
        for i in range(sim.model.njnt):
            # Get the starting index for the joint name in the `names` array
            start_idx = sim.model.name_jntadr[i]
            # Extract the name
            name = ''
            while sim.model.names[start_idx] != 0:  # 0 is the null terminator in the names array
                name += chr(sim.model.names[start_idx])
                start_idx += 1
            print(f"Joint {i}: {name}")
        """
        obs_dict['qvel'] = sim.data.qvel[:].copy()*self.dt
        obs_dict['act'] = sim.data.act[:].copy() if sim.model.na>0 else np.zeros_like(obs_dict['qpos'])
        obs_dict['pose_err'] = self.target_jnt_value - obs_dict['qpos'][:21]
        obs_dict['tip_pos'] = np.array([])
        obs_dict['target_pos'] = np.array([])

        ### we append the target position of the two feet
        

        for isite in range(len(self.tip_sids)):
            obs_dict['tip_pos'] = np.append(obs_dict['tip_pos'], sim.data.site_xpos[self.tip_sids[isite]].copy())
            obs_dict['target_pos'] = np.append(obs_dict['target_pos'], sim.data.site_xpos[self.target_sids[isite]].copy())
        obs_dict['reach_err'] = np.array(obs_dict['target_pos'])-np.array(obs_dict['tip_pos'])
        x, y = np.array([]), np.array([])
        for label in ['calcn_r', 'calcn_l', 'toes_l', 'toes_r']:
            xpos = np.array(sim.data.xipos[sim.model.body_name2id(label)].copy())[:2] # select x and y position of the current body
            x = np.append(x, xpos[0])
            y = np.append(y, xpos[1]) 
        obs_dict['base_support'] = np.append(x, y)
        #obs_dict['ver_sep'] = np.array(max(y), min(y))
        # CoM is considered to be the center of mass of the pelvis (for now) 
        pos = sim.data.xipos.copy()
        vel = sim.data.cvel.copy()
        mass = sim.model.body_mass
        com = np.sum(pos * mass.reshape((-1, 1)), axis=0) / np.sum(mass)
        com_v = np.sum(vel *  mass.reshape((-1, 1)), axis=0) / np.sum(mass)
        obs_dict['com_v'] = com_v[-3:]
        obs_dict['com'] = com[:2]
        obs_dict['com_height'] = com[-1:]# self.sim.data.body('pelvis').xipos.copy()
        baseSupport = obs_dict['base_support'].reshape(2,4)
        obs_dict['centroid'] = np.array(Polygon(zip(baseSupport[0], baseSupport[1])).centroid.coords)
        obs_dict['err_com'] = np.array(obs_dict['centroid']- obs_dict['com'])
        #print('compare', self.target_jnt_value, obs_dict['qpos'])
        return obs_dict

    def get_reward_dict(self, obs_dict):
        pose_dist = np.linalg.norm(obs_dict['pose_err'], axis=-1)
        act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)
        if self.sim.model.na !=0: act_mag= act_mag/self.sim.model.na
        far_th = np.pi/2
        positionError = np.linalg.norm(obs_dict['reach_err'], axis=-1)
        comError = np.linalg.norm(obs_dict['err_com'], axis=-1)
        #print(pose_dist)

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('positionError', np.exp(-.1*positionError) ),
            ('pose',    -1.*pose_dist),
            ('bonus',   1.*(pose_dist<self.pose_thd) + 1.*(pose_dist<1.5*self.pose_thd)),
            ('penalty', -1.*(pose_dist>far_th)),
            ('act_reg', -1.*act_mag),
            ('com_error',             np.exp(-2.*np.abs(comError))),
            # Must keys
            ('sparse',  -1.0*pose_dist),
            ('solved',  pose_dist<self.pose_thd),
            ('done',    pose_dist[0][0]>far_th),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    # generate a valid target pose
    def get_target_pose(self):
        if self.target_type == "fixed":
            return self.target_jnt_value
        elif self.target_type == "generate":
            return self.np_random.uniform(low=self.target_jnt_range[:,0], high=self.target_jnt_range[:,1])
        else:
            raise TypeError("Unknown Target type: {}".format(self.target_type))

    # update sim with a new target pose
    def update_target(self, restore_sim=False):
        if restore_sim:
            qpos = self.sim.data.qpos[:].copy()
            qvel = self.sim.data.qvel[:].copy()
        # generate targets
        self.target_jnt_value = self.get_target_pose()
        # update finger-tip target viz
        self.sim.data.qpos[:21] = self.target_jnt_value.copy()
        self.sim.forward()
        for isite in range(len(self.tip_sids)):
            self.sim.model.site_pos[self.target_sids[isite]] = self.sim.data.site_xpos[self.tip_sids[isite]].copy()
        if restore_sim:
            self.sim.data.qpos[:] = qpos[:]
            self.sim.data.qvel[:] = qvel[:]
        self.sim.forward()

    # reset_type = none; init; random
    # target_type = generate; switch
    def reset(self, **kwargs):

        # udpate wegith
        if self.weight_bodyname is not None:
            bid = self.sim.model.body_name2id(self.weight_bodyname)
            gid = self.sim.model.body_geomadr[bid]
            weight = self.np_random.uniform(low=self.weight_range[0], high=self.weight_range[1])
            self.sim.model.body_mass[bid] = weight
            self.sim_obsd.model.body_mass[bid] = weight
            # self.sim_obsd.model.geom_size[gid] = self.sim.model.geom_size[gid] * weight/10
            self.sim.model.geom_size[gid][0] = 0.01 + 2.5*weight/100
            # self.sim_obsd.model.geom_size[gid][0] = weight/10

        # update target
        if self.target_type == "generate":
            # use target_jnt_range to generate targets
            self.update_target(restore_sim=True)
        elif self.target_type == "switch":
            # switch between given target choices
            # TODO: Remove hard-coded numbers
            if self.target_jnt_value[0] != -0.145125:
                self.target_jnt_value = np.array([-0.145125, 0.92524251, 1.08978337, 1.39425813, -0.78286243, -0.77179383, -0.15042819, 0.64445902])
                self.sim.model.site_pos[self.target_sids[0]] = np.array([-0.11000209, -0.01753063, 0.20817679])
                self.sim.model.site_pos[self.target_sids[1]] = np.array([-0.1825131, 0.07417956, 0.11407256])
                self.sim.forward()
            else:
                self.target_jnt_value = np.array([-0.12756566, 0.06741454, 1.51352705, 0.91777418, -0.63884237, 0.22452487, 0.42103326, 0.4139465])
                self.sim.model.site_pos[self.target_sids[0]] = np.array([-0.11647777, -0.05180014, 0.19044284])
                self.sim.model.site_pos[self.target_sids[1]] = np.array([-0.17728016, 0.01489491, 0.17953786])
        elif self.target_type == "fixed":
            self.update_target(restore_sim=True)
        else:
            print("{} Target Type not found ".format(self.target_type))

        # update init state
        if self.reset_type is None or self.reset_type == "none":
            # no reset; use last state
            ## NOTE: fatigue is also not reset in this case!
            obs = self.get_obs()
        elif self.reset_type == "init":
            # reset to init state
            obs = super().reset(**kwargs)
        elif self.reset_type == "random":
            # reset to random state
            jnt_init = self.np_random.uniform(high=self.sim.model.jnt_range[:,1], low=self.sim.model.jnt_range[:,0])
            obs = super().reset(reset_qpos=jnt_init, **kwargs)
        else:
            print("Reset Type not found")

        return obs