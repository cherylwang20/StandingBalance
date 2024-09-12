def individualMuscleMAPlot(muscle, joints, nJnt, joint_ranges_osim, joint_ranges_mjc, ma_mat_osim, ma_mat_mjc, nEval_osim, nEval_mjc):
    """"
    Plot individual muscle moment arm curves. Osim and Mjc are plotted side by side
    for the situation that multiple joints are coupling in the plot joint moment arm.
    In this case, a number of mesh points are checked on these coupling joints. And in
    this case, plot Osim and Mjc moment arms on top of each other.
    """

    # super title
    supTitle = muscle
    for joint in joints:
        supTitle = supTitle + " - " + joint
    
    for ij, joint in enumerate(joints):

        joint_range_osim = joint_ranges_osim[ij]
        joint_range_mjc = joint_ranges_mjc[ij]
        
        f = plt.figure(figsize=(10, 8))

        ax1 = f.add_subplot(1, 2, 1)
        ax2 = f.add_subplot(1, 2, 2)

        x_osim = np.linspace(joint_range_osim[0], joint_range_osim[1], nEval_osim, endpoint=True)
        x_mjc = np.linspace(joint_range_mjc[0], joint_range_mjc[1], nEval_mjc, endpoint=True)

        # find the minimal and maximum values in mjc and osim ma data, to set the axis equal
        # osim and mjc model have oppsite signs in moment arms
        max_ma = np.maximum(ma_mat_osim[ij, :, :].max(), -ma_mat_mjc[ij, :, :].min())
        min_ma = np.minimum(ma_mat_osim[ij, :, :].min(), -ma_mat_mjc[ij, :, :].max())
        
        for c in range(nEval_osim**(nJnt-1)):
            line_color = tuple([c, c, nEval_mjc**(nJnt-1)]/np.sqrt(2*c**2 + (nEval_mjc**(nJnt-1))**2))
            ax1.plot(x_osim, ma_mat_osim[ij, :, c]*100, marker = "s", color=line_color)

        ax1.set_ylabel("moment arms (cm)")
        ax1.set_xlabel(joint + " (rad)")
        ax1.set_ylim([min_ma*100, max_ma*100])
        ax1.set_title("OSIM")

        for c in range(nEval_mjc**(nJnt-1)):
            line_color = tuple([c, c, nEval_mjc**(nJnt-1)]/np.sqrt(2*c**2 + (nEval_mjc**(nJnt-1))**2))
            ax2.plot(x_mjc, -ma_mat_mjc[ij, :, c]*100, marker = "s", color=line_color)

        ax2.set_ylabel("moment arms (cm)")
        ax2.set_xlabel(joint + " (rad)")
        ax2.set_ylim([min_ma*100, max_ma*100])
        ax2.set_title("MJC")

        plt.suptitle(supTitle)

        f.savefig(self.save_path + '/moment_arms/' + muscle + '_' + joint + '.svg', format = "svg")
        plt.close(f)

def compMomentArmResults(mjc_model_path = None):
    """
    Plot the moment arm results before and after the above optimization step

    Parameters
    ----------
    mjc_model_path: string, optional
        mujoco model path
    
    Returns
    -------
    None.

    """

    os.makedirs(self.save_path + '/moment_arms', exist_ok = True)

    # load mujoco model if given.
    if mjc_model_path:
        mjc_model = mujoco.MjModel.from_xml_path(mjc_model_path)
    else:
        mjc_model_path = self.mjc_model_path[0:-8] + 'cvt2.xml'
        mjc_model = mujoco.MjModel.from_xml_path(mjc_model_path)

    if self.muscle_list:
        muscle_list = self.muscle_list  # if selected muscle given, only validate these
    else:
        # Scan the result folder to find all the <muscle_name>.pkl files and plot all of them
        muscle_list = [os.path.split(f)[1][0:-4] for f in glob.glob(self.save_path + "/*.pkl")]

    # find all the joints from mjc model
    joint_list = [mujoco.mj_id2name(mjc_model, mujoco.mjtObj.mjOBJ_JOINT, idx) for idx in range(mjc_model.njnt)]

    rMax = len(muscle_list)
    cMax = len(joint_list)
    
    cost_org_mat = np.zeros((rMax, cMax))
    cost_opt_mat = np.zeros((rMax, cMax))
    
    for i_mus, muscle in enumerate(muscle_list):
        # load the mus_para data from the saved files
        with open(self.save_path + '/' + muscle + '.pkl', "rb+") as muscle_file:
            muscle_para_opt = pickle.load(muscle_file)

            # Generate the moment arm curves for comparison plots. Moment arm of Osim model
            # will just use the reference data that pre-generated for optimization. Moment arm
            # of Mjc model will be regenerated to increase the mesh density. This is to check
            # if there are any random jumpping paths (that were not covered by the optimization)

            mjc_ma_data= []

            for ij, joints in enumerate(muscle_para_opt['wrapping_coordinates']):

                nEval = muscle_para_opt['evalN'][ij]
                nJnt = len(joints)
                jntRanges_mjc = muscle_para_opt['mjc_coordinate_ranges'][ij]
                jntRanges_osim = muscle_para_opt['osim_coordinate_ranges'][ij]

                # sort the osim moment arms into matrices
                ma_mat_osim = self.maVectorSort(muscle_para_opt['osim_ma_data'][ij], nJnt, nEval)

                # calculate mujoco moment arms and sort it into matrices
                ma_vec_mjc = computeMomentArmMuscleJoints(mjc_model, muscle, joints,\
                                                    jntRanges_mjc, nEval)
                
                ma_mat_mjc = self.maVectorSort(ma_vec_mjc, nJnt, nEval)

                # plot the moment arm plots
                self.individualMuscleMAPlot(muscle, joints, nJnt, jntRanges_osim, jntRanges_mjc, \
                                            ma_mat_osim, ma_mat_mjc, nEval, nEval)
                
                mjc_ma_data.append(ma_vec_mjc)
                        
                for joint in joints:
                    if not joint in joint_list:
                        joint_list.append(joint)

                    jnt_index = joint_list.index(joint)

                    if muscle_para_opt['opt_results'][ij]:  # if results exist
                        cost_org_mat[i_mus, jnt_index] = muscle_para_opt['opt_results'][ij]['cost_org']
                        cost_opt_mat[i_mus, jnt_index] = muscle_para_opt['opt_results'][ij]['cost_opt']

            muscle_para_opt['mjc_ma_data'] = mjc_ma_data

            pickle.dump(muscle_para_opt, muscle_file)

    # save the overall rms differences between moment arms
    rms_saving = {}
    rms_saving['cost_org_mat'] = cost_org_mat
    rms_saving['cost_opt_mat'] = cost_opt_mat
    nonzero_error_array = cost_opt_mat[cost_opt_mat != 0]
    rms_saving['cost_opt_mat_mean'] = np.mean(nonzero_error_array)
    rms_saving['cost_opt_mat_std'] = np.std(nonzero_error_array)

    # save the data based to the muscle parameter file
    with open(self.save_path + '/overall_comp_momentarms.pkl', 'wb') as rms_saving_file:
        pickle.dump(rms_saving, rms_saving_file)

    # generate the overall heat map                 
    self.heatMap(muscle_list, joint_list, cost_org_mat, cost_opt_mat)

def computeMomentArmMuscleJoints(mjc_model, muscle, joints, ang_ranges, evalN):
    '''
    Calculate moment arm matries from given muscles and joints
    '''
    
    if type(joints) != list:
        joints = [joints]
    
    mjc_data = mujoco.MjData(mjc_model)
    mjc_model.opt.timestep = 0.001 ## same time step as opensim
    
    muscles_idx = mujoco.mj_name2id(mjc_model, mujoco.mjtObj.mjOBJ_ACTUATOR, muscle)  # muscle index
        
    joints_idx = []
    for joint in joints:
        joints_idx.append(mujoco.mj_name2id(mjc_model, mujoco.mjtObj.mjOBJ_JOINT, joint))
        
    ang_meshes=[]
    
    for ij, joint in enumerate(joints):
        ang_meshes.append(np.linspace(ang_ranges[ij][0], ang_ranges[ij][1], evalN))

    mom_arm =[]
    #  for im, muscle in enumerate(muscles):  does not run through muscles, only one
    for setAngleDofs in itertools.product(*ang_meshes):
        mjc_data.qpos[:] = np.zeros(len(mjc_data.qpos),)
        mjc_data.qvel[:] = np.zeros(len(mjc_data.qvel),)
        mjc_data.qpos[joints_idx] = setAngleDofs
        
        # assign the value of dependency joints
        dependencyJnts, dependencyJntAngs, freeJnts =\
            dependencyJointAng(mjc_model, joints_idx, setAngleDofs)
        mjc_data.qpos[dependencyJnts] = dependencyJntAngs

        # assign the value of locked joints
        lockedJnts, lockedJointAngs = lockedJointAng(mjc_model)
        mjc_data.qpos[lockedJnts] = lockedJointAngs

        mujoco.mj_step(mjc_model, mjc_data)
            
        mom_arm_sub = mjc_data.actuator_moment[muscles_idx, joints_idx].copy()
                
        mom_arm.append(mom_arm_sub)
        
    return np.array(mom_arm)

def maVectorSort(self, ma_vec, nJnt, nEval):
    """
    This function sort the moment arm vector that generated by optimization procedures.
    Results of this sort function are joint specific lists that contains the moment arms
    of it. The number of list depends on the mesh points of the other coupled joints. The
    length of each list represent the checking points of this joint angle.

    In the optimization, itertools.product will generate the angle mesh list in this structure:
        [Ang11, Ang21, Ang31], [Ang11, Ang21, Ang32], ...
        [Ang11, Ang22, Ang31], [Ang11, Ang22, Ang32], ...
        ...
        [Ang12, Ang21, Ang31], [Ang12, Ang21, Ang32], ...
        ...
        ...
    """
    
    # run through each joint
    ma_mat = np.zeros((nJnt, nEval, nEval**(nJnt-1)))

    for nj in range(nJnt):  # run through joints
        njVec = ma_vec[nj]

        for ne in range(nEval):  # run through evaulation joints
            for ne2 in range(nEval**(nJnt-1)):
                ma_mat[nj, ne, ne2] = ma_vec[ne*nEval**(nJnt-1) + ne2][nj]

    return ma_mat