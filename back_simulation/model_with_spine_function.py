import numpy as np
import mujoco
import cv2
import matplotlib.pyplot as plt
#export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
#<camera name = "front_camera" pos="0.127 -0.264 1.007" xyaxes="0.811 0.585 0.000 0.130 -0.180 0.975"/>
#<tendon limited="false" width="0.005" rgba="0.95 0.3 0.3 0"/>

def create_vid(images):
    height, width, layers = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('vid_dynamic.mp4', fourcc, 200, (width, height))
    for image in images:
        video.write(image)
    video.release()

def main(joint, res, visualize):
    ## Setup
    images = []
    height = 480
    width = 640
    camera_id = 'front_camera'
    model_path = 'myosuite/myosuite/simhive/myo_sim/back/myoback_v2.0.xml'
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    kf = model.keyframe('default-pose')
    data.qpos = kf.qpos
    mujoco.mj_forward(model, data)

    qpos_flex = np.zeros((res, model.nq))
    num_actuators = model.nu
    muscle_forces = []
    if joint=="flex_extension":
        joint_val = np.linspace(-1.222, 0.4538, res)

        qpos_flex[:,0] = 0.03305523 * joint_val
        qpos_flex[:,1] = 0.01101841 * joint_val
        qpos_flex[:,2] = 0.6 * joint_val
        qpos_flex[:,6] = 0.0008971 * joint_val**4 + 0.00427047 * joint_val**3 -0.01851051 * joint_val**2 - 0.05787512 * joint_val - 0.00800539
        qpos_flex[:,7] = 3.89329927e-04 * joint_val**4 - 4.18762151e-03 * joint_val**3 - 1.86233838e-02 * joint_val**2 + 5.78749087e-02 * joint_val
        qpos_flex[:,8] = 0.64285726 * joint_val
        qpos_flex[:,9] = 0.185 * joint_val
        qpos_flex[:,12] = 0.204 * joint_val
        qpos_flex[:,15] = 0.231 * joint_val
        qpos_flex[:,18] = 0.255 * joint_val

        # qpos_flex[:,3] = joint_val
    elif joint=="lat_bending":
        joint_val = np.linspace(-0.4363, 0.4363, res)

        qpos_flex[:,10] = 0.181 * joint_val
        qpos_flex[:,13] = 0.245 * joint_val
        qpos_flex[:,16] = 0.250 * joint_val
        qpos_flex[:,19] = 0.188 * joint_val

        qpos_flex[:,4] = joint_val
    elif joint=="axial_rotation":
        joint_val = np.linspace(-0.7854, 0.7854, res)

        qpos_flex[:,11] = 0.0378 * joint_val
        qpos_flex[:,14] = 0.0378 * joint_val
        qpos_flex[:,17] = 0.0311 * joint_val
        qpos_flex[:,20] = 0.0289 * joint_val

        qpos_flex[:,5] = joint_val
    else:
        print("Select valid joint!")
        return
    
    print(len(data.ctrl))
    ctrl_data = np.zeros(len(data.ctrl))

    qpos_list = []

    for i in range(res):
        data.ctrl = ctrl_data
        mujoco.mj_step(model, data)
        print(data.qpos[3])
        flex = data.qpos[3]
        data.qpos[0] = 0.03305523 * flex
        data.qpos[1] = 0.01101841 * flex
        data.qpos[2] = 0.6 * flex
        data.qpos[6] = 0.0008971 * flex**4 + 0.00427047 * flex**3 -0.01851051 * flex**2 - 0.05787512 * flex - 0.00800539
        data.qpos[7] = 3.89329927e-04 * flex**4 - 4.18762151e-03 * flex**3 - 1.86233838e-02 * flex**2 + 5.78749087e-02 * flex
        data.qpos[8] = 0.64285726 * flex
        data.qpos[9] = 0.185 * flex
        data.qpos[12] = 0.204 * flex
        data.qpos[15] = 0.231 * flex
        data.qpos[18] = 0.255 * flex
        data.qpos[3] = 0
        qpos_list.append(np.copy(data.qpos))
        mujoco.mj_forward(model, data)
        #muscle_forces.append(np.ctypeslib.as_array(data.qfrc_actuator, shape=(num_actuators,)))
        renderer.update_scene(data, camera=camera_id)
        images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
        data.qpos[3] = flex
    qpos_list = np.array(qpos_list)
    plt.plot(qpos_list[:,3]*0.185, label="flex")
    plt.plot(qpos_list[:,9], label="L5")
    plt.plot(qpos_list[:,12], label="L4")
    plt.plot(qpos_list[:,15], label="L3")
    plt.plot(qpos_list[:,18], label="L2")
    plt.legend()
    plt.show()

    #muscle_forces = np.array(muscle_forces)
    #np.save("muscle_forces_mj_{}".format(joint), muscle_forces)
    if visualize:
        create_vid(images)

if __name__ == '__main__':
    main(joint="flex_extension", res=1000, visualize=True)