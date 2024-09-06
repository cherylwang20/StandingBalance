import numpy as np
import mujoco
import cv2
import matplotlib.pyplot as plt
#export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

def detect_and_plot_jumps(name, array, threshold):
    differences = np.diff(array)
    jump_indices = np.where(np.abs(differences) > threshold)[0] + 1

    if len(jump_indices)!=0:
        print(name)
        plt.figure(figsize=(10, 6))
        plt.plot(array, marker='o', linestyle='-', color='b', label='Muscle length progression')
        plt.scatter(jump_indices, array[jump_indices], color='r', label='Jumps', zorder=5)
        
        plt.xlabel('Step')
        plt.ylabel(f'{name} length')
        plt.title('Highlighted Jumps')
        plt.legend()
        
        plt.grid(True)
        plt.show()

def create_vid(images):
    height, width, layers = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('vid.mp4', fourcc, 200, (width, height))
    for image in images:
        video.write(image)
    video.release()


def main(joint, res, threshold):
    ## Setup
    images = []
    height = 480
    width = 640
    camera_id = 'front_camera'
    model_path = './myosuite/myosuite/simhive/myo_sim/back/myobacklegs-Exoskeleton.xml'
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    ## Move to grabbing position
    kf = model.keyframe('default-pose')
    data.qpos = kf.qpos
    mujoco.mj_forward(model, data)
   
    renderer.update_scene(data, camera=camera_id)
    images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

    qpos_flex = np.zeros((res, model.nq))  # res is the number of steps, model.nq is the number of generalized coordinates
    muscle_lengths = []
    qpos_flex[:, 2] = np.ones(res)
    qpos_flex[:, 3] = np.ones(res)*0.707388
    qpos_flex[:, 6] = np.ones(res)*(-0.706825)

    if joint == "flex_extension":
        joint_val = np.linspace(-1.222, 0.4, res)[::-1]

        qpos_flex[:, 7] = 0.03305523 * joint_val
        qpos_flex[:, 8] = 0.01101841 * joint_val
        qpos_flex[:, 9] = 0.6 * joint_val
        qpos_flex[:, 13] = (0.0008971 * joint_val**4 + 0.00427047 * joint_val**3 -
                           0.01851051 * joint_val**2 - 0.05787512 * joint_val - 0.00800539)
        qpos_flex[:, 14] = (3.89329927e-04 * joint_val**4 - 4.18762151e-03 * joint_val**3 -
                           1.86233838e-02 * joint_val**2 + 5.78749087e-02 * joint_val)
        qpos_flex[:, 15] = 0.64285726 * joint_val
        qpos_flex[:, 16] = 0.185 * joint_val
        qpos_flex[:, 19] = 0.204 * joint_val
        qpos_flex[:, 22] = 0.231 * joint_val
        qpos_flex[:, 25] = 0.255 * joint_val

    elif joint == "lat_bending":
        joint_val = np.linspace(-0.4363, 0.4363, res)

        qpos_flex[:, 17] = 0.181 * joint_val
        qpos_flex[:, 20] = 0.245 * joint_val
        qpos_flex[:, 23] = 0.250 * joint_val
        qpos_flex[:, 26] = 0.188 * joint_val

    elif joint == "axial_rotation":
        joint_val = np.linspace(-0.7854, 0.7854, res)

        qpos_flex[:, 18] = 0.0378 * joint_val
        qpos_flex[:, 22] = 0.0378 * joint_val
        qpos_flex[:, 24] = 0.0311 * joint_val
        qpos_flex[:, 27] = 0.0289 * joint_val

    else:
        print("Select valid joint!")
        return

    muscle_lengths = []

    for i in range(res):
        # Set qpos and run forward dynamics
        data.qpos = qpos_flex[i]
        mujoco.mj_forward(model, data)

        # Collect tendon lengths instead of actuator lengths
        muscle_lengths.append(np.copy(data.ten_length))
        # Render the scene
        renderer.update_scene(data, camera=camera_id)
        images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

    # Convert muscle_lengths to a numpy array for easier processing
    muscle_lengths = np.array(muscle_lengths)

    # Loop through the tendons and detect/plot jumps
    for i in range(model.ntendon):
        if i!=model.ntendon-1:
            tendon_name = model.names[model.name_tendonadr[i]:model.name_tendonadr[i+1]].decode()
            detect_and_plot_jumps(tendon_name, muscle_lengths[:, i], threshold=threshold)
        else:
            tendon_name = model.names[model.name_tendonadr[i]:len(model.names)].decode()[:-13]
            detect_and_plot_jumps(tendon_name, muscle_lengths[:, i], threshold=threshold)

    create_vid(images)

if __name__ == '__main__':
    main(joint="flex_extension", res=1000, threshold=0.01)