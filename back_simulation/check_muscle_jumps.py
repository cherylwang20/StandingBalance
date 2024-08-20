import numpy as np
import mujoco
import cv2
import matplotlib.pyplot as plt
#export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

def detect_and_plot_jumps(name, array, threshold):
    print(name)
    differences = np.diff(array)
    jump_indices = np.where(np.abs(differences) > threshold)[0] + 1

    if len(jump_indices)!=0:
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
    model_path = 'myosuite/myosuite/simhive/myo_sim/back/myoback_v2.0.xml'
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    ## Move to grabbing position
    kf = model.keyframe('default-pose')
    data.qpos = kf.qpos
    mujoco.mj_forward(model, data)
   
    renderer.update_scene(data, camera=camera_id)
    images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

    qpos_flex = np.zeros((res, model.nq))
    num_actuators = model.nu
    muscle_lengths = []
    if joint=="flex_extension":
        joint_val = np.linspace(-1.222, 0.4538, res)
        qpos_flex[:,3] = joint_val
    elif joint=="lat_bending":
        joint_val = np.linspace(-0.4363, 0.4363, res)
        qpos_flex[:,4] = joint_val
    elif joint=="axial_rotation":
        joint_val = np.linspace(-0.7854, 0.7854, res)
        qpos_flex[:,5] = joint_val
    else:
        print("Select valid joint!")
        return

    for i in range(res):
        data.qpos = qpos_flex[i]
        mujoco.mj_forward(model, data)
        muscle_lengths.append(np.ctypeslib.as_array(data.actuator_length, shape=(num_actuators,)))
        renderer.update_scene(data, camera=camera_id)
        images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
    
    muscle_lengths = np.array(muscle_lengths)
    for i in range(num_actuators):
        if i!=num_actuators-1:
            detect_and_plot_jumps(model.names[model.name_actuatoradr[i]:model.name_actuatoradr[i+1]].decode(), muscle_lengths[:,i], threshold=threshold)
        else:
            detect_and_plot_jumps(model.names[model.name_actuatoradr[i]:len(model.names)].decode()[:-13], muscle_lengths[:,i], threshold=threshold)
    create_vid(images)

if __name__ == '__main__':
    main(joint="axial_rotation", res=1000, threshold=0.01)