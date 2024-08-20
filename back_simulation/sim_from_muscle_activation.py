import numpy as np
import mujoco
import cv2
import pandas as pd
import matplotlib.pyplot as plt
#export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

def create_vid(images, fps):
    height, width, layers = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('back_simulation/vid.mp4', fourcc, fps, (width, height))
    for image in images:
        video.write(image)
    video.release()

def get_ctrl_data(mot_file_path):
    with open(mot_file_path, 'r') as file:
        for i, line in enumerate(file):
            if "endheader" in line:
                header_lines_to_skip = i + 1
                break
    df = pd.read_csv(mot_file_path, delim_whitespace=True, skiprows=header_lines_to_skip)
    filtered_df = df.filter(regex='(time|activation)', axis=1)
    data = filtered_df.to_numpy()
    return data

def main(mot_file_path, fps):
    ## Setup
    ctrl_data = get_ctrl_data(mot_file_path)
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

    plt.boxplot(np.diff(ctrl_data[:,0]))
    plt.title("Boxplot of time steps")
    plt.ylabel("Time")
    plt.show()
    model.opt.timestep = np.median(np.diff(ctrl_data[:,0]))
    print(model.opt.timestep)
    print(len(ctrl_data))

    for i in range(len(ctrl_data)):
        data.ctrl = ctrl_data[i][1:]
        mujoco.mj_step(model, data)
        if i%(int(0.005/model.opt.timestep)):
            renderer.update_scene(data, camera=camera_id)
            images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

    print(data.time)
    create_vid(images, fps)

if __name__ == '__main__':
    main(mot_file_path = 'back_simulation/Results_TestA_40DegrFlex_20_08.sto', fps=10)