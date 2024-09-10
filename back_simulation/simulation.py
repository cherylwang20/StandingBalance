import numpy as np
import mujoco
import cv2
import os
import matplotlib.pyplot as plt

def create_vid(images):
    height, width, layers = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('vid_old.mp4', fourcc, 200, (width, height))
    for image in images:
        video.write(image)
    video.release()


def main():
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

    # ctrl = np.random.rand(210)
    # ctrl[21]=0
    # ctrl[22]=0
    # ctrl[185:191]=0
    # ctrl[197:203]=0

    ctrl = np.zeros(210)
    qpos_list = []

    for i in range(1000):
        data.ctrl = ctrl
        mujoco.mj_step(model, data)
        qpos_list.append(np.copy(data.qpos))
        renderer.update_scene(data, camera=camera_id)
        images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
    qpos_list = np.array(qpos_list)
    plt.plot(qpos_list[:,3], label="flex")
    plt.plot(qpos_list[:,9], label="L5")
    plt.plot(qpos_list[:,12], label="L4")
    plt.plot(qpos_list[:,15], label="L3")
    plt.plot(qpos_list[:,18], label="L2")
    plt.legend()
    plt.show()
    create_vid(images)
if __name__ == '__main__':
    main()