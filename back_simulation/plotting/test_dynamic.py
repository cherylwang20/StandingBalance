import numpy as np
import mujoco
import cv2
import matplotlib.pyplot as plt
import csv

def create_vid(images):
    height, width, layers = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('vid_dynamic.mp4', fourcc, 200, (width, height))
    for image in images:
        video.write(image)
    video.release()

def main(res):
    images = []
    height = 480
    width = 640
    camera_id = "front_camera"

    model_path = './myosuite/myosuite/simhive/myo_sim/back/myobacklegs-Exoskeleton.xml'
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Activer la visibilité du groupe 1
    vopt = mujoco.MjvOption()
    # Activer la visibilité du groupe 1
    vopt.geomgroup[4] = 1  # 1 pour activer, 0 pour désactiver
    # Appliquer les options de visualisation par défaut
    mujoco.mjv_defaultOption(vopt)

    target_group = 1  # This is the geom group you want to modify (make invisible)
    # Loop through all geoms
    for i in range(model.ngeom):
        # Check if the geom belongs to the target group
        if model.geom_group[i] == target_group:
            # Only change the alpha (last) value of the RGBA, leave the RGB unchanged
            rgba = model.geom_rgba[i]
            rgba[3] = 0  # Set alpha to 0 (make the geom invisible)
            model.geom_rgba[i] = rgba

    renderer = mujoco.Renderer(model, height=height, width=width)
    
    renderer.update_scene(data, camera=camera_id)
    images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

    print(len(data.qpos))

    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)  # Get joint name
        qpos_address = model.jnt_qposadr[joint_id]  # Get the qpos index for the joint
        print(f"Joint Name: {joint_name}, qpos index: {qpos_address}")

    kf = model.keyframe('default-pose')
    data.qpos = kf.qpos

    ctrl_data = np.zeros(len(data.ctrl))
        
    for i in range(res):
        data.ctrl = ctrl_data
        mujoco.mj_step(model, data)

        flex = data.qpos[10]
        lat = data.qpos[11]
        axial = data.qpos[12]

        data.qpos[7] = 0.03305523 * flex
        data.qpos[8] = 0.01101841 * flex
        data.qpos[9] = 0.6 * flex
        data.qpos[13] = 0.03 * flex
        data.qpos[14] = 3.89329927e-04 * flex**4 - 4.18762151e-03 * flex**3 - 1.86233838e-02 * flex**2 + 5.78749087e-02 * flex
        data.qpos[15] = 0.64285726 * flex
        data.qpos[16] = 0.185 * flex
        data.qpos[19] = 0.204 * flex
        data.qpos[22] = 0.231 * flex
        data.qpos[25] = 0.255 * flex
        data.qpos[17] = 0.181 * lat
        data.qpos[20] = 0.245 * lat
        data.qpos[23] = 0.250 * lat
        data.qpos[26] = 0.188 * lat
        data.qpos[18] = 0.0378 * axial
        data.qpos[22] = 0.0378 * axial
        data.qpos[24] = 0.0311 * axial
        data.qpos[27] = 0.0289 * axial

        data.qpos[10] = 0
        data.qpos[11] = 0
        data.qpos[12] = 0

        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera_id)
        images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

        data.qpos[10] = flex
        data.qpos[11] = lat
        data.qpos[12] = axial
    

    create_vid(images)


if __name__ == '__main__':
    main(res=500)


 
