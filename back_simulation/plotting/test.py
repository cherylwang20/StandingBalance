import numpy as np
import mujoco
import cv2
import matplotlib.pyplot as plt
import csv

def create_vid(images):
    height, width, layers = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('vid.mp4', fourcc, 200, (width, height))
    for image in images:
        video.write(image)
    video.release()

def main(joint, res):
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
    
    # Obtenez l'identifiant des tendons à partir de leurs noms
    tendon_id_1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, "Exo_LS_RL")
    tendon_id_2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, "Exo_RS_LL")
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'rect_abd_r')


    mujoco.mj_forward(model, data)

    print(data.ten_length[tendon_id_1])
    print(data.ten_length[tendon_id_2])

    # Récupérez les propriétés des tendons
    #tendon_1 = model.tendon[tendon_id_1]
    #tendon_2 = model.tendon[tendon_id_2]    
    # La rigidité est généralement stockée dans les attributs de tendons, comme `stiffness`
    # Vérifiez la documentation pour les attributs exacts disponibles pour le tendon
    #stiffness_1 = tendon_1.stiffness
    #stiffness_2 = tendon_2.stiffness
    
    stiffness_1 = model.tendon_stiffness[tendon_id_1]
    stiffness_2 = model.tendon_stiffness[tendon_id_2]
    print(stiffness_1)
    print(stiffness_2)

    # Nom du site
    top_site_name = "Exo_RightShoulder"
    bottom_site_name = "Exo_LeftLeg"


    num_actuators = model.nu
    exo_forces = []
    muscle_length = []
    distances = []

    # Initialize qpos_flex
    qpos_flex = np.zeros((res, model.nq))  # res is the number of steps, model.nq is the number of generalized coordinates
    qpos_flex[:, 2] = np.ones(res)
    qpos_flex[:, 3] = np.ones(res)*0.707388
    qpos_flex[:, 6] = np.ones(res)*(-0.706825)

    if joint == "flex_extension":
        joint_val = np.linspace(-1.222, 0, res)[::-1]

        qpos_flex[:, 7] = 0.03305523 * joint_val
        qpos_flex[:, 8] = 0.01101841 * joint_val
        qpos_flex[:, 9] = 0.6 * joint_val
        qpos_flex[:, 13] = 0.03*joint_val #(0.0008971 * joint_val**4 + 0.00427047 * joint_val**3 -
                           #0.01851051 * joint_val**2 - 0.05787512 * joint_val - 0.00800539) + np.sin(0.64285726 * joint_val)*0.04
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

        
    for i in range(res):
        data.qpos = qpos_flex[i]
        mujoco.mj_forward(model, data)
        #tendon_length_1 = model.tendon_length0[tendon_id_1]
        #tendon_length_2 = model.tendon_length0[tendon_id_2]
        tendon_length_1 = data.ten_length[tendon_id_1]
        tendon_length_2 = data.ten_length[tendon_id_2]
        tendon_force_1=(tendon_length_1-0.4264202995590148)*stiffness_1
        tendon_force_2=(tendon_length_2-0.4264202995590148)*stiffness_2
        exo_forces.append([tendon_force_1, tendon_force_2])
        renderer.update_scene(data, camera=camera_id)
        images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
    

    exo_forces = np.array(exo_forces)
    plt.plot(joint_val, exo_forces[:,0], label =  'Exo_LS_RL')
    plt.plot(joint_val, exo_forces[:,1], label =  'Exo_RS_LL')
    plt.legend()
    plt.show()

    np.save("exo_forces_{}".format(joint), exo_forces)
    create_vid(images)


if __name__ == '__main__':
    main(joint="flex_extension", res=500)


 
