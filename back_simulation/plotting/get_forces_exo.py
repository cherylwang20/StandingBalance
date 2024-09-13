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
    model_path = 'myosuite/myosuite/simhive/myo_sim/back/myobacklegs-Exoskeleton.xml'
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    images = []
    height = 480
    width = 640
    camera_id = "front_camera"
    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data, camera=camera_id)
    images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

    kf = model.keyframe('default-pose')
    data.qpos = kf.qpos
    
    # Obtenez l'identifiant des tendons à partir de leurs noms
    tendon_id_1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, "Exo_LS_RL")
    tendon_id_2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, "Exo_RS_LL")

    mujoco.mj_forward(model, data)

    # Récupérez les propriétés des tendons
    #tendon_1 = model.tendon[tendon_id_1]
    #tendon_2 = model.tendon[tendon_id_2]    
    # La rigidité est généralement stockée dans les attributs de tendons, comme `stiffness`
    # Vérifiez la documentation pour les attributs exacts disponibles pour le tendon
    #stiffness_1 = tendon_1.stiffness
    #stiffness_2 = tendon_2.stiffness
    
    stiffness_1 = model.tendon_stiffness[tendon_id_1]
    stiffness_2 = model.tendon_stiffness[tendon_id_2]

    num_actuators = model.nu
    exo_forces = []

    # Initialize qpos_flex
    qpos_flex = np.zeros((res, model.nq))  # res is the number of steps, model.nq is the number of generalized coordinates

    if joint == "flex_extension":
        joint_val = np.linspace(-1.222, 0.4538, res)[::-1]

        qpos_flex[:, 0] = 0.03305523 * joint_val
        qpos_flex[:, 1] = 0.01101841 * joint_val
        qpos_flex[:, 2] = 0.6 * joint_val
        qpos_flex[:, 6] = (0.0008971 * joint_val**4 + 0.00427047 * joint_val**3 -
                           0.01851051 * joint_val**2 - 0.05787512 * joint_val - 0.00800539)
        qpos_flex[:, 7] = (3.89329927e-04 * joint_val**4 - 4.18762151e-03 * joint_val**3 -
                           1.86233838e-02 * joint_val**2 + 5.78749087e-02 * joint_val)
        qpos_flex[:, 8] = 0.64285726 * joint_val
        qpos_flex[:, 9] = 0.185 * joint_val
        qpos_flex[:, 12] = 0.204 * joint_val
        qpos_flex[:, 15] = 0.231 * joint_val
        qpos_flex[:, 18] = 0.255 * joint_val

    elif joint == "lat_bending":
        joint_val = np.linspace(-0.4363, 0.4363, res)

        qpos_flex[:, 10] = 0.181 * joint_val
        qpos_flex[:, 13] = 0.245 * joint_val
        qpos_flex[:, 16] = 0.250 * joint_val
        qpos_flex[:, 19] = 0.188 * joint_val

        qpos_flex[:, 4] = joint_val

    elif joint == "axial_rotation":
        joint_val = np.linspace(-0.7854, 0.7854, res)

        qpos_flex[:, 11] = 0.0378 * joint_val
        qpos_flex[:, 14] = 0.0378 * joint_val
        qpos_flex[:, 17] = 0.0311 * joint_val
        qpos_flex[:, 20] = 0.0289 * joint_val

        qpos_flex[:, 5] = joint_val

    if joint == "squat":
        joint_val = np.linspace(0, 2.09, res)

        qpos_flex[:, 3] = -0.384
        qpos_flex[:, 21] = joint_val
        qpos_flex[:, 26] = joint_val
        qpos_flex[:, 35] = joint_val
        qpos_flex[:, 40] = joint_val


    else:
        print("Select valid joint!")
        return

    with open('exo_forces.csv', 'w', newline='') as csvfile:
        # Créer un objet writer
        writer = csv.writer(csvfile)
        
        # Écrire l'en-tête du fichier CSV
        writer.writerow(['Exo_LS_RL', 'Exo_RS_LL'])
        
        for i in range(res):
            data.qpos = qpos_flex[i]
            mujoco.mj_forward(model, data)
            tendon_length_1 = data.ten_length[tendon_id_1]
            print("La longueur du tendon pour l'angle ",joint_val[i]," est ",tendon_length_1)
            tendon_length_2 = data.ten_length[tendon_id_1]
            tendon_force_1=(tendon_length_1-0.41017729885414067)*stiffness_1
            print("La force du tendon pour l'angle ",joint_val[i]," est ",tendon_force_1)
            tendon_force_2=(tendon_length_2-0.41017729885414067)*stiffness_2
            exo_forces.append([tendon_force_1, tendon_force_2])
            # Écrire la valeur des deux forces dans le fichier CSV
            writer.writerow([tendon_force_1, tendon_force_2])
            renderer.update_scene(data, camera=camera_id)
            images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

    exo_forces = np.array(exo_forces)
    np.save("exo_forces_{}".format(joint), exo_forces)
    create_vid(images)


if __name__ == '__main__':
    main(joint="flex_extension", res=500)
 
