import numpy as np
import mujoco

# Accès au modèle de dos
model_path = 'myosuite/myosuite/simhive/myo_sim/back/myoback_v2.0 _exo.xml'
model = mujoco.MjModel.from_xml_path(model_path)

# Récupération des données du modèle de dos
data = mujoco.MjData(model)

# Placement du modèle dans sa position par défault, à savoir debout le dos droit
kf = model.keyframe('default-pose')
data.qpos = kf.qpos
mujoco.mj_forward(model, data)

# Récupération de la longueur du tendon
tendon_name = "Exo_RS_LL"
tendon_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, tendon_name)
tendon_length = data.ten_length[tendon_index]
print(tendon_name, ":", tendon_length)


