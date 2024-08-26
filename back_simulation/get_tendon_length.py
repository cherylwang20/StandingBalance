import numpy as np
import mujoco
import cv2
import matplotlib.pyplot as plt

model_path = 'myosuite/myosuite/simhive/myo_sim/back/myoback_v2.0 _exo.xml'
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
kf = model.keyframe('default-pose')
data.qpos = kf.qpos
mujoco.mj_forward(model, data)
tendon_name = "Eco_RS_LL"
tendon_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, tendon_name)
tendon_length = data.ten_length[tendon_index]
print(tendon_name, ":", tendon_length)