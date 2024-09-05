import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mujoco

mj_data = np.load('muscle_forces_mj_flex_extension.npy')
osim_data = np.load('muscle_forces_osim_flex_extension.npy')
model_path = 'myosuite/myosuite/simhive/myo_sim/back/myoback_v2.0.xml'
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
muscle_name = "rect_abd_l"
muscle_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, muscle_name)
print(muscle_index)
#print(mj_data[:,muscle_index])
joint_val_mj = np.linspace(-1.222, 0.4538, len(mj_data))
joint_val_osim = np.linspace(-1.222, 0.4538, len(osim_data))
#plt.plot(joint_val_mj, mj_data[:,muscle_index])
plt.plot(joint_val_osim, osim_data[:,muscle_index])
plt.show()