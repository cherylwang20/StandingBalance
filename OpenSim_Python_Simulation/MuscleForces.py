import opensim as osim
import math
import numpy as np
import csv

# Cargar el modelo
model = osim.Model('Lumbar_C_210.osim')
state = model.initSystem()
joint = "lat_bending"
# Acceder a la coordenada del joint que quieres fijar
coord = model.getCoordinateSet().get(joint)

# Abrir un archivo CSV para escribir los datos

angle = -0.4363  # en radianes
increment = 0.4363  # en radianes
muscle_forces = []

while angle <= 0.7854:
    degrees = math.degrees(angle)
    coord.setValue(state, angle)
    coord.setSpeedValue(state, 0)

    model.realizeDynamics(state)

    temp =[]

    for muscle in model.getMuscles():
        muscle_name = muscle.getName()
        force = muscle.getActuation(state)  # Acceder a la fuerza del músculo
        temp.append(force)
    muscle_forces.append(temp)
    angle += increment
muscle_forces = np.array(muscle_forces)
print(muscle_forces.shape)
np.save("muscle_forces_osim_{}".format(joint), muscle_forces)