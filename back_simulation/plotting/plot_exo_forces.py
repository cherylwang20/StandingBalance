import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('C:/Users/morga/MyoBack/back_simulation/plotting')
import get_average_exp_static_loadcell 
import math

def plot_exo_forces(joint):
    # Charger les forces enregistrées
    exo_forces = np.load("exo_forces_{}.npy".format(joint))

    # Extraire les forces des deux actionneurs
    forces_actuator_1 = exo_forces[:, 0]
    forces_actuator_2 = exo_forces[:, 1]

    exp_values_1=get_average_exp_static_loadcell.get_data_side1()
    #angle_40= [math.radians(40)] * 9
    #angle_60= [math.radians(60)] * 9
    #angle_80= [math.radians(80)] * 9
    angle = [-math.radians(40),-math.radians(60),-math.radians(80)]
    means=[np.mean(exp_values_1[0]),np.mean(exp_values_1[1]),np.mean(exp_values_1[2])]
    stds = [np.std(exp_values_1[0]),np.std(exp_values_1[1]),np.std(exp_values_1[2])]
    
    # Créer un vecteur pour l'axe des x (index des étapes de simulation)
    x = np.linspace(-1.222, 0.4538, 1000)[::-1]

    # Tracer les forces
    plt.figure(figsize=(10, 6))
    plt.ylim(0, 100)  # Limites de l'axe y de 0 à 100
    plt.plot(x, forces_actuator_1, label='Exo_LS_RL_Actuator Force', color='blue')
    plt.plot(x, forces_actuator_2, label='Exo_RS_LL_Actuator Force', color='red')

    #plt.scatter(angle, means, color='green', marker='o', s=100, label='valeurs expérimentales 40 degrés')
    plt.errorbar(angle, means, yerr=stds, fmt='-o', label='valeurs expérimentales')
    #plt.scatter(angle_40, exp_values_1, color='green', marker='o', s=100, label='valeurs expérimentales 40 degrés')
    #plt.scatter(angle_60, exp_values_1[1], color='yellow', marker='o', s=100, label='valeurs expérimentales 60 degrés')
    #plt.scatter(angle_80, exp_values_1[2], color='purple', marker='o', s=100, label='valeurs expérimentales 80 degrés')
        
    # Ajouter des titres et des légendes
    plt.title("Exoskeleton Forces for Joint: {}".format(joint))
    plt.xlabel("Flexion extension angle, (flexion on the left)")
    plt.ylabel("Force (N)")
    plt.legend()
    
    # Afficher le graphique
    plt.show()

if __name__ == '__main__':
    plot_exo_forces(joint="flex_extension")
