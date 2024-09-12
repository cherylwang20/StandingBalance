import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('C:/Users/morga/MyoBack/back_simulation/plotting')
import get_average_exp_static_loadcell 
import math
import cv2

def plot_exo_forces(joint):
    # Charger les forces enregistrées
    exo_forces = np.load("exo_forces_{}.npy".format(joint))

    # Extraire les forces des deux actionneurs
    forces_actuator_1 = exo_forces[:, 0]
    forces_actuator_2 = exo_forces[:, 1]

    exp_values_40_g=get_average_exp_static_loadcell.get_data_side1('lc_static_stoop_40')
    exp_values_60_g=get_average_exp_static_loadcell.get_data_side1('lc_static_stoop_60')
    exp_values_80_g=get_average_exp_static_loadcell.get_data_side1('lc_static_stoop_80')
    angle = [-math.radians(40),-math.radians(60),-math.radians(80)]
    std40_g = np.std(exp_values_40_g)
    mean40_g= np.mean(exp_values_40_g)
    std60_g = np.std(exp_values_60_g)
    mean60_g= np.mean(exp_values_60_g)
    std80_g = np.std(exp_values_80_g)
    mean80_g= np.mean(exp_values_80_g)
    means_g=[mean40_g,mean60_g,mean80_g]
    stds_g=[std40_g,std60_g,std80_g]

    exp_values_40_d=get_average_exp_static_loadcell.get_data_side2('lc_static_stoop_40')
    exp_values_60_d=get_average_exp_static_loadcell.get_data_side2('lc_static_stoop_60')
    exp_values_80_d=get_average_exp_static_loadcell.get_data_side2('lc_static_stoop_80')
    angle = [-math.radians(40),-math.radians(60),-math.radians(80)]
    std40_d = np.std(exp_values_40_d)
    mean40_d= np.mean(exp_values_40_d)
    std60_d = np.std(exp_values_60_d)
    mean60_d= np.mean(exp_values_60_d)
    std80_d = np.std(exp_values_80_d)
    mean80_d= np.mean(exp_values_80_d)
    means_d=[mean40_d,mean60_d,mean80_d]
    stds_d=[std40_d,std60_d,std80_d]
    
    res=500
    # Créer un vecteur pour l'axe des x (index des étapes de simulation)
    x = np.linspace(-1.222, 0.4, res)[::-1]

    # Tracer les forces
    plt.figure(figsize=(10, 6))
    plt.ylim(0, 100)  # Limites de l'axe y de 0 à 100
    plt.plot(x, forces_actuator_1, label='Exo_LS_RL_Actuator Force', color='blue')
    plt.plot(x, forces_actuator_2, label='Exo_RS_LL_Actuator Force', color='red')

    plt.errorbar(angle, means_g, yerr=stds_g, fmt='-o', label='Experimental values left')
    plt.errorbar(angle, means_d, yerr=stds_d, fmt='-o', label='Experimental values right')   

    # Ajouter des titres et des légendes
    plt.title("Exoskeleton Forces for Joint: {}".format(joint))
    plt.xlabel("Simulation Step")
    plt.ylabel("Force (N)")
    plt.legend()
    
    # Afficher le graphique
    plt.show()

if __name__ == '__main__':
    plot_exo_forces(joint="squat")
