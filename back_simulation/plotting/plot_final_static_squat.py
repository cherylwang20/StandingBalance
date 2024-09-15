import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/rwalia/MyoBack/back_simulation/plotting')
import get_average_exp_static_loadcell 
import get_average_exp_static_marker
import math
import cv2

def plot_exo_forces(joint):
    # Charger les forces enregistrées
    exo_forces = np.load("exo_forces_static_squat.npy")

    # Extraire les forces des deux actionneurs
    forces_actuator_1 = exo_forces[:, 0]

    exp_values_70 = [22.704, 19.239, 15.339000000000002, 21.025, 21.351, 11.459999999999999, 18.845999999999997, 21.956000000000003, 16.723000000000003, 16.754, 19.782, 12.109000000000002, 15.656000000000002, 22.985, 12.860999999999999, 10.893, 11.681, 20.491, 15.019, 8.882999999999997, 11.575999999999999, 10.562, 23.737000000000002, 17.754000000000005, 19.228999999999996, 19.791999999999998, 12.987000000000002, 15.523000000000001, 22.711000000000002]
    exp_values_90 = [19.207999999999995, 21.393, 22.862000000000002, 22.629, 28.941999999999997, 26.909, 12.835999999999997, 11.401, 11.215, 29.622, 29.447, 26.86, 22.886000000000003, 23.407000000000004, 24.279000000000007, 25.3, 29.514000000000003, 16.130000000000003, 15.418000000000001, 15.802000000000001, 25.130000000000003, 29.633, 28.422, 14.405, 13.950000000000001, 17.277999999999995, 10.450000000000001, 26.632, 24.305, 19.663000000000004, 26.226000000000003, 29.358999999999998, 13.558000000000002]
    exp_values_110 = [32.68, 32.077, 26.729000000000003, 22.818, 22.894000000000002, 26.695999999999994, 28.798, 30.975, 30.218, 25.143000000000004, 18.138999999999996, 33.006, 34.849000000000004, 28.933, 30.372999999999998, 28.846, 28.231, 24.441000000000003, 7.7909999999999995, 12.695, 26.355000000000004, 21.231, 25.605999999999998, 21.597, 21.316000000000006, 28.260999999999996, 9.779000000000002, 12.853, 10.705000000000002, 26.64, 23.131, 12.018]
    angle_values_70 = [71.97, 76.24, 68.85, 68.72, 73.801, 69.21, 67.97, 71.33, 65.37, 72.29, 81.209, 65.418, 75, 67.21, 66.7, 74.13, 72.47, 75.733, 71.96, 69.78, 73.22]
    angle_values_90 = [91.58, 92.02, 82.07, 90.91, 89.36, 82.69, 81.95, 80.96, 83.36, 82.10, 85.99, 80.08, 91.79, 92.76, 82.76, 89.59, 88.85, 85.25, 83.30, 83.02, 80.64, 81.50, 82.38]
    angle_values_110 = [95.58, 97.38, 118.92, 95.31, 99.08, 95.94, 124.32, 101.46, 100.18, 97.05, 79.13, 99.93, 96.82, 98.39, 100.3, 104.1, 98.35, 95.85, 119.52, 100.9, 98.6, 101.35, 101.99, 95.64, 94.86, 94.84]
    exp_values_70_m = [10.92108474926947, 16.303458202921185, 12.566090567802021, 12.273412863395189, 8.278463834145972, 20.191315877372073,13.54079050593533, 16.832420920756284, 11.03340083416837]
    exp_values_90_m = [23.64719279550465, 25.82544110820422, 17.41008924960152, 14.558738882914048, 12.384633864669846, 15.188403995963622, 11.553212395018026, 11.150380912240252, 17.89777415022992]
    exp_values_110_m = [18.248533194762697, 20.228566298703157, 27.29257246901804, 15.334230534660481, 16.900615972152597, 13.086051477032354]

    std70 = np.std(exp_values_70)
    mean70= np.mean(exp_values_70)
    std90 = np.std(exp_values_90)
    mean90= np.mean(exp_values_90)
    std110 = np.std(exp_values_110)
    mean110= np.mean(exp_values_110)
    means =[mean70,mean90,mean110]
    stds =[std70,std90,std110]

    std70_m = np.std(exp_values_70_m)
    mean70_m= np.mean(exp_values_70_m)
    std90_m = np.std(exp_values_90_m)
    mean90_m= np.mean(exp_values_90_m)
    std110_m = np.std(exp_values_110_m)
    mean110_m= np.mean(exp_values_110_m)
    means_m=[mean70_m,mean90_m,mean110_m]
    stds_m=[std70_m,std90_m,std110_m]

    std70_angle = np.std(angle_values_70)
    mean70_angle = np.mean(angle_values_70)
    std90_angle = np.std(angle_values_90)
    mean90_angle = np.mean(angle_values_90)
    std110_angle = np.std(angle_values_110)
    mean110_angle = np.mean(angle_values_110)
    means_angle = [mean70_angle,mean90_angle,mean110_angle]
    stds_angle = [std70_angle,std90_angle,std110_angle]
    print(means_angle)
    res=500
    # Créer un vecteur pour l'axe des x (index des étapes de simulation)
    x = -(np.linspace(1.13446, 1.91986, res)[::-1])*(180/np.pi)

    # Tracer les forces
    plt.figure(figsize=(10, 6))
    plt.ylim(0, 40)  # Limites de l'axe y de 0 à 100
    plt.xlim(62, 113)
    plt.plot(-x, forces_actuator_1, label='Simulated Exoskeleton force', color='black')
    plt.errorbar(means_angle, means_m, xerr=stds_angle, yerr=stds_m, fmt='-o', label='Experimental marker force', color='tab:red')
    plt.errorbar(means_angle, means, xerr=stds_angle, yerr=stds, fmt='-o', label='Experimental loadcell force', color='tab:green')

    # Ajouter des titres et des légendes
    plt.xlabel("Knee angle [°]", fontdict={'fontname': 'Times New Roman'})
    plt.ylabel("Force [N]", fontdict={'fontname': 'Times New Roman'})
    plt.legend()
    plt.grid(True)
    
    # Afficher le graphique
    plt.savefig("static_squat.svg")

if __name__ == '__main__':
    plot_exo_forces(joint="squat")
