import numpy as np
import matplotlib.pyplot as plt

def plot_exo_forces(joint):
    # Charger les forces enregistrées
    exo_forces = np.load("exo_forces_{}.npy".format(joint))

    # Extraire les forces des deux actionneurs
    forces_actuator_1 = exo_forces[:, 0]
    forces_actuator_2 = exo_forces[:, 1]

    # Créer un vecteur pour l'axe des x (index des étapes de simulation)
    x = np.arange(len(forces_actuator_1))

    # Tracer les forces
    plt.figure(figsize=(10, 6))
    plt.plot(x, forces_actuator_1, label='Exo_LS_RL_Actuator Force', color='blue')
    plt.plot(x, forces_actuator_2, label='Exo_RS_LL_Actuator Force', color='red')
    
    # Ajouter des titres et des légendes
    plt.title("Exoskeleton Forces for Joint: {}".format(joint))
    plt.xlabel("Simulation Step")
    plt.ylabel("Force (N)")
    plt.legend()
    
    # Afficher le graphique
    plt.show()

if __name__ == '__main__':
    plot_exo_forces(joint="flex_extension")
