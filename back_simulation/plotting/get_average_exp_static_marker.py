
import pandas as pd
import numpy as np
import os
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Dossier contenant les fichiers CSV
# directory = 'C:/Users/morga/MyoBack/back_simulation/plotting/Experiment_data/lc_static_stoop_40'  # Remplacez par le chemin de votre dossier

# Initialisation des matrices 3x9
matrix_col2 = np.zeros((3, 9))
matrix_col3 = np.zeros((3, 9))
stiffness = 1527.51
exp_files=[1,2,3,4,5,6,7,8,9]

def parse_column(column):
    """Parse column data into a 2D numpy array."""
    return np.array([np.fromstring(x.strip('[]'), sep=',') for x in column])

# Define a moving average function
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def get_data(folder): 

    directory = f'/home/rwalia/MyoBack/back_simulation/plotting/Experiment_data/{folder}'
    max_left = []
    max_right = []

    # Parcourir tous les fichiers CSV dans le dossier
    for i in exp_files:  # Pour SUB1 à SUB9
        file_path = os.path.join(directory, f'SUB{i}.csv')
        
        if not os.path.isfile(file_path):
            print(f"Fichier non trouvé: {file_path}")
            continue
        
        try:
            # Lire le fichier avec pandas
            df = pd.read_csv(file_path, delimiter=';', decimal=',')
            
            data_RL1 = parse_column(df['RL1'])
            data_RL2 = parse_column(df['RL2'])
            data_LL2 = parse_column(df['LL2'])
            data_LL1 = parse_column(df['LL1'])
            
            # Compute Euclidean distances
            dist_RL1_RL2 = np.sqrt(((data_RL1 - data_RL2) ** 2).sum(axis=1))/1000
            dist_LL2_LL1 = np.sqrt(((data_LL2 - data_LL1) ** 2).sum(axis=1))/1000

            dist_RL1_RL2 -= np.median(dist_RL1_RL2[:100], axis=0)
            dist_LL2_LL1 -= np.median(dist_LL2_LL1[:100], axis=0)

            dist_RL1_RL2 *= stiffness
            dist_LL2_LL1 *= stiffness

            mask = (dist_RL1_RL2 >= 1) & (dist_LL2_LL1 >= 1)
            
            time = np.arange(0, len(dist_RL1_RL2 ) * 0.01, 0.01)[mask]
            col2_filtered = dist_RL1_RL2[mask]
            col3_filtered = dist_LL2_LL1[mask]


            # Find large jumps in time to divide into three arrays
            jumps, _ = find_peaks(np.diff(time), height=2)  # Adjust height as necessary

            if len(jumps)==2:
                # Split the filtered data based on jumps
                split_indices = np.concatenate(([0], jumps, [len(time)]))
                col2_splits = [col2_filtered[split_indices[i]:split_indices[i+1]] for i in range(len(split_indices) - 1)]
                col3_splits = [col3_filtered[split_indices[i]:split_indices[i+1]] for i in range(len(split_indices) - 1)]

                # Calculate moving averages and find max values
                window_size = 100
                col2_maxes = []
                col3_maxes = []

                for col2, col3 in zip(col2_splits, col3_splits):
                    if len(col2) >= window_size and len(col3) >= window_size:
                        avg_col2 = moving_average(col2, window_size)
                        avg_col3 = moving_average(col3, window_size)
                        col2_maxes.append(np.max(avg_col2))
                        col3_maxes.append(np.max(avg_col3))

                # Print the results
                # print("Max values for Column2 after moving average:", col2_maxes)
                # print("Max values for Column3 after moving average:", col3_maxes)
            
                max_left.extend(col2_maxes)
                max_right.extend(col3_maxes)
            else:
                print("File ignored:", file_path)
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier {file_path}: {e}")
    return max_left, max_right
