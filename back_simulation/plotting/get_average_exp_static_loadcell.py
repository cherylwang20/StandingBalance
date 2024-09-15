
import pandas as pd
import numpy as np
import os
from scipy.signal import find_peaks

# Dossier contenant les fichiers CSV
# directory = 'C:/Users/morga/MyoBack/back_simulation/plotting/Experiment_data/lc_static_stoop_40'  # Remplacez par le chemin de votre dossier

exp_files=[1,2,3,4,5,6,7,8,9]

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
            df = pd.read_csv(file_path, delimiter=';', decimal='.')
            
            if df.shape[1] < 3:
                print(f"Le fichier {file_path} n'a pas assez de colonnes.")
                continue
            
            # Step 2: Filter values in Column2 and Column3
            filtered_df = df[(df['Column2'] > 5) & (df['Column3'] > 5)]
            #print(filtered_df.head())
            # Extract filtered columns
            time = filtered_df['Column1'].values
            col2_filtered = filtered_df['Column2'].values
            col3_filtered = filtered_df['Column3'].values

            # Find large jumps in time to divide into three arrays
            jumps, _ = find_peaks(np.diff(time), height=2000)  # Adjust height as necessary
            #print(jumps)

            if len(jumps)==2:
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
                #print("Max values for Column2 after moving average:", col2_maxes)
                #print("Max values for Column3 after moving average:", col3_maxes)
                
                max_left.extend(col2_maxes)
                max_right.extend(col3_maxes)
            else:
                print("File ignored:", file_path)
            
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier {file_path}: {e}")
    print(max_left)
    print(max_right)
    return max_left, max_right
