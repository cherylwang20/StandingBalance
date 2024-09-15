import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

stiffness = 1527.51

def parse_column(column):
    """Parse column data into a 2D numpy array."""
    return np.array([np.fromstring(x.strip('[]'), sep=',') for x in column])

def compute_and_plot_distances(folder_name):
    # List of column names to process
    columns_to_read = ['RL1', 'RL2', 'LL2', 'LL1']
    
    # Iterate through CSV files in the folder
    for i in range(1, 10):
        file_name = f'SUB{i}.csv'
        file_path = '/home/rwalia/MyoBack/back_simulation/plotting/Experiment_data/'+folder_name+"/"+file_name
        print(file_path)
        
        if os.path.isfile(file_path):
            # Read the CSV file
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

            mask = (dist_RL1_RL2 >= 2) & (dist_LL2_LL1 >= 2)
            
            time = np.arange(0, len(dist_RL1_RL2 ) * 0.01, 0.01)[mask]
            col2_filtered = dist_RL1_RL2[mask]
            col3_filtered = dist_LL2_LL1[mask]
            
            # Plot the distances
            plt.figure(figsize=(12, 6))
            plt.plot(np.array(range(len(dist_RL1_RL2)))/100, dist_RL1_RL2, label='Distance between RL1 and RL2', color='blue')
            plt.plot(np.array(range(len(dist_LL2_LL1)))/100, dist_LL2_LL1, label='Distance between LL2 and LL1', color='red')
            plt.xlabel('Time')
            plt.ylabel('Euclidean Distance')
            plt.title(f'Euclidean Distances for {file_name}')
            plt.legend()
            plt.grid(True)
            plt.show()

# Example usage
compute_and_plot_distances('aux_static_squat_110')