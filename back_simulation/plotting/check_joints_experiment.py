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
            
            data_RLM = parse_column(df['RLM'])
            data_LLM = parse_column(df['LLM'])
            data_RMFE = parse_column(df['RMFE'])
            data_LMFE = parse_column(df['LMFE'])
            data_RASI = parse_column(df['RASI'])
            data_LASI = parse_column(df['LASI'])

            data_RL1 = parse_column(df['RL1'])
            data_RL2 = parse_column(df['RL2'])
            data_LL2 = parse_column(df['LL2'])
            data_LL1 = parse_column(df['LL1'])

            vect_RMFE_RASI = data_RMFE - data_RASI
            vect_LMFE_LASI = data_LMFE - data_LASI

            vect_RLM_RMFE = data_RLM-data_RMFE
            vect_LLM_LMFE = data_LLM-data_LMFE

            vect_flex_r = data_LL2-data_RL1
            vect_flex_l = data_RL2-data_LL1

            downward_vector = np.array([0, -1, 0])
            hip_right = np.arccos(np.dot(vect_RMFE_RASI, downward_vector) / (np.linalg.norm(vect_RMFE_RASI, axis=1) * np.linalg.norm(downward_vector)))
            hip_left = np.arccos(np.dot(vect_LMFE_LASI, downward_vector) / (np.linalg.norm(vect_LMFE_LASI, axis=1) * np.linalg.norm(downward_vector)))

            knee_right = np.arccos(np.dot(vect_RLM_RMFE, downward_vector) / (np.linalg.norm(vect_RLM_RMFE, axis=1) * np.linalg.norm(downward_vector)))
            knee_left = np.arccos(np.dot(vect_LLM_LMFE, downward_vector) / (np.linalg.norm(vect_LLM_LMFE, axis=1) * np.linalg.norm(downward_vector)))

            flex_right = np.arccos(np.dot(vect_flex_r, downward_vector) / (np.linalg.norm(vect_flex_r, axis=1) * np.linalg.norm(downward_vector)))
            flex_left = np.arccos(np.dot(vect_flex_l, downward_vector) / (np.linalg.norm(vect_flex_l, axis=1) * np.linalg.norm(downward_vector)))

            flex_right -= np.median(flex_right[:100], axis=0)
            flex_left -= np.median(flex_left[:100], axis=0)
            
            # Plot the distances
            plt.figure(figsize=(12, 6))
            plt.plot(np.degrees(hip_right), label='Angle between hip and knee (right)', color='blue')
            plt.plot(np.degrees(hip_left), label='Angle between hip and knee (left)', color='red')
            # plt.plot(np.degrees(knee_right), label='Angle between knee and foot (right)', color='blue')
            # plt.plot(np.degrees(knee_left), label='Angle between knee and foot (left)', color='red')
            # plt.plot(np.degrees(flex_right), label='Flex extension (right)', color='blue')
            # plt.plot(np.degrees(flex_left), label='Flex extension (left)', color='red')
            # plt.plot(np.array(range(len(dist_LL2_LL1)))/100, dist_LL2_LL1, label='Distance between LL2 and LL1', color='red')
            plt.xlabel('Time')
            plt.ylabel('Euclidean Distance')
            plt.title(f'Euclidean Distances for {file_name}')
            plt.legend()
            plt.grid(True)
            plt.show()

# Example usage
compute_and_plot_distances('aux_static_stoop_80')

