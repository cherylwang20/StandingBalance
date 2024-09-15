import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_csv_files(folder_name):
    # Iterate over CSV files from SUB1.csv to SUB9.csv
    for i in range(1, 10):
        file_name = f'SUB{i}.csv'
        file_path = '/home/rwalia/MyoBack/back_simulation/plotting/Experiment_data/'+folder_name+"/"+file_name
        
        if not os.path.isfile(file_path):
            print(f"{file_path} does not exist.")
            continue
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path, delimiter=';', decimal='.')
        
        # Ensure the required columns exist in the DataFrame
        if {'Column1', 'Column2', 'Column3'}.issubset(df.columns):
            # Plot Column2 and Column3 against Column1
            plt.figure(figsize=(10, 6))
            plt.plot(df['Column1']/1000, df['Column2'], label='Loadcell 1', marker='o')
            plt.plot(df['Column1']/1000, df['Column3'], label='Loadcell 2', marker='x')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.title(f'Plot for {file_name}')
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            print(f"Missing required columns in {file_path}")


plot_csv_files('lc_aux_static_squat_70')
