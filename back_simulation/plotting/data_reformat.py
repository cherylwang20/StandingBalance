import pandas as pd
import os

def excel_to_csv(excel_filename):
    # Extract the base name of the file (without extension)
    base_name = os.path.splitext(excel_filename)[0]
    
    # Create a directory named after the base name
    if not os.path.exists(base_name):
        os.makedirs(base_name)
    
    # Load the Excel file
    xls = pd.ExcelFile(excel_filename)
    
    # Iterate over sheet names and save each as a CSV file
    for i, sheet_name in enumerate(xls.sheet_names):
        if i >= 9:
            break  # Only process the first 9 sheets
        df = pd.read_excel(xls, sheet_name)
        csv_filename = os.path.join(base_name, f'SUB{i+1}.csv')
        df.to_csv(csv_filename, sep=";", index=False)
        print(f'Saved {csv_filename}')

# Example usage
if __name__ == "__main__":
    excel_to_csv("/home/rwalia/Downloads/OneDrive_1_9-10-2024/loadcelldata/lc_aux_dynamic_squat_0kg.xlsx")