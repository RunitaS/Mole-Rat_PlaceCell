# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 10:31:02 2023

@author: shirdhankar
"""

"""
Code to extract time stamps from .smi files
"""

import re
import glob
import os
import pandas as pd

# Directory containing the .smi files to batch process
smi_dir = 'X:/NMR_group_data/Runita/Tracking/Fa5384Tracking'

smi_files = glob.glob(os.path.join(smi_dir, '*.smi'))

for smi_path in smi_files:
    # Read the .smi file
    with open(smi_path, 'r') as file:
        smi_content = file.read()

    # Use regular expression to find all 16-digit numbers
    timestamps = re.findall(r'\b\d{16}\b', smi_content)

    # Create a DataFrame
    df = pd.DataFrame({'time': timestamps})

    # Write to Excel file, same name and location as the source .smi file
    output_path = os.path.splitext(smi_path)[0] + '.xlsx'
    df.to_excel(output_path, index=False)

    print(f'Extracted {len(timestamps)} timestamps from {smi_path} -> {output_path}')

#%%

# # Read the CSV file
# df_second = pd.read_csv('C:/Runita/NMR/Data/FA8477/SAB/Open/19Aug22/VT1_DLC.csv')

# # Save the DataFrame to an Excel file
# excel_output_path = 'C:/Runita/NMR/Data/FA8477/SAB/Open/19Aug22/DLC.xlsx'
# df_second.to_excel(excel_output_path, index=False)

# # Read the Excel file
# df_second = pd.read_excel(excel_output_path)

# # Assuming 'Column_E' is the column name in the second Excel file
# # Add data from column E3 up to the end of the column E to Column B1 to the end of the column in the output DataFrame
# df['Column_B'] = df_second['Column_E'].iloc[2:]

# # Assuming 'Column_E' is the column name in the second Excel file
# # Add data from column E3 up to the end of the column E to Column B1 to the end of the column in the output DataFrame
# df['Column_C'] = df_second['Column_F'].iloc[2:]
