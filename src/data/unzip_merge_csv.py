#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 01:44:02 2023

@author: frodo
"""

import os
import zipfile
import pandas as pd

def unzip_and_merge_csvs(base_folder):
    all_csvs = []
    zip_found = False
    csv_found = False

    # 1. Traverse the directory to find all subdirectories
    for subdir, _, _ in os.walk(base_folder):
        print(f"Checking subdirectory: {subdir}")
        
        # 2. In each subdirectory, find all .zip files
        for zip_file in [f for f in os.listdir(subdir) if f.endswith('.zip')]:
            zip_found = True
            zip_path = os.path.join(subdir, zip_file)
            print(f"Processing .zip file: {zip_path}")
            
            # 3. Unzip each .zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(subdir)
                for extracted_file in zip_ref.namelist():
                    print(f"Extracted: {extracted_file}")
                    
                    if extracted_file.endswith('.csv'):
                        csv_found = True
                        csv_path = os.path.join(subdir, extracted_file)
                        print(f"Reading .csv file: {csv_path}")
                        df = pd.read_csv(csv_path)
                        all_csvs.append(df)

    if not zip_found:
        print("No .zip files found in the provided directory and its subdirectories.")
        return

    if not csv_found:
        print("No .csv files found in the .zip files.")
        return

    # 4. Merge all the unzipped .csv files
    merged_df = pd.concat(all_csvs, ignore_index=True)
    #merged_df.to_csv(os.path.join(base_folder, "merged.csv"), index=False)

    print(f"All CSV files from {base_folder} and its subdirectories have been merged into merged.csv")
    
    return merged_df

if __name__ == "__main__":
    current_dir = os.getcwd()
    base_folder_path = os.path.join(current_dir, "data/raw/itineraries_csv")
    unzip_and_merge_csvs(base_folder_path)

