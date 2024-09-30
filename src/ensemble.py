import os
import pandas as pd

# List of specific subfolders that you want to process
selected_folders = ['GPT-4o-mini', 'InternVL2-8B-AWQ']  # Example list of subfolders

# Path to the parent folder containing the subfolders
result_folder = 'Results'

# Initialize lists to store file paths for public and private results
public_result_files = []
private_result_files = []

# Iterate over each folder in the provided list
for folder in selected_folders:
    folder_path = os.path.join(result_folder, folder)  # Create full path to the subfolder
    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            # Identify and collect public and private result files
            if 'public' in file_name and file_name.endswith('.xlsx'):
                public_result_files.append(file_path)
            elif 'private' in file_name and file_name.endswith('.xlsx'):
                private_result_files.append(file_path)

# Function to read Excel files and perform ensemble
def process_ensemble(file_paths):
    # Read all Excel files and concatenate them
    data_frames = [pd.read_excel(file) for file in file_paths]
    
    # Combine all data into one DataFrame
    combined_data = pd.concat(data_frames, ignore_index=True)
    
    # Convert the 'answer' column to integer and filter for valid answers (1, 2, 3, or 4)
    combined_data['answer'] = pd.to_numeric(combined_data['answer'], errors='coerce')
    combined_data['answer'] = combined_data['answer'].astype('Int64')
    filtered_data = combined_data[combined_data['answer'].isin([1, 2, 3, 4])]
    
    # Group by 'file_name' and find the most common 'answer' for each
    most_common_answers = filtered_data.groupby('file_name')['answer'].agg(lambda x: x.mode()[0] if not x.mode().empty else None).reset_index()
    
    return most_common_answers

# Process public and private results separately
public_result = process_ensemble(public_result_files)
private_result = process_ensemble(private_result_files)

# Save the results to CSV files
public_result.to_csv('public.csv', index=False)
private_result.to_csv('private.csv', index=False)