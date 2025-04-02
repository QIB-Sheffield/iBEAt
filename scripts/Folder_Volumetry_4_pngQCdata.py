import os
import pandas as pd
import matplotlib.pyplot as plt
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import tempfile
from datetime import datetime

# Step 1: Authenticate using PyDrive
gauth = GoogleAuth()
gauth.LocalWebserverAuth()  # This creates a local webserver for OAuth authentication
drive = GoogleDrive(gauth)

upload_folder_id = '1dlYVT3vP10rKvx_yiCvf6sfKvq3XTIFS'

# Step 2: List all CSV files in the Google Drive folder
folder_id = '1xhT6DlThcXqxY4_d0dbZ1EV7kDk8Xfh-'  # Replace with your folder ID
file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed = false and mimeType='application/vnd.ms-excel'"}).GetList()

# Step 3: Create a figure for the scatter plot
plt.figure(figsize=(10, 6))
plt.xlabel('Volume (Value)')
plt.ylabel('Compactness (Value)')
plt.title('Volume vs Compactness')

# Step 4: Loop through each CSV file, download, and process the data
for file in file_list:
    file_name = file['title']
    file_id = file['id']
    
    # Step 4a: Download the file to a temporary folder
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_file_path = os.path.join(tmpdir, file_name)
        
        # Download the file content
        downloaded = file.GetContentString()
        
        # Save the CSV content to a temporary file
        with open(temp_file_path, 'w') as f:
            f.write(downloaded)
        
        # Step 4b: Read the CSV content into a DataFrame
        df = pd.read_csv(temp_file_path)
        
        # Filter the DataFrame for 'Volume' and 'Compactness' parameters
        df_filtered = df[df['Parameter'].isin(['Volume', 'Compactness'])]
        
        # Pivot the data to have 'SeriesDescription' as index and parameters as columns
        pivot_df = df_filtered.pivot_table(index='SeriesDescription', columns='Parameter', values='Value')
        
        # Step 4c: Add points to the scatter plot using 'Value' for Volume and Compactness
        plt.scatter(pivot_df['Volume'], pivot_df['Compactness'], alpha=0.5)

        # Annotate each point with "PatientID"
        for idx, row in pivot_df.iterrows():
            patient_id = file_name[:-4]  # Get PatientID
            plt.text(row['Volume'], row['Compactness'], patient_id, fontsize=1, color='black', alpha=0.7)  # Add annotation

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
file_count = len(file_list)  # Total number of CSV files
scatter_plot_path = f'{current_time}_QC_volumetry_{file_count}_files.png'

plt.savefig(scatter_plot_path, dpi=600)
plt.close()  # Close the plot after saving it

# Step 6: Upload the final plot to Google Drive
file_to_upload = drive.CreateFile({'title': scatter_plot_path, 'parents': [{'id': upload_folder_id}]})
file_to_upload.SetContentFile(scatter_plot_path)
file_to_upload.Upload()

#os.remove(scatter_plot_path)

print("Final scatter plot uploaded to Google Drive.")