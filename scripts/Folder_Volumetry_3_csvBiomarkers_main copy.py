import os
import sys
from datetime import datetime
import shutil
import tempfile
import zipfile
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import pandas as pd
import dbdicom as db


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utilities.folder_3_harmonize import standardize_name
from scripts import steps_internal

def unzip_file(zip_path, extract_to):
    """Unzips a file to a specified folder."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Unzipped: {zip_path} to {extract_to}")
    except zipfile.BadZipFile:
        print(f"Error: {zip_path} is not a valid ZIP file!")

def find_and_rename_csv(folder_path, new_name):
    """Finds the first CSV file in a folder and renames it."""
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):  # Check if it's a CSV file
            old_path = os.path.join(folder_path, file)
            new_path = os.path.join(folder_path, new_name)

            os.rename(old_path, new_path)  # Rename the file
            print(f"Renamed '{file}' to '{new_name}' in {folder_path}")
            return  # Exit after renaming the first CSV file

    print("No CSV file found in the folder.")

# Authenticate and create PyDrive client
gauth = GoogleAuth()  # Opens a web browser for authentication
drive = GoogleDrive(gauth)

# Define the folder IDs for the two folders
folder_id = '1iVU4aOqZPjYY1chcHKwDkVCT5_QTNojL'  # Folder for Need_editing == 0
folder_id_editing = '1QxUPnd61INezsGj0Xlnb8c24bjxDcR1A'  # Folder for Need_editing == 1
folder_id_upload = '1xhT6DlThcXqxY4_d0dbZ1EV7kDk8Xfh-'
folder_id_main = '19AcX0V1R0ozCif_9Oy9dNZPdF2c6peUZ'

timestamp = datetime.now().strftime("%Y%m%d_%H%M")

# Create a temporary folder in the current working directory
temp_folder = tempfile.mkdtemp(dir=os.getcwd())  # Create temp folder in current directory
print(f"Temporary folder created at: {temp_folder}")

# Load the spreadsheet into a Pandas DataFrame
file_name = "20250217_1014_XNAT_full"  # Name of the Google Sheets file
file_list = drive.ListFile({'q': f"title = '{file_name}' and mimeType = 'application/vnd.google-apps.spreadsheet'"}).GetList()

if file_list:
    file_id = file_list[0]['id']  # Get the file ID
    file = drive.CreateFile({'id': file_id})
    
    # Export as Excel (.xlsx)
    export_url = f"https://docs.google.com/feeds/download/spreadsheets/Export?key={file_id}&exportFormat=xlsx"
    file.GetContentFile(os.path.join(temp_folder, "downloaded_file.xlsx"), mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
    print(f"Downloaded: {file_name} as Excel into {temp_folder}")

    # Load the DataFrame
    df = pd.read_excel(os.path.join(temp_folder, "downloaded_file.xlsx"))

    # Add a new "CSV" column with default value 0
    df["CSV"] = 0 

    # Filter and print rows where 'Need_editing' == 1
    if "Need_editing" in df.columns:
        #Handle Need_editing == 1 (from 'folder_id_editing')
        df_filtered_editing = df[df["Need_editing"] == 1]
        if not df_filtered_editing.empty:
            print("Filtered DataFrame (Need_editing == 1):")
            print(df_filtered_editing)

            # Now handle cases where 'Need_editing' == 1, search in the 'editing' folder
            for index, row in df_filtered_editing.iterrows():
                patient_id = row["PatientID"]


                if 'followup' in patient_id:
                    baseline = 0
                else:
                    baseline = 1
                patient_id = standardize_name(patient_id)
                patient_id_zip = patient_id + "_E.zip"

                query_check = f"'{folder_id_upload}' in parents and trashed = false and title = '{patient_id + '.csv'}' and trashed = false"
                file_list_check = drive.ListFile({'q': query_check}).GetList()

                if len(file_list_check) > 0:
                    file_list_check = []
                    query_check = []
                    df.at[index, "CSV"] = 1
                    continue
                else:

                    print(f"Looking for file for PatientID (Need_editing == 1): {patient_id}")

                    # Search for the corresponding file in the Google Drive 'editing' folder
                    file_list = drive.ListFile({'q': f"'{folder_id_editing}' in parents and trashed = false and title contains '{patient_id_zip}'"}).GetList()

                    if file_list:
                        for file in file_list:
                            file_id = file['id']
                            file_title = file['title']

                            if 'followup' in file_title:
                                file_baseline = 0
                            else:
                                file_baseline = 1
                            
                            if baseline != file_baseline:
                                continue


                            print(f"Downloading file: {file_title} for PatientID: {patient_id} from editing folder")

                            # Download the file to the temp folder
                            file.GetContentFile(os.path.join(temp_folder, f"{file_title}"))
                            print(f"Downloaded: {patient_id}_{file_title} to {temp_folder}")

                            unzip_file(os.path.join(temp_folder,file_title), temp_folder)

                            database = db.database(path=os.path.join(temp_folder,file_title[:-4]))
                            steps_internal.measure_kidney_volumetrics_paper_volumetry(database)

                            find_and_rename_csv(os.path.join(database.path() + '_output'), patient_id+'.csv')

                            filename_csv = os.path.join(database.path() + '_output',patient_id + '.csv')
                            
                            #Export results csv
                            gfile = drive.CreateFile({'title':os.path.basename(filename_csv) ,'parents': [{'id': folder_id_upload}]})
                            gfile.SetContentFile(filename_csv)
                            gfile.Upload(param={'supportsTeamDrives': True})

                            df.at[index, "CSV"] = 1
                            
                            path_to_delete = database.path()
                            database.close()
                            shutil.rmtree(path_to_delete)
                            os.remove(path_to_delete + '.zip')

                            file_list_check = []
                            query_check = []

                    else:
                        print(f"No file found for PatientID: {patient_id} in editing folder")

                file_list_check = []
                query_check = []

        df_filtered_editing = df[df["Need_editing"] == 0]
        if not df_filtered_editing.empty:
            print("Filtered DataFrame (Need_editing == 1):")
            print(df_filtered_editing)

            # Now handle cases where 'Need_editing' == 1, search in the 'editing' folder
            for index, row in df_filtered_editing.iterrows():
                patient_id = row["PatientID"]
                if 'followup' in patient_id:
                    baseline = 0
                else:
                    baseline = 1
                patient_id = standardize_name(patient_id)
                patient_id_zip = patient_id + '.zip'

                query_check = f"'{folder_id_upload}' in parents and title = '{patient_id + '.csv'}' and trashed = false"
                file_list_check = drive.ListFile({'q': query_check}).GetList()

                if len(file_list_check) > 0:
                    file_list_check = []
                    query_check = []
                    df.at[index, "CSV"] = 1
                    continue
                else:

                    print(f"Looking for file for PatientID (Need_editing == 0): {patient_id}")

                    # Search for the corresponding file in the Google Drive 'editing' folder
                    folder_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed = false and mimeType='application/vnd.google-apps.folder' and title contains '{patient_id}'"}).GetList()
                    
                    for folder in folder_list:
                        if 'followup' in folder['title']:
                            file_baseline = 0
                        else:
                            file_baseline = 1
                        
                        if baseline != file_baseline:
                            continue
                        else:
                            folder_id_dataset = folder['id']

                    file_list = drive.ListFile({'q': f"'{folder_id_dataset}' in parents and trashed = false and  title contains '{patient_id_zip}'"}).GetList()

                    if file_list:
                        for file in file_list:
                            file_id = file['id']
                            file_title = file['title']

                            print(f"Downloading file: {file_title} for PatientID: {patient_id} from autogenerated folder")

                            # Download the file to the temp folder
                            file.GetContentFile(os.path.join(temp_folder, f"{file_title}"))
                            print(f"Downloaded: {patient_id}_{file_title} to {temp_folder}")

                            unzip_file(os.path.join(temp_folder,file_title), temp_folder)

                            database = db.database(path=os.path.join(temp_folder,file_title[:-4]))
                            steps_internal.measure_kidney_volumetrics_paper_volumetry(database)
                            

                            find_and_rename_csv(os.path.join(database.path() + '_output'), patient_id+'.csv')

                            filename_csv = os.path.join(database.path() + '_output',patient_id + '.csv')
                            
                                #Export results csv
                            gfile = drive.CreateFile({'title':os.path.basename(filename_csv) ,'parents': [{'id': folder_id_upload}]})
                            gfile.SetContentFile(filename_csv)
                            gfile.Upload(param={'supportsTeamDrives': True})

                            df.at[index, "CSV"] = 1

                            path_to_delete = database.path()
                            database.close()
                            shutil.rmtree(path_to_delete)
                            os.remove(path_to_delete + '.zip')

                            file_list_check = []
                            query_check = []

                    else:
                        print(f"No file found for PatientID: {patient_id} in editing folder")
                

df_with_CSV = len(df[df["CSV"] == 1])
total_patients = df["PatientID"].nunique()


# Define the filename
filename_csv_overview = os.path.join(temp_folder, f"{timestamp}_volumetry_overview_" + str(df_with_CSV) +"_out_of_" + str(total_patients) + ".csv")

# Save DataFrame as CSV
df.to_csv(filename_csv_overview, index=False)

gfile = drive.CreateFile({'title':os.path.basename(filename_csv_overview) ,'parents': [{'id': folder_id_main}]})
gfile.SetContentFile(filename_csv_overview)
gfile.Upload(param={'supportsTeamDrives': True})


