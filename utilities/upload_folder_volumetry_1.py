import os
import os.path
import time
import datetime
import zipfile
import shutil

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), 
                       os.path.relpath(os.path.join(root, file), 
                                       os.path.join(path, '..')))

def GoogleDrive_Upload(database,filename_log):

    gauth = GoogleAuth()
    drive = GoogleDrive (gauth)

    gfile = drive.CreateFile({'title':database.PatientName ,'parents': [{'id': '1qnEA7h7t1jX0DzbEUgYWvN0S_I44EYZV'}],'mimeType': 'application/vnd.google-apps.folder'})
    gfile.Upload(param={'supportsTeamDrives': True})
    folder_id_parent = gfile['id']

    extensions = ['.png']

    selected_files = []
    for filename in os.listdir(os.path.join(database.path() + '_output')):
        if any(filename.endswith(ext) for ext in extensions):
            print(filename)
            selected_files.append(os.path.join(database.path() + '_output', filename))

    for upload_file in selected_files:
        print(os.path.basename(upload_file))

        if os.path.basename(upload_file).endswith(".png"):
            gfile = drive.CreateFile({'title':os.path.basename(upload_file) ,'parents': [{'id': folder_id_parent}]})
            gfile.SetContentFile(upload_file)
            gfile.Upload(param={'supportsTeamDrives': True})

    pathSegmentation = os.path.join(database.path() + '_output',database.PatientName)

    with zipfile.ZipFile(pathSegmentation + '.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipdir(pathSegmentation, zipf)

    upload_file_list = [pathSegmentation + '.zip', filename_log]

    for upload_file in upload_file_list:
        try:
            gfile = drive.CreateFile({'title':os.path.basename(upload_file) ,'parents': [{'id': folder_id_parent}]})
            gfile.SetContentFile(upload_file)
            gfile.Upload(param={'supportsTeamDrives': True}) # Upload the file.
        except:
            continue

    os.remove(pathSegmentation + '.zip')


def main(database,filename_log):

    try:
        file = open(filename_log, 'a')
        file.write("\n"+str(datetime.datetime.now())[0:19] + ": Uploading to Google Drive has started")
        file.close()

        GoogleDrive_Upload(database,filename_log)

    except Exception as e: 
        file = open(filename_log, 'a')
        file.write("\n"+str(datetime.datetime.now())[0:19] + ": Uploading to Google Drive was NOT completed ; error: "+str(e)) 
        file.close()


