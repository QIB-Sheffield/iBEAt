""" 
@author: Joao Periquito 
iBEAt XNAT import
2022  
XNAT Dataset auto-import script 
"""

import xnat
import zipfile
import datetime
import os

def XNAT_download(username,password,path,DatasetSelected,SpecificDataset=None):
    url = "https://qib.shef.ac.uk"

    if DatasetSelected[0] == 7:
        DatasetSelected[1] = DatasetSelected[2]

    if SpecificDataset!=None:
        
        site = SpecificDataset.split('_')[0]
        patient = SpecificDataset.split('_')[1]
        
        if   site == '1128': #Bari
            DatasetSelected[0] = 5
            DatasetSelected[1] = 0
        elif site == '2128': #Bordeaux
            DatasetSelected[0] = 2
            DatasetSelected[1] = 1 #Baseline
        elif site == '3128': #Exeter
            DatasetSelected[0] = 3
            DatasetSelected[1] = 0
        elif site == '4128': #Leeds
            DatasetSelected[0] = 6
            DatasetSelected[1] = 0
        elif site == '5128': #Turku
            DatasetSelected[0] = 4
            DatasetSelected[1] = 3 #GE
        elif site == '2178': #Sheffield
            DatasetSelected[0] = 7
            DatasetSelected[1] = DatasetSelected[2]


    with xnat.connect(url, user=username, password=password) as session:
        xnatProjects = [project.secondary_id for project in session.projects.values()]
        #for x in range(len(xnatProjects)):
            #print (str(x) +": " + xnatProjects[x])
        #print("Select the project:")
        #projectSelected = int(input())
        #projectSelected = 6
        projectSelected = DatasetSelected[0]
        projectID = xnatProjects[projectSelected]
        print(projectID)
        
        projectName = [project.name for project in session.projects.values() if project.secondary_id == projectID][0]
        if projectName:
            xnatSubjects = [subject.label for subject in session.projects[projectName].subjects.values()]
            for x_2 in range(len(xnatSubjects)):
                print (str(x_2) +": " + xnatSubjects[x_2])
            #print("Select the project:")
            #xnatSubjectsSelected = int(input())
            #xnatSubjectsSelected = 0
            xnatSubjectsSelected = DatasetSelected[1]
            #print(xnatSubjects[xnatSubjectsSelected])
            subjectName = xnatSubjects[xnatSubjectsSelected]
            dataset = session.projects[projectName]

            if DatasetSelected[0] == 7:
                dataset = session.projects[projectName].subjects[subjectName]
                dataset.download_dir(path)
                print(str(xnatSubjects[xnatSubjectsSelected]))
                return str(xnatSubjects[xnatSubjectsSelected])


            xnatExperiments = [experiment.label for experiment in session.projects[projectName].subjects[subjectName].experiments.values()]
            if SpecificDataset!=None:
                for x_3 in range(len(xnatExperiments)):
                #print(str(x_3) +": " + xnatExperiments[x_3])
                    if patient in xnatExperiments[x_3]:
                        DatasetSelected[2] = x_3
                        break
                
            #print("Selected the project:")
            #xnatExperimentsSelected = int(input())
            #xnatExperimentsSelected = 14
            xnatExperimentsSelected = DatasetSelected[2]
            print("Selected project: " + str(xnatExperiments[xnatExperimentsSelected]))	
            experimentName = xnatExperiments[xnatExperimentsSelected]
            dataset = session.projects[projectName].subjects[subjectName].experiments[experimentName]
            dataset.download_dir(path)
            return experimentName

def zipFiles(listPaths):
    dt = datetime.datetime.now()
    zip_file = zipfile.ZipFile(dt.strftime('%Y%m%d') + '_xnat_upload.zip', 'w')
    for file in listPaths:
        zip_file.write(file, compress_type=zipfile.ZIP_DEFLATED)
    zip_file.close()
    zip_path = os.path.realpath(zip_file.filename)
    return zip_path



#####################################################
################ UPLOAD DOES NOT WORK ###############
def XNAT_upload(username,password,path):
    url = "https://qib.shef.ac.uk"

    with xnat.connect(url, user=username, password=password) as session:
        xnatProjects = [project.secondary_id for project in session.projects.values()]
        for x in range(len(xnatProjects)):
            print (str(x) +": " + xnatProjects[x])
        print("Select the project:")
        projectID = xnatProjects[9]
        print(projectID)
        #uploadPaths = [image.file for image in app.folder.instances()]
        uploadPaths = [image.file for image in path]
        uploadZipFile = zipFiles(uploadPaths)

    return
################ UPLOAD DOES NOT WORK ###############
#####################################################


def main(username, password, path,DatasetSelected=[0,0,0],SpecificDataset=None):

    if SpecificDataset==None:
        experimentName = XNAT_download(username,password,path,DatasetSelected)

    else:
        experimentName = XNAT_download(username,password,path,DatasetSelected=[0,0,0],SpecificDataset=SpecificDataset)

    return experimentName

    

