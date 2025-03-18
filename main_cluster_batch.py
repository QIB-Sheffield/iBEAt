""" 
@author: Joao Periquito 
iBEAt Analysis MAIN CLUSTER Scrpit
2022
Download XNAT dataset -> Name Standardization ->    Execute MDR   -> Custom Moddeling (T1, T2...) ->     Biomarker extraction   -> Google Drive Upload
  pipelines.xnat.py   -> pipelines.rename.py  -> pipelines.mdr.py ->      pipelines.mapping.py    -> pipelines.export_ROI_stats -> scripts.upload.py

TO RUN THE SCRIPT YOU USE: python main_cluster.py --num n (WHERE n is an integer with the value of the XNAT dataset)
"""
import argparse
from scripts.Folder_Volumetry_2_dcmFinalMasks_Dixons import single_subject
import utilities.XNAT_credentials as XNAT_cred
import os

if __name__ == '__main__':

    #XNAT Credentials
    username, password = XNAT_cred.main()
    
    path = "//mnt//fastdata//" + username + "//FODLER_2" #CLUSTER PATH TO SAVE DATA, ADD YOUR LOCAL PATH IF YOU WANT TO RUN IT LOCALLY
    if not os.path.exists(path):
        os.mkdir(path)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num',
                        dest='num',
                        help='Define the XNAT dataset',
                        type=int)

    args = parser.parse_args()

    #SELECT YOUR DATASET
    #dataset = [site, study, dataset]
    dataset = [2,1,args.num]

    ################################################# EXAMPLE DATASET SELECTION #############################################################
    #DATASET CODE FOR LEEDS
    #  (FIRST NUMBER)                 (SECOND NUMBER)                                 (THIRD NUMBER - INNPUT from --num when you run the main script: python main_cluster.py --num n)
    #  2: BEAt-DKD-WP4-Bordeaux    (selected) BEAt-DKD-WP4-Leeds                 (selected) BEAt-DKD-WP4-Leeds -> (selected) Leeds_Patients
    #  3: BEAt-DKD-WP4-Exeter       ->0: Leeds_Patients                            0: Leeds_Patient_4128001
    #  4: BEAt-DKD-WP4-Turku          1: Leeds_volunteer_repeatability_study       1: Leeds_Patient_4128002
    #  5: BEAt-DKD-WP4-Bari           2: Leeds_Phantom_scans                       2: Leeds_Patient_4128004
    #->6: BEAt-DKD-WP4-Leeds          3: Leeds_RAVE_Reconstructions                       ..........
    #  7: BEAt-DKD-WP4-Sheffield      4: Leeds_setup_scans                      ->14: Leeds_Patient_4128015
    #########################################################################################################################################


    #offset = 0                  #Bordeaux   Baseline 2,1    =  49
    #offset = 49                 #Exeter     Baseline 3,0    = 111
    #offset = 49+111             #Exeter     Followup 3,3    =  44
    #offset = 49+111+44          #Leeds      Baseline 6,0    =  62
    #offset = 49+111+44+62        #Bordeaux   Followup 2,0    =  26



    single_subject(username, password, path, dataset)

