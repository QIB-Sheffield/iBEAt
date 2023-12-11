import os
import datetime
import dbdicom as db

import pipelines.rename as rename
import pipelines.mdr as mdr
import pipelines.mapping as map
import pipelines.export_ROI_stats as export_ROIs
import pipelines.apply_AI_segmentation as AI_segmentation

import scripts.upload as upload
import scripts.QC_rename as check_rename
import scripts.QC_mdr as check_mdr
import scripts.QC_masks as check_masks
import scripts.QC_mapping as check_maps
from scripts import xnat

def single_subject(username, password, path, dataset):
    
    #Import data from XNAT
    ExperimentName = xnat.main(username, password, path, dataset)
    pathScan = path + "//" + ExperimentName
    filename_log = pathScan +"_"+ datetime.datetime.now().strftime('%Y%m%d_%H%M_') + "MDRauto_LogFile.txt" #TODO FIND ANOTHER WAY TO GET A PATH
    
    #Available CPU cores
    try: 
        UsedCores = int(len(os.sched_getaffinity(0)))
    except: 
        UsedCores = int(os.cpu_count())

    folder = db.database(path=pathScan)
    folder.set_log(filename_log)
    folder.log("Analysis of " + pathScan.split('//')[-1] + " has started!")
    folder.log("CPU cores: " + str(UsedCores))
    
    #Name standardization 
    try:
        print("starting renaming")
        rename.main(folder)
        check_rename.main(folder)
    except Exception as e:
        folder.log("Renaming was NOT completed; error: " + str(e))

    #Apply motion correction using MDR
    try:
        print("starting mdr")
        mdr.main(folder)
        check_mdr.main(folder)
    except Exception as e:
        folder.log("Renaming was NOT completed; error: " + str(e))

    #Apply UNETR to segment right/left kidney
    try:
        print("starting kidney segmentation")
        AI_segmentation.main(folder)
        check_masks.main(folder)
    except Exception as e:
        folder.log("Kidney segmentation was NOT completed; error: " + str(e))

    #Custom modelling
    try:
        print("staring mapping")
        map.main(folder)
        check_maps.main(folder)
    except Exception as e:
        folder.log("Modelling was NOT completed; error: " + str(e))

    #Generate masks using unetr, apply alignment, extract biomarkers to a .csv
    try:
        print('starting parameter extraction')
        filename_csv = export_ROIs.main(folder,ExperimentName)
    except Exception as e:
        folder.log("Parameter extraction was NOT completed; error: " + str(e))
    
    #upload images, logfile and csv to google drive
    upload.main(pathScan, filename_log, filename_csv)
