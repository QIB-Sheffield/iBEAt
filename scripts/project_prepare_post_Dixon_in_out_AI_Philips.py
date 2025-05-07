import os
import datetime
import dbdicom as db

from utilities import xnat
from scripts import steps_core, steps_internal


def single_subject(username, password, path, dataset,subject_ID=None):
    
    #Import data from XNAT
    if isinstance(dataset,str) and '_' in dataset:
        ExperimentName = xnat.main(username, password, path, SpecificDataset=dataset)
        pathScan = os.path.join(path, ExperimentName)
    elif len(dataset)==3:
        ExperimentName = xnat.main(username, password, path, dataset)
        pathScan = os.path.join(path, ExperimentName)
    elif dataset == 'load':
        pathScan = os.path.join(path)
    
    filename_log = pathScan +"_"+ datetime.datetime.now().strftime('%Y%m%d_%H%M_') + "MDRauto_LogFile.txt" #TODO FIND ANOTHER WAY TO GET A PATH

    #Available CPU cores
    try: 
        UsedCores = int(len(os.sched_getaffinity(0)))
    except: 
        UsedCores = int(os.cpu_count())

    database = db.database(path=pathScan)
    database.set_log(filename_log)
    database.log("Analysis of " + pathScan.split('//')[-1] + " has started!")
    database.log("CPU cores: " + str(UsedCores))
    
    # HARMONIZATION

    steps_core.rename_all_series(database)
    steps_core.harmonize_dce(database)
    steps_core.harmonize_subject_name(database)

    # MAPPING

    steps_internal.map_post_contrast_water_dominant(database)

    # ALIGNMENT
    subject_ID ='000'
    steps_internal.export_project_post_contrast_in_out_Dixon_to_AI(database,subject_ID)
    #steps_internal.export_project_post_Dixon_whole_kidney_only_segmentations_as_png(database)
