import os
import datetime
import dbdicom as db

from utilities import xnat, upload_folder_volumetry_1
from scripts import steps_core
from scripts import steps_internal


def single_subject(username, password, path, dataset):
    
    #Import data from XNAT
    if isinstance(dataset,str) and '_' in dataset:
        ExperimentName = xnat.main(username, password, path, SpecificDataset=dataset)
        pathScan = os.path.join(path, ExperimentName)
    elif len(dataset)==3:
        ExperimentName = xnat.main(username, password, path, dataset)
        pathScan = os.path.join(path, ExperimentName)
    elif dataset == 'load':
        pathScan = os.path.join(path)
        dataset = [3,3,0]
    
    

    #Available CPU cores
    try: 
        UsedCores = int(len(os.sched_getaffinity(0)))
    except: 
        UsedCores = int(os.cpu_count())

    database = db.database(path=pathScan)
    

    steps_internal.harmonize_subject_name(database,dataset)
    
    filename_log = os.path.join(database.path(), database.PatientName +"_"+ datetime.datetime.now().strftime('%Y%m%d_%H%M_') + "LogFile.txt")
    database.set_log(filename_log)
    
    # HARMONIZATION

    steps_core.rename_all_series(database)

    # # SEGMENTATION
    
    steps_internal.compute_whole_kidney_canvas(database)
    steps_core.fetch_nnunet_models(database)
    steps_core.segment_kidneys_nnunet(database)
    steps_internal.export_segmentations_folder_volumetry_1(database)

    upload_folder_volumetry_1.main(database, filename_log)

