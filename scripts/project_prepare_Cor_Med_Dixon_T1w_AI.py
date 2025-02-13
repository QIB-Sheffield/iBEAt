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

    steps_core.rename_all_series(database)
    steps_core.harmonize_dce(database)
    steps_core.harmonize_subject_name(database)

    steps_core.fetch_dl_models(database)
    steps_internal.fetch_kidney_masks(database)
    steps_core.segment_kidneys(database)
    steps_core.segment_aorta_on_dce(database)

    steps_core.mdreg_dce(database)

    steps_core.map_DCE(database)
    steps_core.export_mapping(database)

    steps_core.align_dixon(database)
    steps_core.align_DCE(database)
    steps_core.align_T1w(database)
    steps_core.export_alignment(database)

    steps_core.fill_gaps(database)
    steps_core.cortex_medulla(database)
