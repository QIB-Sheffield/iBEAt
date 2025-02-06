import os
import datetime
import dbdicom as db

from utilities import xnat, upload_project_t2s
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
    
    filename_log = pathScan +"_"+ datetime.datetime.now().strftime('%Y%m%d_%H%M_') + "LogFile.txt"

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
    steps_core.harmonize_t2(database)
    #steps_core.harmonize_dce(database)
    steps_core.harmonize_subject_name(database,dataset)
    
    # SEGMENTATION

    steps_core.fetch_dl_models(database)
    steps_internal.fetch_kidney_masks(database)
    steps_core.segment_kidneys(database)
    #steps_core.segment_aorta_on_dce(database)

    # MOTION CORRECTION

    steps_core.mdreg_t2star(database)
    #steps_core.mdreg_dce(database)
    steps_core.export_mdreg(database)
    

    # MAPPING

    steps_core.map_T2star(database)
    #steps_core.map_DCE(database)
    steps_core.export_mapping(database)

    # ALIGNMENT

    steps_core.align_T2star(database)
    #steps_core.align_DCE(database)
    steps_internal.export_alignment_t2s_project(database)

    # MEASUREMENT

    #steps_core.fill_gaps(database)
    #steps_core.cortex_medulla(database)

    steps_core.measure_t2star_maps(database)

    # ROI analysis

    steps_core.roi_fit_T2star(database)

    steps_internal.export_whole_kidney_only_segmentations_as_png(database)

    steps_internal.export_dicom_t2s_project(database)

    #upload images, logfile and csv to google drive
    filename_csv = os.path.join(database.path() + '_output',database.PatientName[0] + '_biomarkers.csv')
    upload_project_t2s.main(pathScan, filename_log, filename_csv)
