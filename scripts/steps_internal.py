import time
from pipelines import (
    fetch_Drive_mask,
    mdr, 
    mapping, 
    harmonize,
    segment,
    export, 
)


## HARMONIZATION

def harmonize_t1_t2(database):
    start_time = time.time()
    database.log("Harmonizing T1 and T2 series (merged) has started!")
    try:
        harmonize.t1_t2_merge(database)
        database.log("Harmonizing T1 and T2 series (merged) was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e:
        database.log("Harmonizing T1 and T2 series (merged) was NOT completed; error: " + str(e)) 
        database.restore()

def harmonize_subject_name(database,dataset):
    start_time = time.time()
    database.log("Harmonizing subject name has started!")
    try:
        harmonize.subject_name_internal(database,dataset)
        database.log("Harmonizing subject name was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e:
        database.log("Harmonizing subject name was NOT completed; error: " + str(e)) 
        database.restore()

## SEGMENTATION

def fetch_kidney_masks(database):
    start_time = time.time()
    database.log("Fetching kidney masks has started")
    try:
        fetch_Drive_mask.kidney_masks(database)
        database.log("Fetching kidney masks was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e:
        database.log("Fetching kidney masks was NOT completed; error: "+str(e))
        database.restore()

def compute_whole_kidney_canvas(database):
    start_time = time.time()
    database.log('Starting computing canvas')
    try:
        segment.compute_whole_kidney_canvas(database)
        database.log("Computing canvas was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e: 
        database.log("Computing canvas was NOT computed; error: "+str(e))
        database.restore()

def export_sinus_fat_segmentations(database):
    start_time = time.time()
    database.log("Export kidney segmentations has started")
    try:
        export.kidney_masks_as_png(database)
    except Exception as e:
        database.log("Export kidney segmentations was NOT completed; error: "+str(e))
    
    try:
        export.kidney_masks_sinus_fat_dixon_as_dicom(database)
    except Exception as e:
        database.log("Export kidney masks sinus fat masks and dixon was NOT completed; error: "+str(e))

    database.log("Export kidney segmentations was completed --- %s seconds ---" % (int(time.time() - start_time)))

def export_whole_kidney_only_segmentations_as_png(database):
    start_time = time.time()
    database.log("Export kidney segmentations has started")

    try:
        export.kidney_masks_as_png(database,backgroud_series = 'Dixon_post_contrast_out_phase',RK_mask = 'RK', LK_mask = 'LK',mask_name = 'Whole Kidney_masks')
    except Exception as e:
        database.log("Export kidney segmentations was NOT completed; error: "+str(e))

    database.log("Export kidney segmentations was completed --- %s seconds ---" % (int(time.time() - start_time)))

def export_project_pre_Dixon_whole_kidney_only_segmentations_as_png(database):
    start_time = time.time()
    database.log("Export kidney segmentations has started")

    try:
        export.kidney_masks_as_png(database,backgroud_series = 'Dixon_out_phase [coreg]',RK_mask = 'RK', LK_mask = 'LK',mask_name = 'Whole Kidney_masks')
    except Exception as e:
        database.log("Export kidney segmentations was NOT completed; error: "+str(e))

    database.log("Export kidney segmentations was completed --- %s seconds ---" % (int(time.time() - start_time)))

def export_project_post_Dixon_whole_kidney_only_segmentations_as_png(database):
    start_time = time.time()
    database.log("Export kidney segmentations has started")

    try:
        export.kidney_masks_as_png(database,backgroud_series = 'Dixon_post_contrast_out_phase',RK_mask = 'RK', LK_mask = 'LK',mask_name = 'Whole Kidney_masks')
    except Exception as e:
        database.log("Export kidney segmentations was NOT completed; error: "+str(e))

    database.log("Export kidney segmentations was completed --- %s seconds ---" % (int(time.time() - start_time)))

def export_segmentations_folder_volumetry_1(database):
    start_time = time.time()
    database.log("Export kidney segmentations has started")
    try:
        export.kidney_masks_as_dicom_folder_1(database)
        export.kidney_masks_as_png_folder_1(database)
        database.log("Export kidney segmentations was completed --- %s seconds ---" % (int(time.time() - start_time)))
    except Exception as e:
        database.log("Export kidney segmentations was NOT completed; error: "+str(e))

def fill_DCE_masks(database):
    start_time = time.time()
    database.log("Filling DCE masks has started")
    try:
        segment.fill_DCE_cor_med_masks(database)
        database.log("Filling DCE masks was completed --- %s seconds ---" % (int(time.time() - start_time)))
    except Exception as e:
        database.log("Filling DCE masks was NOT completed; error: "+str(e))

## MODEL-DRIVEN MOTION CORRECTION

def mdreg_t1_t2(database):
    start_time = time.time()
    database.log("Model-driven registration for T1 T2 has started")
    try:
        mdr.T1_T2(database)
        database.log("Model-driven registration for T1 T2 was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e:
        database.log("Model-driven registration for T1 T2 was NOT completed; error: "+str(e))
        database.restore()
    
## MAPPING

def map_post_contrast_water_dominant(database):
    start_time = time.time()
    print('Starting fat dominant mapping')
    database.log("Fat dominant mapping has started")
    try:
        mapping.Dixon_post_contrast_water_dominant(database)
        database.log("Fat dominant was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e: 
        database.log("Fat dominant was NOT completed; error: "+str(e))
        database.restore()

def map_fat_dominant(database):
    start_time = time.time()
    print('Starting fat dominant mapping')
    database.log("Fat dominant mapping has started")
    try:
        mapping.Dixon_fat_dominant(database)
        database.log("Fat dominant was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e: 
        database.log("Fat dominant was NOT completed; error: "+str(e))
        database.restore()

def map_T1_from_T1_T2_mdr(database):
    start_time = time.time()
    print('Starting T1 mapping from T1_T2 MDR')
    database.log("T1 mapping from T1_T2 MDR has started")
    try:
        mapping.T1_from_T1_T2_mdr(database)
        database.log("T1 mapping from T1_T2 MDR was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e: 
        database.log("T1 mapping from T1_T2 MDR was NOT completed; error: "+str(e))
        database.restore()

def map_T2_from_T1_T2_mdr(database):
    start_time = time.time()
    print('Starting T2 mapping from T1_T2 MDR')
    database.log("T2 mapping from T1_T2 MDR has started")
    try:
        mapping.T2_from_T1_T2_mdr(database)
        database.log("T2 mapping from T1_T2 MDR was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e: 
        database.log("T2 mapping from T1_T2 MDR was NOT completed; error: "+str(e))
        database.restore()


## ALIGNMENT

def export_alignment_t2s_project(database):
    start_time = time.time()
    print('Exporting alignment')
    database.log("Export alignment has started")
    try:
        export.alignment_t2s_project(database)
        database.log("Export alignment was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e: 
        database.log("Export alignment was NOT completed; error: "+str(e))
        database.restore()

## MEASURE
    
## ROI ANALYSIS

## DATA EXPORT

def export_dicom_t2s_project(database):
    start_time = time.time()
    print('Exporting alignment')
    database.log("Export alignment has started")
    try:
        export.project_t2s_prepare_dicom(database)
        database.log("Export alignment was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e: 
        database.log("Export alignment was NOT completed; error: "+str(e))
        database.restore()

def export_project_pre_Dixon_to_AI(database,subject_ID):
    start_time = time.time()
    database.log("Export to AI has started")
    try:
        export.pre_Dixon_to_AI(database,subject_ID)
        database.log("Export to AI was completed --- %s seconds ---" % (int(time.time() - start_time)))
    except Exception as e:
        database.log("Export to AI was NOT completed; error: "+str(e))

def export_project_pre_Dixon_in_out_to_AI(database,subject_ID):
    start_time = time.time()
    database.log("Export to AI has started")
    try:
        export.pre_Dixon_in_out_to_AI(database,subject_ID)
        database.log("Export to AI was completed --- %s seconds ---" % (int(time.time() - start_time)))
    except Exception as e:
        database.log("Export to AI was NOT completed; error: "+str(e))

def export_project_post_contrast_Dixon_to_AI(database,subject_ID):
    start_time = time.time()
    database.log("Export to AI has started")
    try:
        export.post_contrast_Dixon_to_AI(database,subject_ID)
        database.log("Export to AI was completed --- %s seconds ---" % (int(time.time() - start_time)))
    except Exception as e:
        database.log("Export to AI was NOT completed; error: "+str(e))

def export_project_post_contrast_in_out_Dixon_to_AI(database,subject_ID):
    start_time = time.time()
    database.log("Export to AI has started")
    try:
        export.post_contrast_in_out_Dixon_to_AI(database,subject_ID)
        database.log("Export to AI was completed --- %s seconds ---" % (int(time.time() - start_time)))
    except Exception as e:
        database.log("Export to AI was NOT completed; error: "+str(e))


def export_DCE_AI(database,subject_ID):
    start_time = time.time()
    database.log("Export DCE AI has started")
    try:
        export.DCE_AI(database,subject_ID)
        database.log("Export DCE AI was completed --- %s seconds ---" % (int(time.time() - start_time)))
    except Exception as e:
        database.log("Export DCE AI was NOT completed; error: "+str(e))


def export_DCE_AI_segmentations(database):
    start_time = time.time()
    database.log("Export kidney segmentations has started")

    try:
        export.kidney_masks_as_png(database,mask_name = 'Kidney_Masks')
    except Exception as e:
        database.log("Export kidney segmentations was NOT completed; error: "+str(e))
    try:
        export.aif_as_png(database)
    except Exception as e:
        database.log("Export kidney segmentations was NOT completed; error: "+str(e))

    try:
        export.kidney_masks_as_png(database,backgroud_series = 'Dixon_out_phase [coreg]',RK_mask = 'RKM', LK_mask = 'LKM',mask_name = 'Medulla_masks')
    except Exception as e:
        database.log("Export kidney segmentations was NOT completed; error: "+str(e))

    try:
        export.kidney_masks_as_png(database,backgroud_series = 'Dixon_out_phase [coreg]',RK_mask = 'RKC', LK_mask = 'LKC', mask_name = 'Cortex_masks')
    except Exception as e:
        database.log("Export kidney segmentations was NOT completed; error: "+str(e))

    try:
        export.kidney_masks_as_png(database,backgroud_series = 'T1w_magnitude_LK_align_fill',RK_mask = 'RKC', LK_mask = 'LKM', mask_name = 'T1w_LK_medulla_masks')
    except Exception as e:
        database.log("Export kidney segmentations was NOT completed; error: "+str(e))

    try:
        export.kidney_masks_as_png(database,backgroud_series = 'T1w_magnitude_RK_align_fill',RK_mask = 'RKM', LK_mask = 'LKC', mask_name = 'T1w_RK_medulla_masks')
    except Exception as e:
        database.log("Export kidney segmentations was NOT completed; error: "+str(e))


    
    