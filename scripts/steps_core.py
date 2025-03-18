import time
from pipelines import (
    fetch_AI_model,
    fetch_nnunet_model,
    fetch_Drive_mask,
    segment, 
    measure, 
    export, 
    rename, 
    mdr, 
    mapping, 
    harmonize, 
    align,
    roi_fit,
)



## HARMONIZATION

def rename_all_series(database):
    start_time = time.time()
    database.log("Renaming has started!")
    try:
        rename.all_series(database)
        rename.check(database)
        database.log("Renaming was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e:
        database.log("Renaming was NOT completed; error: " + str(e))  
        database.restore()
        raise RuntimeError('Critical step failed (renaming) - exiting pipeline.')


def harmonize_pc(database):
    start_time = time.time()
    database.log("Harmonizing PC series has started!")
    try:
        harmonize.pc_left(database)
        database.save()
    except Exception as e:
        database.log("Harmonizing PC left series was NOT completed; error: " + str(e)) 
        database.restore()

    try:
        harmonize.pc_right(database)
        database.save()
    except Exception as e:
        database.log("Harmonizing PC right series was NOT completed; error: " + str(e)) 
        database.restore()
    database.log("Harmonizing PC series was completed --- %s seconds ---" % (int(time.time() - start_time)))

def harmonize_t2(database):
    start_time = time.time()
    database.log("Harmonizing T2 series has started!")
    try:
        harmonize.t2(database)
        database.log("Harmonizing T2 series was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e:
        database.log("Harmonizing T2 series was NOT completed; error: " + str(e)) 
        database.restore()

def harmonize_mt(database):
    start_time = time.time()
    database.log("Harmonizing MT series has started!")
    try:
        harmonize.mt(database)
        database.log("Harmonizing MT series was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e:
        database.log("Harmonizing MT series was NOT completed; error: " + str(e)) 
        database.restore()

def harmonize_dti(database):
    start_time = time.time()
    database.log("Harmonizing DTI series has started!")
    try:
        harmonize.dti(database)
        database.log("Harmonizing DTI series was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e:
        database.log("Harmonizing DTI series was NOT completed; error: " + str(e)) 
        database.restore()

def harmonize_ivim(database):
    start_time = time.time()
    database.log("Harmonizing IVIM series has started!")
    try:
        harmonize.ivim(database)
        database.log("Harmonizing IVIM series was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e:
        database.log("Harmonizing IVIM series was NOT completed; error: " + str(e)) 
        database.restore()

def harmonize_dce(database):
    start_time = time.time()
    database.log("Harmonizing DCE series has started!")
    try:
        harmonize.dce(database)
        database.log("Harmonizing DCE series was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e:
        database.log("Harmonizing DCE series was NOT completed; error: " + str(e)) 
        database.restore()

def harmonize_subject_name(database):
    start_time = time.time()
    database.log("Harmonizing subject name has started!")
    try:
        harmonize.subject_name(database)
        database.log("Harmonizing subject name was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e:
        database.log("Harmonizing subject name was NOT completed; error: " + str(e)) 
        database.restore()


## SEGMENTATION


def fetch_dl_models(database):
    start_time = time.time()
    database.log("Fetching deep-learning models has started")
    try:
        fetch_AI_model.dl_models(database)
        database.log("Fetching deep-learning models was completed --- %s seconds ---" % (int(time.time() - start_time)))
    except Exception as e:
        database.log("Fetching deep-learning models was NOT completed; error: "+str(e))
        database.restore()

def fetch_nnunet_models(database):
    start_time = time.time()
    database.log("Fetching deep-learning models has started")
    try:
        fetch_nnunet_model.nnunet_models(database)
        database.log("Fetching deep-learning models was completed --- %s seconds ---" % (int(time.time() - start_time)))
    except Exception as e:
        database.log("Fetching deep-learning models was NOT completed; error: "+str(e))
        database.restore()

def fetch_kidney_masks(database):
    start_time = time.time()
    database.log("Fetching kidney masks has started")
    try:
        fetch_Drive_mask.kidney_masks(database)
        database.log("Fetching kidney masks was completed --- %s seconds ---" % (int(time.time() - start_time)))
    except Exception as e:
        database.log("Fetching kidney masks was NOT completed; error: "+str(e))
        database.restore()


def segment_kidneys(database):
    start_time = time.time()
    database.log("Kidney segmentation has started")
    try:

        lk = database.series(SeriesDescription='LK')
        rk = database.series(SeriesDescription='RK')

        if len(lk) == 0 or len(rk) == 0:
            database.log("Starting AI kidney segmentation")
            segment.kidneys(database)
            database.log("AI Kidney segmentation was completed")
        else:
            database.log('Both masks were already present - no AI kidney segmentation was performed.')
        database.save()
        
    except Exception as e:
        database.log("AI Kidney segmentation was NOT completed; error: "+str(e))
        database.restore()
        raise RuntimeError('Critical step failed (kidney segmentation) - exiting pipeline.')

    database.log("Kidney segmentation was completed --- %s seconds ---" % (int(time.time() - start_time)))

def segment_kidneys_nnunet(database):
    start_time = time.time()
    database.log("Kidney segmentation has started")
    try:

        lk = database.series(SeriesDescription='LK')
        rk = database.series(SeriesDescription='RK')

        if len(lk) == 0 or len(rk) == 0:
            database.log("Starting nnunet kidney segmentation")
            segment.kidneys_nnunet(database)
            database.log("nnunet Kidney segmentation was completed")
        else:
            database.log('Both masks were already present - no nnunet kidney segmentation was performed.')
        database.save()
        
    except Exception as e:
        database.log("nnunet Kidney segmentation was NOT completed; error: "+str(e))
        database.restore()
        raise RuntimeError('Critical step failed (kidney segmentation) - exiting pipeline.')

    database.log("Kidney segmentation was completed --- %s seconds ---" % (int(time.time() - start_time)))

def segment_renal_sinus_fat(database):
    start_time = time.time()
    database.log("Sinus fat segmentation has started")
    try:
        segment.renal_sinus_fat(database)
        database.log("Sinus segmentation was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e:
        database.log("Sinus fat segmentation was NOT completed; error: "+str(e))
        database.restore()


def segment_aorta_on_dce(database):
    start_time = time.time()
    database.log("Aorta segmentation has started")
    try:
        segment.aorta_on_dce(database)
        database.log("Aorta segmentation was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e:
        database.log("Aorta segmentation was NOT completed; error: "+str(e))
        database.restore()


def segment_left_renal_artery(database):
    start_time = time.time()
    database.log("Left renal artery segmentation has started")
    try:
        segment.left_renal_artery(database)
        database.log("Left renal artery segmentation was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e:
        database.log("Left renal artery segmentation was NOT completed; error: "+str(e))
        database.restore()

def segment_right_renal_artery(database):
    start_time = time.time()
    database.log("Right renal artery segmentation has started")
    try:
        segment.right_renal_artery(database)
        database.log("Right renal artery segmentation was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e:
        database.log("Right renal artery segmentation was NOT completed; error: "+str(e))
        database.restore()


# Set up exports as individual steps
def export_segmentations(database):
    start_time = time.time()
    database.log("Export kidney segmentations has started")
    try:
        export.kidney_masks_as_dicom(database)
        export.kidney_masks_as_png(database)
        export.aif_as_png(database)
        export.right_renal_artery_masks_as_png(database)
        export.left_renal_artery_masks_as_png(database)
        database.log("Export kidney segmentations was completed --- %s seconds ---" % (int(time.time() - start_time)))
    except Exception as e:
        database.log("Export kidney segmentations was NOT completed; error: "+str(e))

def export_COR_MED_segmentations(database):
    start_time = time.time()
    database.log("Export kidney segmentations has started")

    try:
        export.kidney_masks_as_png(database,backgroud_series = 'Dixon_post_contrast_out_phase',RK_mask = 'RK', LK_mask = 'LK',mask_name = 'Whole Kidney_masks')
    except Exception as e:
        database.log("Export kidney segmentations was NOT completed; error: "+str(e))
    try:
        export.aif_as_png(database)
    except Exception as e:
        database.log("Export kidney segmentations was NOT completed; error: "+str(e))

    try:
        export.kidney_masks_as_png(database,backgroud_series = 'Dixon_post_contrast_out_phase',RK_mask = 'RKM', LK_mask = 'LKM',mask_name = 'Medulla_masks')
    except Exception as e:
        database.log("Export kidney segmentations was NOT completed; error: "+str(e))

    try:
        export.kidney_masks_as_png(database,backgroud_series = 'Dixon_post_contrast_out_phase',RK_mask = 'RKC', LK_mask = 'LKC', mask_name = 'Cortex_masks')
    except Exception as e:
        database.log("Export kidney segmentations was NOT completed; error: "+str(e))

    database.log("Export kidney segmentations was completed --- %s seconds ---" % (int(time.time() - start_time)))

## MODEL-DRIVEN MOTION CORRECTION



def mdreg_t1(database):
    start_time = time.time()
    database.log("Model-driven registration for T1 has started")
    try:
        mdr.T1(database)
        database.log("Model-driven registration for T1 was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e:
        database.log("Model-driven registration for T1 was NOT completed; error: "+str(e))
        database.restore()

def mdreg_t2(database):
    start_time = time.time()
    database.log("Model-driven registration for T2 has started")
    try:
        mdr.T2(database)
        database.log("Model-driven registration for T2 was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e:
        database.log("Model-driven registration for T2 was NOT completed; error: "+str(e))
        database.restore()


def mdreg_t2star(database):
    start_time = time.time()
    database.log("Model-driven registration for T2* has started")
    try:
        mdr.T2star(database)
        database.log("Model-driven registration for T2* was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e:
        database.log("Model-driven registration for T2* was NOT completed; error: "+str(e))
        database.restore()


def mdreg_mt(database):
    start_time = time.time()
    database.log("Model-driven registration for MT has started")
    try:
        mdr.MT(database)
        database.log("Model-driven registration for MT was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e:
        database.log("Model-driven registration for MT was NOT completed; error: "+str(e))
        database.restore()


def mdreg_dti(database):
    start_time = time.time()
    database.log("Model-driven registration for DTI has started")
    try:
        mdr.DTI(database)
        database.log("Model-driven registration for DTI was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e:
        database.log("Model-driven registration for DTI was NOT completed; error: "+str(e))
        database.restore()


def mdreg_ivim(database):
    start_time = time.time()
    database.log("Model-driven registration for IVIM has started")
    try:
        mdr.IVIM(database)
        database.log("Model-driven registration for IVIM was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e:
        database.log("Model-driven registration for IVIM was NOT completed; error: "+str(e))
        database.restore()


def mdreg_dce(database):
    start_time = time.time()
    database.log("Model-driven registration for DCE has started")
    try:
        mdr.DCE(database)
        database.log("Model-driven registration for DCE was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e:
        database.log("Model-driven registration for DCE was NOT completed; error: "+str(e))
        database.restore()


def export_mdreg(database):
    start_time = time.time()
    database.log("Exporting MDR results has started")
    try:
        export.mdreg(database)
        database.log("Exporting MDR results was completed --- %s seconds ---" % (int(time.time() - start_time)))
    except Exception as e:
        database.log("Exporting MDR results was NOT completed; error: "+str(e))
        database.restore()

    
## MAPPING

def map_T1(database):
    start_time = time.time()
    print('Starting T1 mapping')
    database.log("T1 mapping has started")
    try:
        mapping.T1(database)
        database.log("T1 mapping was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e: 
        database.log("T1 mapping was NOT completed; error: "+str(e))
        database.restore()


def map_T2(database):
    start_time = time.time()
    print('Starting T2 mapping')
    database.log("T2 mapping has started")
    try:
        mapping.T2(database)
        database.log("T2 mapping was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e: 
        database.log("T2 mapping was NOT completed; error: "+str(e))
        database.restore()

        
def map_T2star(database):
    start_time = time.time()
    print('Starting T2*')
    database.log("T2* mapping has started")
    try:
        mapping.T2star(database)
        database.log("T2* mapping was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e: 
        database.log("T2* mapping was NOT completed; error: "+str(e))
        database.restore()


def map_MT(database):
    start_time = time.time()
    print('Starting MTR')
    database.log("MTR mapping has started")
    try:
        mapping.MT(database)
        database.log("MTR mapping was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e: 
        database.log("MTR mapping was NOT completed; error: "+str(e))
        database.restore()


def map_DTI(database):
    start_time = time.time()
    print('Starting DTI')
    database.log("DTI mapping has started")
    try:
        mapping.DTI(database)
        database.log("DTI mapping was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e: 
        database.log("DTI-FA & ADC mapping was NOT completed; error: "+str(e))
        database.restore()


def map_IVIM(database):
    start_time = time.time()
    print('Starting IVIM')
    database.log("IVIM mapping has started")
    try:
        mapping.IVIM(database)
        database.log("IVIM mapping was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e: 
        database.log("IVIM mapping was NOT completed; error: "+str(e))
        database.restore()


def map_DCE(database):
    start_time = time.time()
    print('Starting DCE_MAX')
    database.log("DCE mapping has started")
    try:
        mapping.DCE(database)
        database.log("DCE mapping was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e: 
        database.log("DCE mapping was NOT completed; error: "+str(e))
        database.restore()

def export_mapping(database):
    start_time = time.time()
    database.log("Exporting mapping results has started")
    try:
        export.mapping(database)
        database.log("Exporting maping results was completed --- %s seconds ---" % (int(time.time() - start_time)))
    except Exception as e:
        database.log("Exporting mapping results was NOT completed; error: "+str(e))



## ALIGNMENT
        

def align_dixon(database):
    start_time = time.time()
    print('Starting DIXON alignment')
    database.log("DIXON alignment has started")
    try:
        align.dixon(database)
        database.log("DIXON alignment was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e: 
        database.log("DIXON alignment was NOT completed; error: "+str(e))
        database.restore()
        

def align_T1(database):
    start_time = time.time()
    print('Starting T1 alignment')
    database.log("T1 alignment has started")
    try:
        align.t1(database)
        database.log("T1 alignment was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e: 
        database.log("T1 alignment was NOT completed; error: "+str(e))
        database.restore()

def align_T2(database):
    start_time = time.time()
    print('Starting T2 alignment')
    database.log("T2 alignment has started")
    try:
        align.t2(database)
        database.log("T2 alignment was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e: 
        database.log("T2 alignment was NOT completed; error: "+str(e))
        database.restore()

def align_T2star(database):
    start_time = time.time()
    print('Starting T2* alignment')
    database.log("T2* alignment has started")
    try:
        align.t2star(database)
        database.log("T2* alignment was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e: 
        database.log("T2* alignment was NOT completed; error: "+str(e))
        database.restore()

def align_MT(database):
    start_time = time.time()
    print('Starting MT alignment')
    database.log("MT alignment has started")
    try:
        align.mt(database)
        database.log("MT alignment was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e: 
        database.log("MT alignment was NOT completed; error: "+str(e))
        database.restore()

def align_IVIM(database):
    start_time = time.time()
    print('Starting IVIM alignment')
    database.log("IVIM alignment has started")
    try:
        align.ivim(database)
        database.log("IVIM alignment was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e: 
        database.log("IVIM alignment was NOT completed; error: "+str(e))
        database.restore()

def align_DTI(database):
    start_time = time.time()
    print('Starting DTI alignment')
    database.log("DTI alignment has started")
    try:
        align.dti(database)
        database.log("DTI alignment was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e: 
        database.log("DTI alignment was NOT completed; error: "+str(e))
        database.restore()

def align_DCE(database):
    start_time = time.time()
    print('Starting DCE alignment')
    database.log("DCE alignment has started")
    try:
        align.dce(database)
        database.log("DCE alignment was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e: 
        database.log("DCE alignment was NOT completed; error: "+str(e))
        database.restore()

def align_ASL(database):
    start_time = time.time()
    print('Starting ASL alignment')
    database.log("ASL alignment has started")
    try:
        align.asl(database)
        database.log("ASL alignment was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e: 
        database.log("ASL alignment was NOT completed; error: "+str(e))
        database.restore()

def align_T1w(database):
    start_time = time.time()
    print('Starting T1w alignment')
    database.log("T1w alignment has started")
    try:
        align.t1w(database)
        database.log("T1w alignment was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e: 
        database.log("T1w alignment was NOT completed; error: "+str(e))
        database.restore()

def export_alignment(database):
    start_time = time.time()
    print('Exporting alignment')
    database.log("Export alignment has started")
    try:
        export.alignment(database)
        database.log("Export alignment was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e: 
        database.log("Export alignment was NOT completed; error: "+str(e))
        database.restore()

## MEASURE
        
def measure_kidney_volumetrics(database):
    start_time = time.time()
    database.log("Kidney volumetrics has started")
    try:
        measure.kidney_volumetrics(database)
        database.log("Kidney volumetrics was completed --- %s seconds ---" % (int(time.time() - start_time)))
    except Exception as e:
        database.log("Kidney volumetrics was NOT completed; error: "+str(e))

def measure_sinus_fat_volumetrics(database):
    start_time = time.time()
    database.log("Sinus fat volumetrics has started")
    try:
        measure.sinus_fat_volumetrics(database)
        database.log("Sinus fat volumetrics was completed --- %s seconds ---" % (int(time.time() - start_time)))
    except Exception as e:
        database.log("Sinus fat volumetrics was NOT completed; error: "+str(e))

def measure_t1_maps(database):
    start_time = time.time()
    database.log("T1 map measurement has started")
    try:
        measure.t1_maps(database)
        database.log("T1 map measurement was completed --- %s seconds ---" % (int(time.time() - start_time)))
    except Exception as e:
        database.log("T1 map measurement was NOT completed; error: "+str(e))  

def measure_t2_maps(database):
    start_time = time.time()
    database.log("T2 map measurement has started")
    try:
        measure.t2_maps(database)
        database.log("T2 map measurement was completed --- %s seconds ---" % (int(time.time() - start_time)))
    except Exception as e:
        database.log("T2 map measurement was NOT completed; error: "+str(e))     

def measure_t2star_maps(database):
    start_time = time.time()
    database.log("T2* map measurement has started")
    try:
        measure.t2star_maps(database)
        database.log("T2* map measurement was completed --- %s seconds ---" % (int(time.time() - start_time)))
    except Exception as e:
        database.log("T2* map measurement was NOT completed; error: "+str(e))   

def measure_mt_maps(database):
    start_time = time.time()
    database.log("MT map measurement has started")
    try:
        measure.mt_maps(database)
        database.log("MT map measurement was completed --- %s seconds ---" % (int(time.time() - start_time)))
    except Exception as e:
        database.log("MT map measurement was NOT completed; error: "+str(e)) 

def measure_ivim_maps(database):
    start_time = time.time()
    database.log("IVIM map measurement has started")
    try:
        measure.ivim_maps(database)
        database.log("IVIM map measurement was completed --- %s seconds ---" % (int(time.time() - start_time)))
    except Exception as e:
        database.log("IVIM map measurement was NOT completed; error: "+str(e)) 

def measure_dti_maps(database):
    start_time = time.time()
    database.log("DTI map measurement has started")
    try:
        measure.dti_maps(database)
        database.log("DTI map measurement was completed --- %s seconds ---" % (int(time.time() - start_time)))
    except Exception as e:
        database.log("DTI map measurement was NOT completed; error: "+str(e))  

def measure_dce_maps(database):
    start_time = time.time()
    database.log("DCE map measurement has started")
    try:
        measure.dce_maps(database)
        database.log("DCE map measurement was completed --- %s seconds ---" % (int(time.time() - start_time)))
    except Exception as e:
        database.log("DCE map measurement was NOT completed; error: "+str(e))  

def measure_asl_maps(database):
    start_time = time.time()
    database.log("ASL perfusion measurement has started")
    try:
        measure.asl_maps(database)
        database.log("ASL perfusion measurement was completed --- %s seconds ---" % (int(time.time() - start_time)))
    except Exception as e:
        database.log("ASL perfusion measurement was NOT completed; error: "+str(e))  

        

## ROI analysis
        

def fill_gaps(database):
    start_time = time.time()
    database.log("Filling gaps has started")
    try:
        align.fill_gaps(database)
        database.log("Filling gaps was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e:
        database.log("Filling gaps was NOT completed; error: "+str(e))
        database.restore()


def cortex_medulla(database):
    start_time = time.time()
    database.log("Cortex-medulla segmentation has started")
    try:
        segment.cortex_medulla(database)
        database.log("Cortex-medulla segmentations was completed --- %s seconds ---" % (int(time.time() - start_time)))
        database.save()
    except Exception as e:
        database.log("Cortex-medulla segmentations was NOT completed; error: "+str(e))
        database.restore()
        

def roi_fit_T1(database):
    start_time = time.time()
    database.log("ROI analysis for T1 has started")
    try:
        roi_fit.T1(database)
        database.log("ROI analysis for T1 was completed --- %s seconds ---" % (int(time.time() - start_time)))
    except Exception as e:
        database.log("ROI analysis for T1 was NOT completed; error: "+str(e))

def roi_fit_T2(database):
    start_time = time.time()
    database.log("ROI analysis for T2 has started")
    try:
        roi_fit.T2(database)
        database.log("ROI analysis for T2 was completed --- %s seconds ---" % (int(time.time() - start_time)))
    except Exception as e:
        database.log("ROI analysis for T2 was NOT completed; error: "+str(e))

def roi_fit_T2star(database):
    start_time = time.time()
    database.log("ROI analysis for T2* has started")
    try:
        roi_fit.T2star(database)
        database.log("ROI analysis for T2* was completed --- %s seconds ---" % (int(time.time() - start_time)))
    except Exception as e:
        database.log("ROI analysis for T2* was NOT completed; error: "+str(e))

def roi_fit_PC(database):
    start_time = time.time()
    database.log("ROI analysis for PC has started")
    try:
        roi_fit.PC(database)
        database.log("ROI analysis for PC was completed --- %s seconds ---" % (int(time.time() - start_time)))
    except Exception as e:
        database.log("ROI analysis for PC was NOT completed; error: "+str(e))

def roi_fit_IVIM(database):
    start_time = time.time()
    database.log("ROI analysis for IVIM has started")
    try:
        roi_fit.IVIM(database)
        database.log("ROI analysis for IVIM was completed --- %s seconds ---" % (int(time.time() - start_time)))
    except Exception as e:
        database.log("ROI analysis for IVIM was NOT completed; error: "+str(e))

def roi_fit_DCE(database):
    start_time = time.time()
    database.log("ROI analysis for DCE has started")
    try:
        roi_fit.dce(database)
        database.log("ROI analysis for DCE was completed --- %s seconds ---" % (int(time.time() - start_time)))
    except Exception as e:
        database.log("ROI analysis for DCE was NOT completed; error: "+str(e))

def roi_fit_DCE_cm(database):
    start_time = time.time()
    database.log("ROI analysis for DCE (Cortex-Medulla) has started")
    try:
        roi_fit.dce_cm(database)
        database.log("ROI analysis for DCE (Cortex-Medulla) was completed --- %s seconds ---" % (int(time.time() - start_time)))
    except Exception as e:
        database.log("ROI analysis for DCE (Cortex-Medulla) was NOT completed; error: "+str(e))

def roi_fit_DTI(database):
    start_time = time.time()
    database.log("ROI analysis for DTI has started")
    try:
        roi_fit.DTI(database)
        database.log("ROI analysis for DTI was completed --- %s seconds ---" % (int(time.time() - start_time)))
    except Exception as e:
        database.log("ROI analysis for DTI was NOT completed; error: "+str(e))