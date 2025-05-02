import numpy as np
from dbdicom.extensions import skimage, scipy, dipy, sklearn
from dbdicom.pipelines import input_series
from models import DCE_aorta, PC, UNETR_kidneys_v1, nnUnet_Dixon_v1
import utilities.zenodo_link as UNETR_zenodo
import utilities.zenodo_link_nnunet as nnunet_zenodo
import utilities.fill_DCE_mask_gaps as fill_DCE_mask_gaps
import os
import shutil

export_study = '0: Segmentations'


def kidneys(database):

    # Get weights file and check if valid 
    # if not os.path.isfile(weights):
    #     msg = 'The weights file ' + weights + ' has not been found. \n'
    #     msg += 'Please check that the file with model weights is in the folder, and is named ' + UNETR_kidneys_v1.filename
    #     database.dialog.information(msg)
    #     return

    unetr, unetr_link= UNETR_zenodo.main()
    weights = os.path.join(os.path.dirname(database.path()),unetr)

    database.message('Segmenting kidneys. This could take a few minutes. Please be patient..')

    # Get appropriate series and check if valid
    #series = database.series(SeriesDescription=UNETR_kidneys_v1.trained_on)
    series, study = input_series(database, UNETR_kidneys_v1.trained_on,export_study)
    if series is None:
        msg = 'Cannot autosegment the kidneys: series ' + UNETR_kidneys_v1.trained_on + ' not found.'
        raise RuntimeError(msg)

    if database.PatientName [0:4] == '7128':
    # Loop over the series and create the mask
    #desc = sery.instance().SeriesDescription

        array_out  , header   = series[0].array(['SliceLocation'], pixels_first=True, first_volume=True)
        array_out = array_out[:, :, ::-1,...]
        array_in   , header_in    = series[1].array(['SliceLocation'], pixels_first=True, first_volume=True)
        array_in = array_in[:, :, ::-1,...]
        array_water, header_water = series[2].array(['SliceLocation'], pixels_first=True, first_volume=True)
        array_water = array_water[:, :, ::-1,...]
        array_fat  , header_fat   = series[3].array(['SliceLocation'], pixels_first=True, first_volume=True)
        array_fat = array_fat[:, :, ::-1,...]
        array = np.stack((array_out, array_in, array_water, array_fat), axis=0)
    
    else:

        array_out  , header   = series[0].array(['SliceLocation'], pixels_first=True, first_volume=True)
        array_in   , header_in    = series[1].array(['SliceLocation'], pixels_first=True, first_volume=True)
        array_water, header_water = series[2].array(['SliceLocation'], pixels_first=True, first_volume=True)
        array_fat  , header_fat   = series[3].array(['SliceLocation'], pixels_first=True, first_volume=True)
        array = np.stack((array_out, array_in, array_water, array_fat), axis=0)
    # Calculate predictions 
    masks = UNETR_kidneys_v1.apply(array, weights)
    left_kidney, right_kidney = UNETR_kidneys_v1.kidney_masks(masks)

    # Save UNETR output
    result = study.new_child(SeriesDescription = 'BK')
    result.set_array(masks, header, pixels_first=True)
    # result[['WindowCenter','WindowWidth']] = [1.0, 2.0]

    # Save and display left kidney data
    left = study.new_child(SeriesDescription = 'LK')
    left.set_array(left_kidney, header, pixels_first=True)
    # left[['WindowCenter','WindowWidth']] = [1.0, 2.0]
    
    # Save and display right kidney data
    right = study.new_child(SeriesDescription = 'RK')
    right.set_array(right_kidney, header, pixels_first=True)
    # right[['WindowCenter','WindowWidth']] = [1.0, 2.0]

    database.save()

    kidneys = []
    kidneys.append(left) 
    kidneys.append(right)

    return kidneys

def kidneys_nnunet(database): #ADAPT TO THE nnUet for 

    # Get weights file and check if valid 
    # if not os.path.isfile(weights):
    #     msg = 'The weights file ' + weights + ' has not been found. \n'
    #     msg += 'Please check that the file with model weights is in the folder, and is named ' + UNETR_kidneys_v1.filename
    #     database.dialog.information(msg)
    #     return

    #unetr, unetr_link= UNETR_zenodo.main()
    nnUnet, unetr_link= nnunet_zenodo.main()
    weights = os.path.join(database.path(), nnUnet)

    #copied_folder_path = shutil.copytree(os.path.join(os.path.dirname(database.path()),'Dataset001_Dixon'),os.path.join(database.path(),'Dataset001_Dixon'))
    #copied_folder_path = shutil.copytree(os.path.join(os.path.dirname(database.path()),'nnUNetTrainer__nnUNetPlans__3d_fullres'),os.path.join(database.path(),'nnUNetTrainer__nnUNetPlans__3d_fullres'))

    database.message('Segmenting kidneys. This could take a few minutes. Please be patient..')

    # Get appropriate series and check if valid
    #series = database.series(SeriesDescription=UNETR_kidneys_v1.trained_on)
    sery, study = input_series(database, nnUnet_Dixon_v1.trained_on,export_study)
    if sery is None:
        msg = 'Cannot autosegment the kidneys: series ' + nnUnet_Dixon_v1.trained_on + ' not found.'
        raise RuntimeError(msg)

    array_in,    _      = sery[0].array(['SliceLocation'], pixels_first=True, first_volume=True)
    array_out,   _      = sery[1].array(['SliceLocation'], pixels_first=True, first_volume=True)
    array_water, _      = sery[2].array(['SliceLocation'], pixels_first=True, first_volume=True)
    array_fat,   header = sery[3].array(['SliceLocation'], pixels_first=True, first_volume=True)

    if database.PatientName[0:4] == '7128':
        array_in_temp  = array_in[:, :, ::-1,...]
        array_in = array_in_temp
        array_out_temp = array_out[:, :, ::-1,...]
        array_out = array_out_temp
        array_water_temp = array_water[:, :, ::-1,...]
        array_water = array_water_temp
        array_fat_temp = array_fat[:, :, ::-1,...]
        array_fat = array_fat_temp

    array_to_predict = np.stack([array_in, array_out, array_water, array_fat], axis=0)

    # Calculate predictions 

    masks = nnUnet_Dixon_v1.apply(array_to_predict, weights)

    shutil.rmtree(weights)
    shutil.rmtree(os.path.join(database.path(),'nnUNetTrainer__nnUNetPlans__3d_fullres'))

    rk, lk = nnUnet_Dixon_v1.kidney_masks(masks)

    if database.PatientName[0:4] == '7128':
        masks = masks[:, :, ::-1,...]
        rk    = rk[:, :, ::-1,...]
        lk    = lk[:, :, ::-1,...]

    # Save UNETR output
    
    result = study.new_child(SeriesDescription = 'BK')
    result.set_array(masks, header, pixels_first=True)
    # result[['WindowCenter','WindowWidth']] = [1.0, 2.0]
    
    rk_series = study.new_child(SeriesDescription = 'RK')
    rk_series .set_array(rk, header, pixels_first=True)
    # left[['WindowCenter','WindowWidth']] = [1.0, 2.0]
    
    lk_series = study.new_child(SeriesDescription = 'LK')
    lk_series.set_array(lk, header, pixels_first=True)
    # left[['WindowCenter','WindowWidth']] = [1.0, 2.0]

    database.save()

def renal_sinus_fat(folder):

    fat = folder.series(SeriesDescription='Dixon_post_contrast_fat')
    lk  = folder.series(SeriesDescription='LK')
    rk  = folder.series(SeriesDescription='RK')

    kidneys = lk+rk
    sf_series = []

    if len(kidneys)==[]:
        msg = 'Cannot perform renal sinus fat segmentation: no kidney masks are available.'
        raise RuntimeError(msg)

    fat_image_masked, fat_mask = dipy.median_otsu(fat[0], median_radius=1, numpass=1)

    for kidney in kidneys:
        kidney_hull = skimage.convex_hull_image_3d(kidney)
        sinus_fat = scipy.image_calculator(fat_mask, kidney_hull, 'series 1 * series 2', integer=True)
        #sinus_fat_open = skimage.opening_3d(sinus_fat)
        sinus_fat_largest = scipy.extract_largest_cluster_3d(sinus_fat)
        # closing after selecting largest cluster
        sinus_fat_largest.SeriesDescription = kidney.instance().SeriesDescription + 'SF'
        sf_series.append(sinus_fat_largest)
        # Cleanup
        kidney_hull.remove()
        #sinus.remove()
        sinus_fat.remove()
    
    fat_image_masked.remove()
    fat_mask.remove()   

    return sf_series

def renal_sinus_fat_open_close(folder):

    fat = folder.series(SeriesDescription='Dixon_post_contrast_fat')
    lk  = folder.series(SeriesDescription='LK')
    rk  = folder.series(SeriesDescription='RK')

    kidneys = lk+rk
    sf_series = []

    if len(kidneys)==[]:
        msg = 'Cannot perform renal sinus fat segmentation: no kidney masks are available.'
        raise RuntimeError(msg)

    fat_image_masked, fat_mask = dipy.median_otsu(fat[0], median_radius=1, numpass=1)

    for kidney in kidneys:
        kidney_hull = skimage.convex_hull_image_3d(kidney)
        sinus_fat = scipy.image_calculator(fat_mask, kidney_hull, 'series 1 * series 2', integer=True)
        sinus_fat_open = skimage.opening_3d(sinus_fat)
        sinus_fat_largest = scipy.extract_largest_cluster_3d(sinus_fat_open)
        sinus_fat_largest_close = skimage.closing_3d(sinus_fat_largest)
        sinus_fat_largest.SeriesDescription = kidney.instance().SeriesDescription + 'SF'
        sf_series.append(sinus_fat_largest_close)
        # Cleanup
        kidney_hull.remove()
        #sinus.remove()
        sinus_fat.remove()
    
    fat_image_masked.remove()
    fat_mask.remove()   

    return sf_series

def renal_sinus_fat_close(folder):

    fat = folder.series(SeriesDescription='Dixon_post_contrast_fat')
    lk  = folder.series(SeriesDescription='LK')
    rk  = folder.series(SeriesDescription='RK')

    kidneys = lk+rk
    sf_series = []

    if len(kidneys)==[]:
        msg = 'Cannot perform renal sinus fat segmentation: no kidney masks are available.'
        raise RuntimeError(msg)

    fat_image_masked, fat_mask = dipy.median_otsu(fat[0], median_radius=1, numpass=1)

    for kidney in kidneys:
        kidney_hull = skimage.convex_hull_image_3d(kidney)
        sinus_fat = scipy.image_calculator(fat_mask, kidney_hull, 'series 1 * series 2', integer=True)
        #sinus_fat_open = skimage.opening_3d(sinus_fat)
        sinus_fat_largest = scipy.extract_largest_cluster_3d(sinus_fat)
        sinus_fat_largest_close = skimage.closing_3d(sinus_fat_largest)
        sinus_fat_largest.SeriesDescription = kidney.instance().SeriesDescription + 'SF'
        sf_series.append(sinus_fat_largest_close)
        # Cleanup
        kidney_hull.remove()
        #sinus.remove()
        sinus_fat.remove()
    
    fat_image_masked.remove()
    fat_mask.remove()   

    return sf_series


def compute_whole_kidney_canvas(database):
    series_desc = [
        'Dixon_post_contrast_fat',
        'Dixon_post_contrast_out_phase'
    ] 
    features, study = input_series(database, series_desc, export_study)
    if features is None:
        return
    clusters = sklearn.sequential_kmeans(features, n_clusters=2, multiple_series=True)
    for c in clusters:
        c.move_to(study)
    return clusters


def aorta_on_dce(folder):

    desc = "DCE_aorta_axial_fb"
    series, study = input_series(folder, desc, export_study)
    if series is None:
        raise RuntimeError('Cannot create DCE-AIF mask: series ' + desc + ' does not exist. ')

    axial, header = series.array(['AcquisitionTime'], pixels_first=True, first_volume=True)

    aif_mask = DCE_aorta.segment(axial)

    aif_mask_series = study.new_series(SeriesDescription='DCE-AIF')
    aif_mask_series.set_array(aif_mask, header[0], pixels_first=True)

    return aif_mask_series


def left_renal_artery(folder):

    desc = [
        'PC_left_delta_magnitude',
        'PC_left_velocity',
    ]
    series, study = input_series(folder, desc, export_study)
    if series is None:
        raise RuntimeError('Cannot create PC-RA mask: some series are missing.')
    
    dims = ['InstanceNumber']
    dx = series[0].values('PixelSpacing')[0][0]

    # Calculate left artery mask:
    mag, mag_hdr_left = series[0].array(dims, pixels_first=True, first_volume=True)
    vel, vel_hdr_left = series[1].array(dims, pixels_first=True, first_volume=True)
    left_mask = PC.renal_artery_mask(mag, vel, pixel_spacing=dx)
    #left_mask = PC.renal_artery_mask_alternative(mag)

    # Save as DICOM
    left_mask_series = study.new_series(SeriesDescription='PC-LRA')
    left_mask_series.set_array(left_mask, mag_hdr_left[0], pixels_first=True)

    return left_mask_series

def right_renal_artery(folder):

    desc = [
        'PC_right_delta_magnitude',
        'PC_right_velocity',
    ]

    series, study = input_series(folder, desc, export_study)
    if series is None:
        raise RuntimeError('Cannot create PC-RA mask: some series are missing.')
    
    dims = ['InstanceNumber']
    dx = series[0].values('PixelSpacing')[0][0]

    # Calculate right artery mask:
    mag, mag_hdr_right = series[0].array(dims, pixels_first=True, first_volume=True)
    vel, vel_hdr_right = series[1].array(dims, pixels_first=True, first_volume=True) 
    right_mask = PC.renal_artery_mask(mag, -vel, pixel_spacing=dx)

    # Save as DICOM
    right_mask_series = study.new_series(SeriesDescription='PC-RRA')
    right_mask_series.set_array(right_mask, mag_hdr_right[0], pixels_first=True)


    return right_mask_series


def cortex_medulla(database):

    features = [
        'DCE_mdr_moco_AVD_map',
        'DCE_mdr_moco_RPF_map',
        'DCE_mdr_moco_MTT_map',  
    ]

    output = []
    for kidney in ['LK','RK']:
        desc = [kidney] + [f + '_' + kidney + '_align_fill' for f in features]
        series, study = input_series(database, desc, export_study)
        if series is None:
            raise ValueError('Cannot separate cortex and medulla for kidney '+kidney+': required sequences are missing.')
        clusters, cluster_features = sklearn.kmeans(series[1:], series[0], n_clusters=3, multiple_series=True, return_features=True)
        # Background = cluster with smallest AVD
        background = np.argmin([c[0] for c in cluster_features])
        # Cortex = cluster with largest RPF
        cortex = np.argmax([c[1] for c in cluster_features]) 
        # Medulla = cluster with largest MTT
        medulla = np.argmax([c[2] for c in cluster_features])
        # Check
        remainder = {0,1,2} - {background, cortex, medulla}
        if len(remainder) > 0:
            raise ValueError('Problem separating cortex and medulla: identified clusters do not have the expected values.')
        clusters[cortex].SeriesDescription = kidney + 'C'
        clusters[cortex].move_to(study)
        clusters[medulla].SeriesDescription = kidney + 'M'
        clusters[medulla].move_to(study)
        clusters[background].SeriesDescription = kidney + 'B'
        clusters[background].move_to(study)
        cm = scipy.image_calculator(clusters[cortex], clusters[medulla], '+')
        cm.SeriesDescription = kidney + 'CM'
        output += clusters + [cm]
    return output

def cortex_medulla_local(database):
    
    export_study = '0: Segmentations'

    desc = [ 
        'LK', 
        'LKM',
        'RK', 
        'RKM',
    ]

    series, study = input_series(database, desc, export_study)

    RKC = scipy.image_calculator(series[0], series[1], 'series 1 - series 2')
    LKC = scipy.image_calculator(series[2], series[3], 'series 1 - series 2')

    database.save()

def fill_DCE_cor_med_masks(database):

    export_study = '0: Segmentations'

    desc = [ 
        'LKC', 
        'LKM',
        'RKC', 
        'RKM',
    ]

    series, study = input_series(database, desc, export_study)

    RKM_2 = scipy.series_calculator(series[3], operation='a * series', param=2)
    LKM_2 = scipy.series_calculator(series[1], operation='a * series', param=2)

    LK_prior = scipy.image_calculator(series[0], LKM_2, 'series 1 + series 2',series_desc='LK_prior')
    RK_prior = scipy.image_calculator(series[2], RKM_2, 'series 1 + series 2',series_desc='RK_prior')

    features = [
    'T1w_magnitude',
    ]

    for kidney in ['LK','RK']:
        desc = [kidney] + [features[0] + '_' + kidney + '_align_fill'] + [kidney + '_prior']
        series, study = input_series(database, desc, export_study)
        if series is None:
            raise ValueError('Cannot separate cortex and medulla for kidney '+kidney+': required sequences are missing.')
        
        clusters = fill_DCE_mask_gaps.class_with_ones(series[1], series[0], series[-1])
        clusters.SeriesDescription = kidney[0:2] + '_prior_res_nb'

        #clusters.move_to(study)
        database.save()