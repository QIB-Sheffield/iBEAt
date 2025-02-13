"""
Pipelines for between-sequence alignment.
"""

from dbdicom.extensions import vreg as vreg_dicom
from dbdicom.extensions import elastix
from dbdicom.pipelines import input_series

export_study = "3: Alignment"



#########################################
# Alignment of maps to DIXON kidney masks
#########################################

# The following pipelines lists the series to align with kidney masks for 
# the different contrast mechansisms.
#
# The transformations are found by coregistering the first series in the 
# list to the kidney masks. 
# 
# The same transformation is then applied to all the other series.


def t1(database):
    series_desc = [ 
        'T1m_magnitude_mdr_moco_T1_map',
        'T1m_magnitude_mdr_moco_S0_map',
        'T1m_magnitude_mdr_moco_T1FAcorr_map',
        'T1m_magnitude_mdr_moco',
    ]
    return _align_2d(database, series_desc)
    

def t2(database):
    series_desc = [ 
        'T2m_magnitude_mdr_moco_T2_map',
        'T2m_magnitude_mdr_moco_S0_map',
        'T2m_magnitude_mdr_moco',
    ]
    return _align_2d(database, series_desc)


def t2star(database):
    series_desc = [ 
        'T2starm_magnitude_mdr_moco_T2star_map',
        'T2starm_magnitude_mdr_moco_S0_map',
        'T2starm_magnitude_mdr_moco_f_fat_map',
        'T2starm_magnitude_mdr_moco',
    ]
    return _align_2d(database, series_desc)


def ivim(database):
    series_desc = [ 
        'IVIM_mdr_moco_S0_map',
        'IVIM_mdr_moco_D_map',
        'IVIM_mdr_moco_D_star_map',
        'IVIM_mdr_moco_Perfusion_fraction_map',
        'IVIM_mdr_moco',
    ]
    return _align_2d(database, series_desc)


def dti(database):
    series_desc = [ 
        'DTI_mdr_moco_MD_map',
        'DTI_mdr_moco_RD_map',
        'DTI_mdr_moco_AD_map',
        'DTI_mdr_moco_Planarity_map',
        'DTI_mdr_moco_Linearity_map',
        'DTI_mdr_moco_Sphericity_map',
        'DTI_mdr_moco_FA_map',
        'DTI_mdr_moco_fit',
        'DTI_mdr_moco',
    ]
    return _align_2d(database, series_desc)


def dce(database):
    series_desc = [ 
        'DCE_mdr_moco_AUC_map',
        'DCE_mdr_moco_AVD_map',
        'DCE_mdr_moco_RPF_map',
        'DCE_mdr_moco_MTT_map',
        'DCE_mdr_moco',
    ]
    return _align_2d(database, series_desc)

def t1w(database):

    desc = [ 
        'LK',
        'RK',  
        'T1w_magnitude',
    ]
    return _align_2d(database, desc)

def mt(database):
    series_desc = [ 
        'MT_mdr_moco_AVR_map',
        'MT_mdr_moco_MTR_map',
        'MT_mdr_moco',
    ]
    return _align_3d(database, series_desc)


def asl(database):
    series_desc = [   
        'ASL_M0_moco',
        'ASL_RBF_moco',
    ]
    return _align_3d(database, series_desc)




###########################################
# Alignment of pre- and post-contrast DIXON
###########################################


def dixon(database):
    # A freeform transformation is found by coregistering the precontrast 
    # DIXON fat map to the post-contrast DIXON fat map. The same 
    # transformation is then applied to the other precontrast DIXON images.

    # Get the approrpiate DICOM series
    desc = [   
        'Dixon_out_phase',
        'Dixon_in_phase',
        'Dixon_water',
        'Dixon_fat',
        'Dixon_post_contrast_fat',
    ]
    series, study = input_series(database, desc, export_study)
    if series is None:
        raise RuntimeError('Cannot perform DIXON alignment: not all required data exist.')

    # Perform the corgistration on the DICOM series
    coregistered, followers = elastix.coregister_3d_to_3d(
        series[3], series[4],
        transformation = "Freeform", 
        metric = "AdvancedMeanSquares",
        final_grid_spacing = 25.0,
        apply_to = series[:3],
    )

    # Move the results to the new study
    coregistered.move_to(study)
    for series in followers:
        series.move_to(study)
    return coregistered, followers


# TODO: This does not belong in align. Move to segment?
def fill_gaps(database):

    to_fill = [
        'DCE_mdr_moco_AVD_map',
        'DCE_mdr_moco_RPF_map',
        'DCE_mdr_moco_MTT_map',        
    ]
    ref = ['Dixon_post_contrast_fat']

    output = []
    for kidney in ['LK','RK']:
        desc = ref + [kidney] + [d + '_' + kidney + '_align' for d in to_fill]
        series, study = input_series(database, desc, export_study)
        if series is None:
            raise RuntimeError('Cannot fill gaps between slices on LK: not all required data exist.')
        for ms_series in series[2:]:
            output_series = vreg_dicom.fill_slice_gaps(ms_series, series[0], slice_thickness=1.0, mask=series[1])
            output_series.move_to(study)
            output.append(output_series)

    return output



##################
# HELPER FUNCTIONS
##################



def _align_3d(database, desc):
    # Coregister moving volumes

    study, series, lk, rk, bk = _get_data(database, desc)

    moving = series[0].volume()
    callback = lambda perc, msg: series[0].progress(perc, 100, msg)
    affine = _coreg(moving, lk, rk, bk, callback) 

    return _save_data(study, series, affine)


def _align_2d(database, desc):
    # Coregister moving volume slice-by-slice

    study, series, lk, rk, bk = _get_data(database, desc)

    affine = {'LK':[], 'RK':[]}
    moving = series[0].volumes()
    for z, sz in enumerate(moving):
        callback = lambda perc, msg: series[0].progress(perc, 100, msg + ' (slice ' + str(z+1) +')')
        az = _coreg(sz, lk, rk, bk, callback)
        affine['LK'].append(az['LK'])
        affine['RK'].append(az['RK'])

    return _save_data(study, series, affine)


def _get_data(database, desc):

    # Get the appropriate DICOM series
    desc = ['Dixon_water', 'LK', 'RK'] + desc
    series, study = input_series(database, desc, export_study)
    if series is None:
        raise RuntimeError(
            'Cannot perform '+desc[3]+' alignment: not all required data exist.')

    # Extract the volumes from the series
    dixon = series[0].volume()
    lk = series[1].volume()
    rk = series[2].volume()
    bk = lk.slice_like(dixon).add(rk)

    # Use bounding boxes speed up the computation
    bk = bk.bounding_box()
    lk = lk.bounding_box()
    rk = rk.bounding_box()

    return study, series[3:], lk, rk, bk


def _save_data(study, series, affine):
    # Save results to the database
    transfo = []
    dims = ('SliceLocation', 'InstanceNumber')
    for map in series:
        map_desc = map.SeriesDescription
        for kidney in ['LK','RK']:
            desc = map_desc + '_' + kidney + '_align'
            align = map.copy_to(study, SeriesDescription=desc)
            align.set_affines(affine[kidney], dims)
            transfo.append(align)
    return transfo[:2]


def _coreg(volume, lk, rk, bk, callback):
    # This is the actual coregistration operating on vreg volumes.
    #
    # Registration is performed by 3D translation in 2 steps - first 
    # align caorsely to both kidneys, then fine tune kidney per kidney.
    # 
    # The function returns a dictionary with two affines: one for 
    # coregistration to the left kidney, and one for the right.

    # Settings
    options = {'coords': 'volume'}
    optimizer = {'method': 'brute'}

    # Translate to both kidneys
    optimizer['grid'] = [[-20, 20, 20],
                         [-20, 20, 20],
                         [-5, 5, 5]]
    optimizer['callback'] = lambda perc: callback(perc, 'Coregistering to both kidneys')
    tbk = volume.find_translate_to(bk, optimizer=optimizer, **options) 
    volume = volume.translate(tbk, **options)

    # Per kidney fine tuning
    optimizer['grid'] = 3*[[-2, 2, 10]]
    optimizer['callback'] = lambda perc: callback(perc, 'Coregistering to left kidney')
    tlk = volume.find_translate_to(lk, optimizer=optimizer, **options)
    optimizer['callback'] = lambda perc: callback(perc, 'Coregistering to right kidney')
    trk = volume.find_translate_to(rk, optimizer=optimizer, **options)

    # Return affines for each kidney
    return {
        'LK': volume.translate(tlk, **options).affine,
        'RK': volume.translate(trk, **options).affine,
    }