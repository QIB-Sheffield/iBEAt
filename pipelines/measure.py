import os
import pandas as pd
import numpy as np

from dbdicom.extensions import skimage, vreg

import tempfile
import datetime

import SimpleITK as sitk
import radiomics
from radiomics import featureextractor
import nibabel as nib



def _master_table_file(folder):
    # Get filename and create folder if needed.
    results_path = folder.path() + '_output'
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    return os.path.join(results_path, folder.PatientName[0] + '_biomarkers.csv')

def _load_master_table(folder):
    # Read master table
    filename_csv = _master_table_file(folder)
    if os.path.isfile(filename_csv):
        return pd.read_csv(filename_csv)
    # If the master table does not exist, create it
    row_headers = ['PatientID', 'SeriesDescription', 'Region of Interest', 'Parameter', 'Value', 'Unit', 'Biomarker', 'StudyDescription']
    master_table = pd.DataFrame(columns=row_headers)
    master_table.to_csv(filename_csv, index=False)
    return master_table

def read_master_table(folder, biomarker):
    table = _load_master_table(folder)
    table = table[table.Biomarker == biomarker]
    if table.empty:
        raise ValueError('Biomarker ' + biomarker +' has not yet been calculated.')
    return table.Value.values[0]

def _update_master_table(folder, table):
    master_table = _load_master_table(folder)
    master_table = pd.concat([master_table, table], ignore_index=True)
    master_table = master_table.drop_duplicates(subset='Biomarker', keep='last', ignore_index=True)
    filename_csv = _master_table_file(folder)
    master_table.to_csv(filename_csv, index=False)

def add_rows(folder, rows):
    row_headers = ['PatientID', 'SeriesDescription', 'Region of Interest', 'Parameter', 'Value', 'Unit', 'Biomarker', 'StudyDescription']
    table = pd.DataFrame(data=rows, columns=row_headers)
    _update_master_table(folder, table)


def kidney_volumetrics(folder):
    left  = folder.series(SeriesDescription='LK')
    right = folder.series(SeriesDescription='RK')

    kidneys = [left, right]
    features = skimage.volume_features(kidneys)
    
    features['Biomarker'] = features['SeriesDescription'] + '-' + features['Parameter']
    features['Region of Interest'] = 'Kidney'
    _update_master_table(folder, features)
    return features

def kidney_volumetrics_paper_volumetry_edited_skimage(folder):

    left  = folder.series(SeriesDescription='LK_E')
    if left == []:
        left  = folder.series(SeriesDescription='LK_ed')
        if left == []:
            left  = folder.series(SeriesDescription='LK')

    right  = folder.series(SeriesDescription='RK_E')
    if right == []:
        right  = folder.series(SeriesDescription='RK_ed')
        if right == []:
            right  = folder.series(SeriesDescription='RK')


    kidneys = [left, right]
    features = skimage.volume_features(kidneys)
    features['Biomarker'] = features['SeriesDescription'] + '-' + features['Parameter']
    features['Region of Interest'] = 'Kidney'
    _update_master_table(folder, features)
    return features

def kidney_volumetrics_paper_volumetry_edited_pyradiomics(database):

    def save_nifti(array, affine, filename):
        """Helper function to save array as NIfTI."""
        image = nib.Nifti1Image(array, affine)
        nib.save(image, filename)


    def get_feature_class(feature_name):
        """Helper to extract feature class from full feature name."""
        if feature_name.startswith('original_'):
            feature_name = feature_name[len('original_'):]
        return feature_name.split('_')[0]

    def extract_texture_features(extractor, left_mask, right_mask, series, label_key, temp_folder, biomarker_units):
        """Extract texture features from left kidney, right kidney, and both kidneys (BK) for a given image series."""
        features = []
        arr_img, _ = series.array('SliceLocation', pixels_first=True)
        affine_img = series.affine()
        instance = series.instance()

        # Register left and right masks to the image series
        overlay_left = vreg.map_to(left_mask, series)
        overlay_right = vreg.map_to(right_mask, series)

        arr_left, _ = overlay_left.array('SliceLocation', pixels_first=True)
        arr_right, _ = overlay_right.array('SliceLocation', pixels_first=True)
        arr_bk = arr_left + arr_right

        # Define file paths
        image_path = os.path.join(temp_folder, f'{label_key}.nii.gz')
        left_mask_path = os.path.join(temp_folder, f'left_{label_key}.nii.gz')
        right_mask_path = os.path.join(temp_folder, f'right_{label_key}.nii.gz')
        bk_mask_path = os.path.join(temp_folder, f'bk_{label_key}.nii.gz')

        # Save images and masks
        save_nifti(arr_img, affine_img, image_path)
        save_nifti(arr_left, affine_img, left_mask_path)
        save_nifti(arr_right, affine_img, right_mask_path)
        save_nifti(arr_bk, affine_img, bk_mask_path)

        # Loop over all three masks
        for mask_path, label, region in [
            (left_mask_path, 'LK', 'Kidney'),
            (right_mask_path, 'RK', 'Kidney'),
            (bk_mask_path, 'BK', 'Kidneys')
        ]:
            result = extractor.execute(image_path, mask_path)
            for key, value in result.items():
                if key.startswith('diagnostics_'):
                    continue
                feature_class = get_feature_class(key)
                biomarker_name = key.replace('original_', '')
                unit = biomarker_units.get(biomarker_name, 'unitless')

                features.append({
                    'PatientID': instance.PatientID,
                    'SeriesDescription': label,
                    'Region of Interest': region,
                    'Parameter': key,
                    'Value': value,
                    'Unit': unit,
                    'Biomarker': f"{label}-{key}-{label_key}",
                    'StudyDescription': f"PyRadiomics-{feature_class}"
                })

        return features

    left  = database.series(SeriesDescription='LK_E')
    if left == []:
        left  = database.series(SeriesDescription='LK_ed')
        if left == []:
            left  = database.series(SeriesDescription='LK')

    right  = database.series(SeriesDescription='RK_E')
    if right == []:
        right  = database.series(SeriesDescription='RK_ed')
        if right == []:
            right  = database.series(SeriesDescription='RK')

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_csv = f"radiomics_features_{timestamp}.csv"

    results_path = database.path() + '_output'
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    # Load series
    water = database.series(SeriesDescription='Dixon_post_contrast_water')

    if not (left and right and water):
        raise RuntimeError("Missing required image series (LK, RK, or Dixon water).")

    # Register masks to water image
    overlay_left = vreg.map_to(left[0], water[0])
    overlay_right = vreg.map_to(right[0], water[0])

    # Extract arrays
    arr_water, _ = water[0].array('SliceLocation', pixels_first=True)
    arr_left, _ = overlay_left.array('SliceLocation', pixels_first=True)
    arr_right, _ = overlay_right.array('SliceLocation', pixels_first=True)

    affine = left[0].affine()
    instance = water[0].instance()

    biomarker_units = {
        'firstorder_Energy': 'Intensity^2 units',
        'firstorder_TotalEnergy': 'Intensity^2 units',
        'firstorder_Entropy': 'unitless',
        'firstorder_Kurtosis': 'unitless',
        'firstorder_Mean': 'Intensity units',
        'firstorder_Median': 'Intensity units',
        'firstorder_Minimum': 'Intensity units',
        'firstorder_Maximum': 'Intensity units',
        'firstorder_Skewness': 'unitless',
        'firstorder_StandardDeviation': 'Intensity units',
        'firstorder_Variance': 'Intensity^2 units',
        'firstorder_RootMeanSquared': 'Intensity units',
        'shape_VoxelVolume': 'mm^3',
        'shape_SurfaceArea': 'mm^2',
        'shape_SurfaceVolumeRatio': '1/mm',
        'shape_Compactness1': 'unitless',
        'shape_Compactness2': 'unitless',
        'shape_Sphericity': 'unitless',
        'shape_SphericalDisproportion': 'unitless',
        'shape_Maximum3DDiameter': 'mm',
        'shape_MajorAxisLength': 'mm',
        'shape_MinorAxisLength': 'mm',
        'shape_Elongation': 'unitless',
        'shape_Flatness': 'unitless',
        'glcm_Contrast': 'unitless',
        'glcm_Correlation': 'unitless',
        'glcm_DifferenceEntropy': 'unitless',
        'glcm_Id': 'unitless',
        'glcm_Idm': 'unitless',
        'glcm_Imc1': 'unitless',
        'glcm_Imc2': 'unitless',
        'glcm_InverseVariance': 'unitless',
    }

    all_features = []

    with tempfile.TemporaryDirectory(dir=os.getcwd()) as temp_folder:
        paths = {
            'image': os.path.join(temp_folder, 'Dixon_water.nii.gz'),
            'left': os.path.join(temp_folder, 'left.nii.gz'),
            'right': os.path.join(temp_folder, 'right.nii.gz'),
        }

        save_nifti(arr_water, affine, paths['image'])
        save_nifti(arr_left, affine, paths['left'])
        save_nifti(arr_right, affine, paths['right'])

        # All features for water
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.disableAllFeatures()
        extractor.enableFeatureClassByName('shape')

        for mask_key, label in [('left', 'LK'), ('right', 'RK')]:
            result = extractor.execute(paths['image'], paths[mask_key])
            for key, value in result.items():
                if key.startswith('diagnostics_'):
                    continue
                feature_class = get_feature_class(key)
                biomarker_name = key.replace('original_', '')
                unit = biomarker_units.get(biomarker_name, 'unitless')

                biomarker_label = label + '-' + (
                    key if feature_class == 'shape'
                    else key + '-' + instance.SeriesDescription.split('_')[-1]
                )

                all_features.append({
                    'PatientID': instance.PatientID,
                    'SeriesDescription': label,
                    'Region of Interest': 'Kidney',
                    'Parameter': key,
                    'Value': value,
                    'Unit': unit,
                    'Biomarker': biomarker_label,
                    'StudyDescription': 'PyRadiomics-' + feature_class
                })

        # Texture-only features for fat, in_phase, out_phase
        extractor.disableAllFeatures()
        for cls in ['glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']:
            extractor.enableFeatureClassByName(cls)

        extra_series = {
            'fat': database.series(SeriesDescription='Dixon_post_contrast_fat'),
            'in_phase': database.series(SeriesDescription='Dixon_post_contrast_in_phase'),
            'out_phase': database.series(SeriesDescription='Dixon_post_contrast_out_phase'),
            'water': database.series(SeriesDescription='Dixon_post_contrast_water'),

        }

        for key, series_list in extra_series.items():
            if series_list:
                features = extract_texture_features(
                    extractor, left[0], right[0], series_list[0], key, temp_folder, biomarker_units
                )
                all_features.extend(features)
            else:
                print(f"Series {key} not found — skipping.")

        # Export features
        df = pd.DataFrame(all_features)
        df.to_csv(os.path.join(results_path,output_csv), index=False)
        print(f"Radiomics features saved to: {output_csv}")
    
def sinus_fat_volumetrics(folder):
    left  = folder.series(SeriesDescription='LKSF')
    right = folder.series(SeriesDescription='RKSF')
    sinus_fat = [left, right]
    features = skimage.volume_features(sinus_fat)
    features['Biomarker'] = features['SeriesDescription'] + '-' + features['Parameter']
    features['Region of Interest'] = 'Renal Sinus Fat'
    _update_master_table(folder, features)
    return features


def t1_maps(folder):
    seq = 'T1m_magnitude_mdr_moco'
    pars = ['T1', 'T1FAcorr']
    units = ['msec', 'deg']
    return features(folder, seq, pars, units)

def t2_maps(folder):
    seq = 'T2m_magnitude_mdr_moco'
    #pars = ['T2', 'T2FAcorr']
    pars = ['T2']
    #units = ['msec', 'deg']
    units = ['msec']
    return features(folder, seq, pars, units)

def t2star_maps(folder):
    seq = 'T2starm_magnitude_mdr_moco'
    pars = [ 'T2star','f_fat']
    units = ['msec','']
    return features(folder, seq, pars, units)

def mt_maps(folder):
    seq = 'MT_mdr_moco'
    pars = ['MTR']
    units = ['%']
    return features(folder, seq, pars, units)

def ivim_maps(folder):
    seq = 'IVIM_mdr_moco'
    pars = ['D','D_star','Perfusion_fraction']
    units = ['','','']
    return features(folder, seq, pars, units)

def dti_maps(folder):
    seq = 'DTI_mdr_moco'
    pars = ['MD','RD','AD','Planarity','Linearity','Sphericity','FA']
    units = ['','','','','','','']
    return features(folder, seq, pars, units)

def dce_maps(folder):
    seq = 'DCE_mdr_moco'
    pars = ['AVD', 'RPF', 'MTT']
    units = ['mL/100mL', 'mL/min/100mL', 'sec']
    return features(folder, seq, pars, units)

# Needs to be brought in line with the others - same format
def asl_maps(folder):
    for ROI in ['LK','RK','LKC','RKC','LKM','RKM']:
        try:
            kidney  = folder.series(SeriesDescription=ROI)
            rbf = folder.series(SeriesDescription='RBF - ' + ROI[:2])
            features = vreg.mask_statistics(kidney, rbf)
            features['Biomarker'] = ROI[:2] + '-' + 'RBF' + '-' + features['Parameter']
            features['Region of Interest'] = 'Kidney'
            _update_master_table(folder, features)
        except:
            print('Cannot find: RBF - ' + ROI)
            folder.log('Cannot find: RBF - ' + ROI)
    return features


def features(folder, seq, pars, units):
    cnt=0
    for p, par in enumerate(pars):
        try:
            for subroi in ['','C','M']:
                vals = []
                for ROI in ['LK'+subroi,'RK'+subroi]: 
                    folder.progress(cnt+1, len(pars)*9, 'Exporting metrics (' + par + ' on ' + ROI + ')')
                    cnt+=1
                    desc = seq + '_' + par + '_map_' + ROI[:2] + '_align'

                    kidney = folder.series(SeriesDescription=ROI)[0]
                    series = folder.series(SeriesDescription=desc)[0]
                    kidney_vals = vreg.mask_values(kidney, series)
                    vals.append(kidney_vals)

                    # update master table
                    features = vreg.mask_data_statistics(kidney_vals, kidney, series)
                    features['Biomarker'] = [ROI + '-' + par + '-' + metric for metric in features['Parameter'].values]
                    features['Unit'] = units[p]
                    features['SeriesDescription'] = ROI
                    features['Region of Interest'] = 'Kidney'
                    _update_master_table(folder, features)

                ROI = 'BK' + subroi
                folder.progress(cnt+1, len(pars)*9, 'Exporting metrics (' + par + ' on ' + ROI + ')')
                cnt+=1
                vals = np.concatenate(vals)
                features = vreg.mask_data_statistics(vals, kidney, series)
                features['Biomarker'] = [ROI + '-' + par + '-' + metric for metric in features['Parameter'].values]
                features['Unit'] = units[p]
                features['SeriesDescription'] = ROI
                features['Region of Interest'] = 'Kidney'
                _update_master_table(folder, features)
        except:
            print('cannot find ' + str(ROI) +' ' + par)
            folder.log('cannot find ' + str(ROI) +' ' + par)
            continue

    return features


