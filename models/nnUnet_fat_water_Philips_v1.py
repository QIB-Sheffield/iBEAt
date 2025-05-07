"""
Script to apply the pretrained UNETR for 3D kidney segmentation.

You will need:

- dixon: A 3D numpy array with post-contrast out of phase DIXON data
- weights: A file with pretrained model weights

Apply the model:

- segments = UNETR_kidneys_v1.apply(dixon, weights)

Clean the results and extract masks as numpy arrays:

- left_kidney, right_kidney = UNETR_kidneys_v1.kidney_masks(segments)

or do both in one go:

- left_kidney, right_kidney = UNETR_kidneys_v1.kidneys(dixon, weights)
"""

import os
import numpy as np 
import scipy.ndimage as ndi
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import nibabel as nib
import shutil

# The default model name must be the same as the filename of this module
filename = os.path.basename(os.path.abspath(__file__))
filename = filename[:-3] + ".pth"

# Required - DICOM series description of validated data
trained_on = [
              "Dixon_post_contrast_out_phase",
              "Dixon_post_contrast_in_phase",
             ]
              

# Required
def apply(input_array:np.ndarray, file:str)->np.ndarray:


    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cpu'),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True)

    # initializes the network architecture, loads the checkpoint
    folder_path = os.path.dirname(file)
    nested_folder = os.path.join(folder_path, "Dataset001_Fatwater", "nnUNetTrainer__nnUNetPlans__3d_fullres")
    os.makedirs(nested_folder, exist_ok=True)

    predictor.initialize_from_trained_model_folder(
    nested_folder,
    use_folds='all',
    checkpoint_name='checkpoint_best.pth'
    )
    # Preprocess the input according to the model’s requirements
    # In nnUNetV2, preprocessing is usually defined in the model plans

    temp_folder_results = os.path.join(folder_path,"temp_results")
    temp_folder_data_to_test = os.path.join(folder_path,"temp_results",'data_to_test')
    os.makedirs(temp_folder_data_to_test, exist_ok=True)

    affine = np.eye(4) 
    nii_out_ph = nib.Nifti1Image(input_array[0,...], affine)
    nib.save(nii_out_ph, os.path.join(temp_folder_data_to_test, 'Dixon_999_0000.nii.gz'))

    nii_in_ph = nib.Nifti1Image(input_array[1,...], affine)
    nib.save(nii_in_ph, os.path.join(temp_folder_data_to_test, 'Dixon_999_0001.nii.gz'))

    predictor.predict_from_files(temp_folder_data_to_test,temp_folder_results, save_probabilities=False, overwrite=False)

    # Load the NIfTI file
    nifti_file = nib.load(os.path.join(temp_folder_results,'Dixon_999.nii.gz'))

    # Get the image data as a NumPy array
    image_data = nifti_file.get_fdata()

    shutil.rmtree(temp_folder_results)

    return image_data


def kidneys(input_array:np.ndarray, file:str, **kwargs)->tuple:
    """return kdiney masks directly from data

    Args:
        input_array (np.ndarray): 3D numpy array (x,y,z) with DIXON data.
        file (str): filepath to file with model weights.
        overlap (float, optional): optimization parameter. Defaults to 0.2.

    Returns:
        tuple: A tuple of 3D numpy arrays (left_kidney, right_kidney) with masks for the kidneys.
    """
    fat_water = apply(input_array, file, **kwargs)
    return fat_water