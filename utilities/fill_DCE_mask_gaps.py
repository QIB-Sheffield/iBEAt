import vreg
import numpy as np

def class_with_ones(img_series, mask, prior):
    # Read and process the mask
    if mask is not None:
        mask.message('Reading mask array..')
        mask_arr, _ = vreg.mask_array(mask, on=img_series, dim='AcquisitionTime')
        mask_arr = np.squeeze(mask_arr)
    
    # Extract the raw image and prior cluster mask
    arr, headers = img_series.array(['SliceLocation', 'AcquisitionTime'], pixels_first=True)
    arr = np.squeeze(arr)

    cluster_mask_arr_raw, headers = prior.array(['SliceLocation', 'AcquisitionTime'], pixels_first=True)
    cluster_mask_arr = np.zeros_like(cluster_mask_arr_raw)
    cluster_mask_arr = np.where((cluster_mask_arr_raw > 0.5) & (cluster_mask_arr_raw <= 1.5), 1, cluster_mask_arr)
    cluster_mask_arr = np.where((cluster_mask_arr_raw > 1.5) & (cluster_mask_arr_raw <= 2.5), 2, cluster_mask_arr)
    cluster_mask_arr = np.squeeze(cluster_mask_arr)

    # Identify missing pixels and replace them with 1
    missing_pixels = (mask_arr == 1) & (cluster_mask_arr == 0)
    cluster_mask_arr[missing_pixels] = 1  # Replace missing pixels with 1

    # Expand dimensions for saving
    cluster_mask_arr = cluster_mask_arr[..., np.newaxis, np.newaxis]
    
    # Save results in DICOM
    clusters = img_series.new_sibling(SeriesDescription='Local Classification with Adjacent Pixels')
    clusters.set_array(cluster_mask_arr, headers, pixels_first=True)

    return clusters