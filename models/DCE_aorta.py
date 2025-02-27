import numpy as np
import cv2

from utilities.improc import region_grow_thresh

def segment(axial):

    cutRatio=0.25             #create a window around the center of the image where the aorta is
    filter_kernel=(15,15)     #gaussian kernel for smoothing the image to destroy any noisy single high intensity filter
    threshold = 2     #threshold for the region growing algorithm

    # Calculate max signal enhancement over a window around the center
    cy, cx = int(axial.shape[0]/2), int(axial.shape[1]/2)
    y0, y1 = int(cy-cy*cutRatio), int(cy+cy*cutRatio)
    x0, x1 = int(cx-cx*cutRatio), int(cx+cx*cutRatio)

    axwin = np.zeros(axial.shape)
    axwin[y0:y1, x0:x1,:] = axial[y0:y1, x0:x1,:]
    axenh = np.max(axwin,axis=2) - np.min(axwin,axis=2)

    # Get 3 seed points with maximal enhhancement values
    axenh_blur = cv2.GaussianBlur(axenh, filter_kernel,cv2.BORDER_DEFAULT)
    _, _, _, p1 = cv2.minMaxLoc(axenh_blur)
    axenh_blur[p1[1],p1[0]] = 0
    _, _, _, p2 = cv2.minMaxLoc(axenh_blur)
    axenh_blur[p2[1],p2[0]] = 0
    _, _, _, p3 = cv2.minMaxLoc(axenh_blur)

    max_iteration = 20
    for i in range(max_iteration):
        seeds = [[p1[1],p1[0]], [p2[1],p2[0]], [p3[1],p3[0]]]
        print(seeds)    
        aif_mask = region_grow_thresh(axenh, seeds, threshold)
        print (seeds)
        if len(aif_mask[aif_mask==1]) >= 25:
            
            break
        threshold += 1
        aif_mask = []

    return aif_mask



