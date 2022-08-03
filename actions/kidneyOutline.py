
import numpy as np
import matplotlib.pyplot as plt
import wezel
import cv2 as cv2
from skimage import feature
import actions.rightClicks as rightclick
import actions.reggrow as reg

class kidoutline(wezel.Action):

    def run(self, app):
        ############################################################################
        # Loop through the selected series and extract the images to a numpy array #
        ############################################################################
        array_img_seg =[]

        for i in range(len(app.get_selected(3))):
            SequenceName =app.get_selected(3)[i]['SeriesDescription']
            possibleNames = ['par_T1','par_T2star','DTI_FA_Map','par_ADC','_T1map']


            if any (x in SequenceName for x in possibleNames):
                series_map_seg = app.get_selected(3)[i]
                array_map_seg, header_map_seg = series_map_seg.array(['SliceLocation','AcquisitionTime'], pixels_first=True)
            
            elif 'cor_bh_fat' in SequenceName:
                series_map_seg = app.get_selected(3)[i]
                array_map_seg, header_map_seg = series_map_seg.array(['SliceLocation','AcquisitionTime'],pixels_first=True)
                series_img_seg = series_map_seg

            elif 'moco' in SequenceName:
                series_img_seg = app.get_selected(3)[i]
                array_img_seg, header_img_seg = series_img_seg.array(['SliceLocation','AcquisitionTime'],pixels_first=True)

            elif 'magnitude' in SequenceName[-9:]:
                series_img_seg = app.get_selected(3)[i]
                array_img_seg, header_img_seg = series_img_seg.array(['SliceLocation','AcquisitionTime'],pixels_first=True)

            if(array_img_seg ==[]):
                array_img_seg = array_map_seg
                series_img_seg = series_map_seg
        ############################################################################
 

        #####################################################
        # Loop through the SLICES of the numpy array images #
        #####################################################

        mask_LeftKidneyFinal  = np.zeros((np.shape(array_img_seg)[0],np.shape(array_img_seg)[1],np.shape(array_img_seg)[2]))
        mask_RightKidneyFinal = np.zeros((np.shape(array_img_seg)[0],np.shape(array_img_seg)[1],np.shape(array_img_seg)[2]))



        for slice in range (np.shape(array_img_seg)[2]):

            pixelX_Right = []
            pixelY_Right = []
            pixelX_Left = []
            pixelY_Left = []

            array_img_seg_test = np.squeeze(array_img_seg[:,:,slice,:,0])
            array_map_seg_test = np.squeeze(array_map_seg[:,:,slice,0,0])

            if 'DTI_FA_Map' in series_map_seg['SeriesDescription']:
                array_map_seg_test[array_map_seg_test>1]==1                   #create a threshold to allow a better normalization [0 256], type ('uint8') - needed for open CV

            if 'par_ADC' in series_map_seg['SeriesDescription']:
                array_map_seg_test[array_map_seg_test>0.004]==0.004           #create a threshold to allow a better normalization [0 256], type ('uint8') - needed for open CV

            if 'cor_bh_fat' in SequenceName:
                array_map_seg_test = np.squeeze(array_img_seg[:,:,slice,:,0])
                array_img_seg_test = array_map_seg_test
            
            if (len(np.shape(np.squeeze(header_map_seg)))==2):
                pixelSize = np.squeeze(header_map_seg)[0,0].PixelSpacing     
            elif (len(np.shape(np.squeeze(header_map_seg)))==1):
                pixelSize = np.squeeze(header_map_seg)[0].PixelSpacing     #needed to filter detected contours using values that make sense for the kidney
        #######################################################

        ############################################################################################################################
        # Open images and ask the user to right click on the LEFT (FIRST), RIGHT(SECOND) Kidneys and record the clicked coordinates #
        ############################################################################################################################
            newImg = ((255-0)/(np.max(array_map_seg_test)-np.min(array_map_seg_test)))*(array_map_seg_test-np.min(array_map_seg_test)) #normalize between 0-255 to be cv2 compatible
            img = newImg.astype('uint8')    
            #INPUT: image, OUTPUT:coordinates of selected pixels - open image, right click to select points                                                                                           #cv2 type  
            corrdinates = rightclick.main(img)                          
            if corrdinates ==[]:     #if no cordinates were selected, continue to the next slice
                continue
            
            if (np.shape(corrdinates)[0]<2):
                if (corrdinates [0][1] < array_img_seg.shape[0]/2):
                    pixelX_Right = corrdinates[0][1]
                    pixelY_Right = corrdinates[0][0]
                else:
                    pixelX_Left = corrdinates[0][1]
                    pixelY_Left = corrdinates[0][0]

                    #RightKidney
                    #pixelX_Right = corrdinates[1][1]
                    #pixelY_Right = corrdinates[1][0]
            else:
                    pixelX_Right = corrdinates[1][1]
                    pixelY_Right = corrdinates[1][0]
                    pixelX_Left = corrdinates[0][1]
                    pixelY_Left = corrdinates[0][0]
        ###########################################################################################################################

        ##################################################################################
        # BOOST contrast between kidney and other organs at the given selected sequences #
        ##################################################################################
    
            if 'T1map' in series_img_seg['SeriesDescription']:
                array_img_seg_test_MaxMin = np.squeeze((array_img_seg_test[:,:,0]-array_img_seg_test[:,:,1])*array_map_seg_test) #for T1
                array_img_seg_test_MaxMin_cleaned = array_img_seg_test_MaxMin
                array_img_seg_test_MaxMin_cleaned[array_img_seg_test_MaxMin_cleaned<0] =0                                        #remove possible negative values
            elif 'T2map' in series_img_seg['SeriesDescription']:
                array_img_seg_test_MaxMin = np.squeeze((array_img_seg_test[:,:,0])*array_map_seg_test)                           #for T2
                array_img_seg_test_MaxMin_cleaned = array_img_seg_test_MaxMin 
            else:
                array_img_seg_test_MaxMin_cleaned = np.squeeze(array_map_seg_test)                                               #dixon fat                                                      

        #################################################################################

        #################################################################################################
        # Apply seed growing algorithm if the scan is the the cor_bh_fat scan and visualize the results #
        #################################################################################################
            if 'cor_bh_fat' in series_img_seg['SeriesDescription']: #for cor_bh_fat seed growing is more suitable than edge detection
                
                if pixelY_Left != []:
                    mask_LeftKidney = kidneySeeding(app,array_img_seg_test_MaxMin_cleaned,pixelY_Left,pixelX_Left)
                else:
                    mask_LeftKidney = np.zeros(np.shape(array_img_seg_test_MaxMin_cleaned))

                if pixelY_Right != []:    
                    mask_RightKidney = kidneySeeding(app,array_img_seg_test_MaxMin_cleaned,pixelY_Right,pixelX_Right)
                else:
                    mask_LeftKidney = np.zeros(np.shape(array_img_seg_test_MaxMin_cleaned))

                _vizualizeKidneysWithMask(array_img_seg_test_MaxMin_cleaned,mask_LeftKidney,mask_RightKidney,500)
                mask_LeftKidneyFinal[:,:,slice] = mask_LeftKidney
                mask_RightKidneyFinal[:,:,slice] = mask_RightKidney

                continue
        ######################################################################

        ############################################################
        # Apply edge detection algorithm and visualize the results #
        ############################################################
            if pixelY_Left != []: 
                mask_LeftKidney = kidneySegmentation(app,array_img_seg_test_MaxMin_cleaned,pixelY_Left,pixelX_Left,pixelSize,side='Left')
            else:
                mask_LeftKidney = np.zeros(np.shape(array_img_seg_test_MaxMin_cleaned))

            if pixelY_Right != []: 
                mask_RightKidney = kidneySegmentation(app,array_img_seg_test_MaxMin_cleaned,pixelY_Right,pixelX_Right,pixelSize,side='Right')
            else:
                mask_RightKidney = np.zeros(np.shape(array_img_seg_test_MaxMin_cleaned))

            _vizualizeKidneysWithMask(array_map_seg_test,mask_LeftKidney,mask_RightKidney,np.max(array_map_seg_test[(mask_LeftKidney+mask_RightKidney)==1]))
            mask_LeftKidneyFinal[:,:,slice] = mask_LeftKidney
            mask_RightKidneyFinal[:,:,slice] = mask_RightKidney
        ############################################################
        Left_Mask_series = series_map_seg.SeriesDescription + "KID_L"
        Left_Mask_series = series_map_seg.new_sibling(SeriesDescription=Left_Mask_series)
        Left_Mask_series.set_array(np.squeeze(mask_LeftKidneyFinal),np.squeeze(header_map_seg[:,0,0]),pixels_first=True)

        Right_Mask_series = series_map_seg.SeriesDescription + "KID_R"
        Right_Mask_series = series_map_seg.new_sibling(SeriesDescription=Right_Mask_series)
        Right_Mask_series.set_array(np.squeeze(mask_RightKidneyFinal),np.squeeze(header_map_seg[:,0,0]),pixels_first=True)

        app.refresh()   



def _vizualizeKidneysWithMask(imageToPlot,mask_LeftKidney,mask_RightKidney,Vmax):

    fig, axs = plt.subplots(1,2,figsize=(5,2))
    axs[0].set_title('Original Image',fontsize=24)
    axs[0].imshow(np.transpose(imageToPlot),vmin=0,vmax=Vmax)
    axs[1].set_title('Original Image + Calculated Kidney Masks',fontsize=24)
    axs[1].imshow(np.transpose(imageToPlot + (mask_LeftKidney+mask_RightKidney)*2000),vmin=0,vmax=Vmax)
    plt.show()


############################################################
# seed growing algorithm using the clicked pixels as seeds #
############################################################
def kidneySeeding(app,img_array,pixelY,pixelX):

    img_array_Blurred = cv2.GaussianBlur(img_array, (3,3),cv2.BORDER_DEFAULT)
    
    seeds = [reg.Point(pixelX,pixelY)]
    seedThreshold = 11
    
    mask_Kidney = reg.regionGrow(img_array_Blurred,seeds,seedThreshold)
    if len(mask_Kidney[mask_Kidney==1])>5000:                              #if result of the mask have a crazy high number of pixels, increase the threshold
        for n in range(seedThreshold):
            seedThresholdTemp = seedThreshold-n
            mask_Kidney = reg.regionGrow(img_array_Blurred,seeds,seedThresholdTemp)
            if len(mask_Kidney[mask_Kidney==1])>5000:
                continue
            else:
                break
    return mask_Kidney

##########################################################################################
# edge detection algorithm using the clicked pixels as clues where the contour should be #
##########################################################################################
def kidneySegmentation(app,img_array,pixelY,pixelX,pixelSize,side=None):

    img_array_Blurred = cv2.GaussianBlur(img_array, (31,31),cv2.BORDER_DEFAULT)

    KidneyBlurred = np.zeros(np.shape(img_array_Blurred))
    if side == 'Left':
        KidneyBlurred[int(np.shape(img_array_Blurred)[0]/2):np.shape(img_array_Blurred)[0],:] = img_array_Blurred[int(np.shape(img_array_Blurred)[0]/2):np.shape(img_array_Blurred)[0],:]
    elif side == 'Right':
        KidneyBlurred[0:int(np.shape(img_array_Blurred)[0]/2),:] = img_array_Blurred[0:int(np.shape(img_array_Blurred)[0]/2),:]

    sigmaCanny = 0
    edges = feature.canny(KidneyBlurred, sigma =sigmaCanny)
    edges= edges.astype(np.uint8)
    #plt.imshow(edges)

    maxIteration = 10
    maxIteration_2 =5

    Kidney=[]
    #dilate edges until you find a potential renal contour 
    for j in range(maxIteration):

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1+j,1+j))
        dilated = cv2.dilate(edges, kernel)
        #plt.imshow(dilated)
        cnts_Kidney,hierarchy_Kidney = cv2.findContours(dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #loop through the different contours until you find a potential renal contour 
        for i in range(len(cnts_Kidney)):
            cntTemp = cnts_Kidney[i]
            if cv2.contourArea(cntTemp)*pixelSize[0]*pixelSize[1]>1500: #check if the area of the contour is suitable with the kidneys
                dist = cv2.pointPolygonTest(cntTemp,(pixelY,pixelX),True)
                
                if dist > 0:
                    #print('Dilation iteration: ' +str(j))
                    #print('Contour Number: ' +str(i))
                    #print('Distance: ' +str(dist))
                    #print('ROI Area: ' +str(cv2.contourArea(cntTemp)) +' pixels')
                    #print('Son: '+str(hierarchy_Kidney[0,i][2]))
                    #print('Grandfather: '+str(hierarchy_Kidney[0,i][3]))

                    Kidney = cntTemp
                    mask_Kidney = np.ones(np.shape(img_array))
                    cv2.drawContours(mask_Kidney,[Kidney],0,(0,255,0),thickness=cv2.FILLED)
                    mask_Kidney = np.abs(mask_Kidney + 1 - 2)
                    
                    kernel_mask = np.ones((j+3,j+3)).astype(np.uint8)
                    edges_Kidney_new = (cv2.erode(mask_Kidney,kernel_mask)*edges).astype(np.uint8)
                    
                    kernel_new = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))
                    edges_Kidney_new = cv2.erode(edges_Kidney_new,kernel_new).astype(np.uint8)
                    edges_Kidney_new = cv2.dilate(edges_Kidney_new,kernel_new).astype(np.uint8)

                    edges_Kidney_new_dilated = cv2.dilate(edges_Kidney_new, kernel_new)
                    cnts_Kidney_new,hierarchy_Kidney_new = cv2.findContours(edges_Kidney_new_dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
                    cnts_Kidney_new_Sorted = sorted(cnts_Kidney_new, key=cv2.contourArea, reverse=True)

                    #check if contours needed to be removed from the main mask (nasty pelvis pixels)
                    for i_2 in range(len(cnts_Kidney_new_Sorted)):
                        cntTemp_new = cnts_Kidney_new_Sorted[i_2]
                        if cv2.contourArea(cntTemp_new)==0:
                            break

                        cntHull = cv2.convexHull(cntTemp_new, returnPoints=True)
                        #mask_Kidney_son = np.ones(np.shape(img_array))
                        #cv2.drawContours(mask_Kidney_son,[cntHull],0,(0,255,0),thickness=cv2.FILLED)
                        #plt.imshow(mask_Kidney_son)


                        if (cv2.contourArea(cntHull) < 0.5*cv2.contourArea(cntTemp) and cv2.contourArea(cntHull) > 0.03*cv2.contourArea(cntTemp)):
                            
                            Kidney_son = cntHull

                            mask_Kidney_son = np.ones(np.shape(img_array))
                            cv2.drawContours(mask_Kidney_son,[Kidney_son],0,(0,255,0),thickness=cv2.FILLED)
                            mask_Kidney_son = np.abs(mask_Kidney_son + 1 - 2)
                            mask_Kidney = mask_Kidney - mask_Kidney_son
                            mask_Kidney[mask_Kidney<0]=0

            if Kidney!=[]:
                #print(' Kindey Found')
                return mask_Kidney
    