import numpy as np

from dbdicom.pipelines import input_series
import dcmri

from pipelines.roi_fit import load_aif

import models.t1
import models.t2
import models.T2star
import models.DTI
import models.IVIM
from models.relaxometry import signalSequenceT2prep
from scipy.optimize import curve_fit

export_study = '2: Parameter maps'

def Dixon_post_contrast_water_dominant(folder):

    # Find source DICOM data
    series_Dixon_fat, study_Dixon_fat, desc_Dixon_fat                   = _map_input   (folder, "Dixon_post_contrast_fat")
    series_Dixon_in_phase, study_Dixon_in_phase, desc_Dixon_in_phase    = _map_input   (folder, "Dixon_post_contrast_in_phase")
    series_Dixon_out_phase, study_Dixon_out_phase, desc_Dixon_out_phase = _map_input   (folder, "Dixon_post_contrast_out_phase")

    # Load data
    array_Dixon_fat, header_Dixon_fat             = series_Dixon_fat.array('SliceLocation', pixels_first=True)
    array_Dixon_in_phase, header_Dixon_in_phase   = series_Dixon_in_phase.array('SliceLocation', pixels_first=True)
    array_Dixon_out_phase, header_Dixon_out_phase = series_Dixon_out_phase.array('SliceLocation', pixels_first=True)


    array_water_dominant_map = np.zeros(array_Dixon_fat.shape)
    array_water_dominant_map[array_Dixon_fat<0.5*array_Dixon_in_phase] =1

    array_water_calculated = np.zeros_like(array_Dixon_fat)  # Ensure same shape
    array_water_calculated = np.where(array_water_dominant_map == 1, array_Dixon_in_phase + array_Dixon_out_phase, array_water_calculated)
    array_water_calculated = np.where(array_water_dominant_map == 0, array_Dixon_in_phase - array_Dixon_out_phase, array_water_calculated)

    array_fat_calculated = np.zeros_like(array_Dixon_fat)  # Ensure same shape
    array_fat_calculated = np.where(array_water_dominant_map == 0, array_Dixon_in_phase + array_Dixon_out_phase, array_fat_calculated)
    array_fat_calculated = np.where(array_water_dominant_map == 1, array_Dixon_in_phase - array_Dixon_out_phase, array_fat_calculated)

    series = study_Dixon_fat.new_series(SeriesDescription="Dixon_post_contrast_fat_dominant_map")
    series.set_array(array_water_dominant_map, header_Dixon_fat, pixels_first=True)

    series = study_Dixon_fat.new_series(SeriesDescription="Dixon_post_contrast_water_calculated")
    series.set_array(array_water_calculated, header_Dixon_fat, pixels_first=True)

    series = study_Dixon_fat.new_series(SeriesDescription="Dixon_post_contrast_fat_calculated")
    series.set_array(array_fat_calculated, header_Dixon_fat, pixels_first=True)

def Dixon_map_fat_dominant(folder):

    # Find source DICOM data
    series_Dixon_fat, study_Dixon_fat, desc_Dixon_fat                = _map_input   (folder, "Dixon_fat [coreg]")
    series_Dixon_in_phase, study_Dixon_in_phase, desc_Dixon_in_phase = _map_input   (folder, "Dixon_in_phase [coreg]")

    # Load data
    array_Dixon_fat, header_Dixon_fat           = series_Dixon_fat.array('SliceLocation', pixels_first=True)
    array_Dixon_in_phase, header_Dixon_in_phase = series_Dixon_in_phase.array('SliceLocation', pixels_first=True)

    array_water_dominant_map = np.zeros(array_Dixon_fat.shape)
    array_water_dominant_map[array_Dixon_fat<0.5*array_Dixon_in_phase] =1

    series = study_Dixon_fat.new_series(SeriesDescription="Dixon_post_contrast_fat_dominant_map")
    series.set_array(array_water_dominant_map, header_Dixon_fat, pixels_first=True)


def T1(folder):

    # Find source DICOM data
    series, study, desc = _map_input(folder, "T1m_magnitude")

    # Model parameters (Siemens)
    TR = 4.6 # Echo Spacing is msec but not in header -> Set as TR in harmonize
    FA_cat = [-1, 2, -3, 4, -5] # Catalization module confirmed by Siemens (Peter Schmitt): Magn Reson Med 2003 Jan;49(1):151-7. doi: 10.1002/mrm.10337
    N_T1 = 66 # Number of k-space lines (hardcoded from Siemens protocol)
    FA_nom = 12 # Flip angle in degrees (hardcoded from Siemens protocol)

    model = models.t1.Bloch()
    kwargs = {'TR':TR, 'FA_cat':FA_cat, 'N_T1':N_T1, 'FA':FA_nom}
    dims = ['SliceLocation', 'AcquisitionTime']

    # Load data - note each slice can have different TIs
    array, header = series.array(dims, pixels_first=True, first_volume=True)
    TI = series.values('InversionTime', dims=tuple(dims)).astype(np.float32)

    _map(series, array, header, model, TI, study, desc, **kwargs)

def T1_from_T1_T2_mdr(folder):

    # Find source DICOM data
    series, study, desc = _map_input(folder, "T1m_T2m_magnitude")

    list_T1_T2 = series.split_by('SequenceName')

    list_T1_T2[0].SeriesDescription = 'T1m_from_T1m_T2m_mdr_moco'
    list_T1_T2[1].SeriesDescription = 'T2m_from_T1m_T2m_mdr_moco'

    folder.save()

    series = list_T1_T2[0]
    desc = 'T1m_from_T1m_T2m_mdr_moco'
    

    # Model parameters (Siemens)
    TR = 4.6 # Echo Spacing is msec but not in header -> Set as TR in harmonize
    FA_cat = [-1, 2, -3, 4, -5] # Catalization module confirmed by Siemens (Peter Schmitt): Magn Reson Med 2003 Jan;49(1):151-7. doi: 10.1002/mrm.10337
    N_T1 = 66 # Number of k-space lines (hardcoded from Siemens protocol)
    FA_nom = 12 # Flip angle in degrees (hardcoded from Siemens protocol)

    model = models.t1.Bloch()
    kwargs = {'TR':TR, 'FA_cat':FA_cat, 'N_T1':N_T1, 'FA':FA_nom}
    dims = ['SliceLocation', 'AcquisitionTime']

    # Load data - note each slice can have different TIs
    array, header = series.array(dims, pixels_first=True, first_volume=True)
    TI = series.values('InversionTime', dims=tuple(dims)).astype(np.float32)

    _map(series, array, header, model, TI, study, desc, **kwargs)



def T2(folder):

    # Find source DICOM data
    series, study, desc = _map_input(folder, "T2m_magnitude")
    series_T1_map, study_T1_map  = input_series (folder, "T1m_magnitude_mdr_moco_T1_map",export_study)

    # Load data
    model = models.t2.Bloch()
    dims = ['SliceLocation', 'AcquisitionTime']
    array, header = series.array(dims, pixels_first=True, first_volume=True)
    
    array_T1_map, header_T1_map = series_T1_map.array(dims, pixels_first=True)
    array_T1_map = np.squeeze(array_T1_map)
    
    TI = series.values('InversionTime', dims=tuple(dims)).astype(np.float32)
    
    _map(series, array, header, model, TI, study, desc, const = array_T1_map)


def T2_from_T1_T2_mdr(folder):

    # Find source DICOM data
    series_T2, study_T2, desc_T2 = _map_input   (folder, "T2m_from_T1m_T2m_mdr_moco")
    series_T1_map, study_T1_map  = input_series (folder, "T1m_from_T1m_T2m_mdr_moco_T1_map",export_study)

    # Load data
    dims = ['SliceLocation', 'AcquisitionTime']
    array_T2, header_T2 = series_T2.array(dims, pixels_first=True, first_volume=True)
    array_T1_map, header_T1_map = series_T1_map.array('SliceLocation', pixels_first=True)
    
    TE = np.array([0,30,40,50,60,70,80,90,100,110,120])
    Tspoil = 1# Spoil time in ms
    N = 72# Number of k-space lines (hardcoded from Siemens protocol)
    Trec = 463*2# Recovery time in ms (hardcoded from Siemens protocol)
    TR = 4.6# TR in ms (hardcoded from Siemens protocol)
    FA = 12/180*np.pi
    
    T2_S0_map = np.empty(np.shape(array_T2)[0:3])
    T2_map = np.empty(np.shape(array_T2)[0:3])
    FA_Eff_map = np.empty(np.shape(array_T2)[0:3])
    r_squared_map = np.empty(np.shape(array_T2)[0:3])

    for z in range(np.shape(array_T2)[2]):
        for x in range(np.shape(array_T2)[0]):
            print(x)
            for y in range(np.shape(array_T2)[1]):
                T2_signal = np.squeeze(array_T2[x,y,z,:])
                T2_signal[T2_signal<0] = 0

                T1 = array_T1_map[x,y,z]

                popt, pcov = curve_fit(lambda Tprep, M_eq, T2, FA_Eff: signalSequenceT2prep(Tprep, M_eq, T2, T1[0], Tspoil, FA,FA_Eff, TR, N, Trec), xdata = TE, ydata = T2_signal, p0=[np.amax(T2_signal),80,1], bounds=([0, 0, 0],[100000,200,1]), method='trf',maxfev=3000)

                S0 = popt[0]
                T2 = popt[1]
                FA_Eff = popt[2]

                fit=signalSequenceT2prep(TE, popt[0],popt[1],T1,Tspoil,FA,popt[2],TR,N,Trec)
                residuals = T2_signal-np.squeeze(fit)
                ss_res = np.sum(np.nan_to_num(residuals**2))
                ss_tot = np.sum(np.nan_to_num((T2_signal-np.nanmean(T2_signal))**2))

                if ss_tot  == 0:
                    r_squared = 0
                else:
                    r_squared = np.nan_to_num(1 - (ss_res / ss_tot))

                T2_S0_map[x,y,z] = S0
                T2_map[x,y,z] = T2
                FA_Eff_map[x,y,z] = FA_Eff
                r_squared_map[x,y,z] = r_squared


    T2_S0_map_series = series_T2.SeriesDescription + "_T2_" + "S0_Map"
    T2_S0_map_series = study_T2.new_series(SeriesDescription=T2_S0_map_series)
    T2_S0_map_series.set_array(np.squeeze(T2_S0_map),np.squeeze(header_T2[:,0]),pixels_first=True)

    FA_Eff_map_series = series_T2.SeriesDescription + "_T2_" + "FA_Eff_Map"
    FA_Eff_map_series = study_T2.new_series(SeriesDescription=FA_Eff_map_series)
    FA_Eff_map_series.set_array(np.squeeze(FA_Eff_map),np.squeeze(header_T2[:,0]),pixels_first=True)

    T2_map_series = series_T2.SeriesDescription + "_T2_" + "T2_Map"
    T2_map_series = study_T2.new_series(SeriesDescription=T2_map_series)
    T2_map_series.set_array(np.squeeze(T2_map),np.squeeze(header_T2[:,0]),pixels_first=True)

    r_squared_map_series = series_T2.SeriesDescription + "_T2_" + "r_squared_Map"
    r_squared_map_series = study_T2.new_series(SeriesDescription=r_squared_map_series)
    r_squared_map_series.set_array(np.squeeze(r_squared_map),np.squeeze(header_T2[:,0]),pixels_first=True)
                

def T2star(folder):
    
    # Find source DICOM data
    series, study, desc = _map_input(folder, "T2starm_magnitude")

    model = models.T2star.BiExp()
    dims = ['SliceLocation', 'EchoTime']
    array, header = series.array(dims, pixels_first=True, first_volume=True)
    echo_times = series.values('EchoTime', dims=tuple(dims)).astype(np.float32)

    _map(series, array, header, model, echo_times, study, desc)


def MT(folder):
    
    # Find source DICOM data
    series, study, desc = _map_input(folder, "MT")
        
    array, header = series.array(['SliceLocation', 'AcquisitionTime'], pixels_first=True, first_volume=True)

    # Calculate
    array_mtr = 100*(array[...,0]-array[...,1])/array[...,0]
    array_mtr[array[...,0]==0] = 0
    array_mtr[array_mtr>100] = 100
    array_mtr[array_mtr<-100] = -100
    array_avr = np.mean(array, axis=-1)
    
    # Save as DICOM
    mtr = study.new_series(SeriesDescription=desc + '_MTR_map')
    mtr.set_array(array_mtr, header[:,0], pixels_first=True)
    avr = study.new_series(SeriesDescription=desc + '_AVR_map')
    avr.set_array(array_avr, header[:,0], pixels_first=True)
    
    return (mtr,)


def DTI(folder):

    # Find appropriate series
    series, study, desc = _map_input(folder, "DTI")

    # Read data
    model = models.DTI.DiPy()
    array, header = series.array(['SliceLocation', 'InstanceNumber'], pixels_first=True, first_volume=True)
    bvals, bvecs = series.values('DiffusionBValue', 'DiffusionGradientOrientation', dims=('SliceLocation', 'InstanceNumber'))

    # Compute
    series.message('Fitting DTI model..')
    fit, pars = model.fit(array, bvals[0,:], np.stack(bvecs[0,:]), fit_method='NLLS')
    #err = model.error(array, fit)

    # Save as DICOM
    series = study.new_series(SeriesDescription=desc + '_fit')
    series.set_array(fit, header, pixels_first=True)
    #series = study.new_series(SeriesDescription=desc + '_fiterr_map')
    #series.set_array(err, header[:,0], pixels_first=True)
    maps = []
    for i, p in enumerate(model.pars()):
        series = study.new_series(SeriesDescription=desc + '_' + p +'_map')
        series.set_array(pars[...,i], header[:,0], pixels_first=True)
        maps.append(series)

    return maps[0], maps[1]


def IVIM(folder):

    # Find appropriate series
    series, study, desc = _map_input(folder, "IVIM")

    model = models.IVIM.DiPy()
    array, header = series.array(['SliceLocation', 'InstanceNumber'], pixels_first=True, first_volume=True)
    bvals, bvecs = series.values('DiffusionBValue', 'DiffusionGradientOrientation', dims=('SliceLocation', 'InstanceNumber'))

    array_x = array[:, :, :, 0:10]
    array_y = array[:, :, :, 10:20]
    array_z = array[:, :, :, 20:30]

    # Compute geometric mean using all three orthogonal directions
    product_xyz = array_x * array_y * array_z
    cubic_root_product = np.cbrt(product_xyz)

    # Compute
    series.message('Fitting IVIM model..')
    fit, pars = model.fit(cubic_root_product, bvals[0,0:10], np.stack(bvecs[0,0:10]), fit_method='TRR')
    #err = model.error(array, fit)

    # Save as DICOM
    series = study.new_series(SeriesDescription=desc + '_fit')
    series.set_array(fit, header[:,0:10], pixels_first=True)
    #series = study.new_series(SeriesDescription=desc + '_fiterr_map')
    #series.set_array(err, header[:,0], pixels_first=True)



    # Calculate fit slice by slice so progress bar can be shown
    # fit = np.empty(array.shape)
    # par = np.empty(array.shape[:3] + (len(model.pars()),) )
    # for z in range(array.shape[2]):
    #     series.progress(z+1, array.shape[2], 'Fitting IVIM model')
    #     fit[:,:,z,:], par[:,:,z,:] = model.fit(array[:,:,z,:], bvals[0,:10].astype(np.float32), xtol=1e-3, bounds=True, parallel=True) 
    # S0, Df, D, ff = model.derived(par)

    # Save as DICOM
    # fit_series = study.new_series(SeriesDescription=desc + "_fit")
    # fit_series.set_array(fit, header, pixels_first=True)
    # Df_series = study.new_series(SeriesDescription=desc + "_Df_map")
    # Df_series.set_array(Df, header[:,0], pixels_first=True)
    # MD_series = study.new_series(SeriesDescription=desc + "_D_map")
    # MD_series.set_array(MD, header[:,0], pixels_first=True)
    # S0_series = study.new_series(SeriesDescription=desc + "_S0_map")
    # S0_series.set_array(S0, header[:,0], pixels_first=True)
    # ff_series = study.new_series(SeriesDescription=desc + "_ff_map")
    # ff_series.set_array(ff, header[:,0], pixels_first=True)

    # return Df_series, ff_series

    maps = []
    for i, p in enumerate(model.pars()):
        series = study.new_series(SeriesDescription=desc + '_' + p +'_map')
        series.set_array(pars[...,i], header[:,0], pixels_first=True)
        maps.append(series)

    return maps[0], maps[1]


def DCE(folder):

    # Find appropriate series       
    series, study, desc = _map_input(folder, "DCE")

    # Extract data
    time, aif = load_aif(folder)
    array, header = series.array(['SliceLocation', 'AcquisitionTime'], pixels_first=True, first_volume=True)
    
    # Calculate maps
    series.message('Calculating descriptive parameters..')
    MAX, AUC, ATT, SO = dcmri.pixel_descriptives(array, aif, time[1]-time[0], baseline=15)
    series.message('Performing linear fit..')
    fit, par = dcmri.pixel_2cfm_linfit(array, aif, time, baseline=15)
    series.message('Deconvolving..')
    RPF, AVD, MTT = dcmri.pixel_deconvolve(array, aif, time[1]-time[0], baseline=15, regpar=0.1)
    
    # Save maps as DICOM
    fit_series = study.new_series(SeriesDescription=desc + "_fit")
    fit_series.set_array(fit, header, pixels_first=True)

    MAX[MAX<0]=0
    MAX[MAX>10000]=10000
    AUC[AUC<0]=0
    AUC[AUC>10000]=10000
    ATT[ATT<0]=0
    ATT[ATT>10000]=10000
    RPF[RPF<0]=0
    RPF[RPF>10000]=10000
    AVD[AVD<0]=0
    AVD[AVD>10000]=10000
    MTT[MTT<0]=0
    MTT[MTT>10000]=10000

    MAX_series = study.new_series(SeriesDescription=desc + "_MAX_map")
    AUC_series = study.new_series(SeriesDescription=desc + "_AUC_map")
    ATT_series = study.new_series(SeriesDescription=desc + "_ATT_map")
    RPF_series = study.new_series(SeriesDescription=desc + "_RPF_map")
    AVD_series = study.new_series(SeriesDescription=desc + "_AVD_map")
    MTT_series = study.new_series(SeriesDescription=desc + "_MTT_map")
    FP_series = study.new_series(SeriesDescription=desc + "_FP_map")
    TP_series = study.new_series(SeriesDescription=desc + "_TP_map")
    VP_series = study.new_series(SeriesDescription=desc + "_VP_map")
    FT_series = study.new_series(SeriesDescription=desc + "_FT_map")
    TT_series = study.new_series(SeriesDescription=desc + "_TT_map")

    MAX_series.set_array(MAX, header[:,0], pixels_first=True)
    AUC_series.set_array(AUC, header[:,0], pixels_first=True)
    ATT_series.set_array(ATT, header[:,0], pixels_first=True)
    RPF_series.set_array(RPF, header[:,0], pixels_first=True)
    AVD_series.set_array(AVD, header[:,0], pixels_first=True)
    MTT_series.set_array(MTT, header[:,0], pixels_first=True)
    FP_series.set_array(par[...,0], header[:,0], pixels_first=True)
    TP_series.set_array(par[...,1], header[:,0], pixels_first=True)
    VP_series.set_array(par[...,0]*par[...,1]/60, header[:,0], pixels_first=True)
    FT_series.set_array(par[...,2], header[:,0], pixels_first=True)
    TT_series.set_array(par[...,3], header[:,0], pixels_first=True)

    return RPF_series, AVD_series, MTT_series



def _map_input(folder, desc):
    series, study = input_series(folder, desc + '_mdr_moco', export_study)
    if series is not None:
        return series, study, desc + '_mdr_moco'
    series, study = input_series(folder, desc, export_study)
    if series is None:
        raise RuntimeError('Cannot perform mapping: series ' + desc + 'does not exist. ')
    return series, study, desc


def _map(series, array, header, model, pars, study, desc, **kwargs):

    if 'const' in kwargs:
        if kwargs['const'] is not None:
            const = kwargs['const']

    # Calculate fit slice by slice
    fit = np.empty(array.shape)
    par = np.empty(array.shape[:3] + (len(model.pars()),) )
    for z in range(array.shape[2]):

        series.progress(z+1, array.shape[2], 'Fitting model')
        if 'const' in kwargs:
            if kwargs['const'] is not None:
                kwargs['const'] = const[...,z]

        fit[:,:,z,:], par[:,:,z,:] = model.fit(array[:,:,z,:], pars[z,:], xtol=1e-3, bounds=True, **kwargs)

    err = model.error(array, fit)

    # Save results to DICOM database
    series = study.new_series(SeriesDescription=desc + "_" + 'fit')
    series.set_array(fit, header, pixels_first=True)
    series = study.new_series(SeriesDescription=desc+'_err_map')
    series.set_array(err, header[:,0], pixels_first=True)
    for i, p in enumerate(model.pars()):
        series = study.new_series(SeriesDescription=desc+'_' + p + '_map')
        series.set_array(par[...,i], header[:,0], pixels_first=True)

    return (series,)



