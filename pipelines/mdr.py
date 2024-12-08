import numpy as np

from dbdicom.pipelines import input_series

import mdreg
import dcmri

import models.DWI
import models.t1
import models.t1_t2
import models.t2
import models.T2star
import models.DTI
import models.IVIM

from pipelines.roi_fit import load_aif

export_study = '1: MDR results'


def T1(folder):

    desc = "T1m_magnitude"
    dims = ['SliceLocation', 'InversionTime']

    series, study = input_series(folder, desc, export_study)
    if series is None:
        raise RuntimeError('Cannot perform MDR on T1: series ' + desc + 'does not exist. ')

    TR = 4.6 # Echo Spacing is msec but not in header -> Set as TR in harmonize
    array, header = series.array(dims, pixels_first=True, first_volume=True)

    fit_image = [
        {
            'func': models.t1.MonoExp().fit,
            'xdata': np.array([hdr['InversionTime'] for hdr in header[z,:]]),
            'TR': TR,
        }   
    for z in range(array.shape[2])]
    
    return _mdr(series, array, header, fit_image, study)


def T2(folder):

    desc = "T2m_magnitude"
    dims = ['SliceLocation', 'InversionTime']

    series, study = input_series(folder, desc, export_study)
    if series is None:
        raise RuntimeError(
            'Cannot perform MDR on T2: series ' + desc + 'does not exist. ')

    array, header = series.array(dims, pixels_first=True, first_volume=True)
    TE = series.values('InversionTime', dims=tuple(dims)).astype(np.float16)

    fit_image = [
        {
            'func': models.t2.MonoExpOffset().fit,
            'xdata': TE[z,:],
        } 
    for z in range(TE.shape[0])]
    
    return _mdr(series, array, header, fit_image, study)


def T1_T2(folder):

    desc_T1_T2 = "T1m_T2m_magnitude"
    dims = ['SliceLocation', 'InversionTime']

    series_T1_T2, study_T1_T2 = input_series(folder, desc_T1_T2, export_study)
    if series_T1_T2 is None:
        raise RuntimeError(
            'Cannot perform MDR on T1: series ' + desc_T1 + 'does not exist. ')

    desc_T1 = "T1m_magnitude"
    series_T1, study_T1 = input_series(folder, desc_T1, export_study)
    if series_T1 is None:
        raise RuntimeError(
            'Cannot perform MDR on T1: series ' + desc_T1 + 'does not exist. ')
    
    desc_T2 = "T2m_magnitude"
    series_T2, study_T2 = input_series(folder, desc_T2, export_study)
    if series_T2 is None:
        raise RuntimeError(
            'Cannot perform MDR on T1: series ' + desc_T2 + 'does not exist. ')

    array_T1, header_T1 = series_T1.array(dims, pixels_first=True, first_volume=True)
    array_T2, header_T2 = series_T2.array(dims, pixels_first=True, first_volume=True)
    
    TE = series_T2.values('InversionTime', dims=tuple(dims)).astype(np.float16)
    TI = series_T1.values('InversionTime', dims=tuple(dims)).astype(np.float16)

    array = np.concatenate((array_T1, array_T2), axis=3)
    header = np.concatenate((header_T1, header_T2), axis=1)
    signal_pars_T1_T2 = np.concatenate((TI,TE), axis=1)

    fit_image = [
        {
            'func': models.t1_t2.MonoExp_T1_T2().fit,
            'xdata': signal_pars_T1_T2[z,:]
        } 
    for z in range(signal_pars_T1_T2.shape[0])]
    
    return _mdr(series_T1_T2, array, header, fit_image, study_T1_T2)


def T2star(folder):

    desc = "T2starm_magnitude"
    dims = ['SliceLocation', 'EchoTime']

    series, study = input_series(folder, desc, export_study)
    if series is None:
        raise RuntimeError('Cannot perform MDR on T2*: series ' + desc + 'does not exist. ')

    array, header = series.array(dims, pixels_first=True, first_volume=True)

    fit_image = [
        {
            'func': models.T2star.MonoExp().fit,
            'xdata': np.array([hdr.EchoTime for hdr in header[z,:]]),
        } 
    for z in range(array.shape[2])]

    return _mdr(series, array, header, fit_image, study)


def MT(folder): 

    desc = 'MT'
    dims = ['SliceLocation', 'AcquisitionTime']

    series, study = input_series(folder, desc, export_study)
    if series is None:
        raise RuntimeError('Cannot perform MDR on MT: series ' + desc + ' does not exist. ')

    array, header = series.array(dims, pixels_first=True, first_volume=True)

    fit_image = None

    return _mdr(series, array, header, fit_image, study, force_2d=False)


def DTI(folder):

    desc = "DTI"
    dims = ['SliceLocation', 'InstanceNumber']

    series, study = input_series(folder, desc, export_study)
    if series is None:
        raise RuntimeError(
            'Cannot perform MDR on DTI: series ' + desc + 'does not exist. ')

    bvals, bvecs = series.values(
        'DiffusionBValue', 'DiffusionGradientOrientation', dims=dims)
    array, header = series.array(
        dims, pixels_first=True, first_volume=True)
    
    fit_image = [
        {
            'func': models.DTI.DiPy().fit,
            'bvals': bvals[z,:],
            'bvecs': np.stack(bvecs[z,:]),
            'fit_method': 'WLS',
        } 
    for z in range(array.shape[2])]

    return _mdr(series, array, header, fit_image, study)


def IVIM(folder, series=None,study=None):

    desc = 'IVIM'
    dims = ['SliceLocation', 'InstanceNumber']

    series, study = input_series(folder, desc, export_study)
    if series is None:
        raise RuntimeError(
            'Cannot perform MDR on IVIM: series ' + desc + 'does not exist. ')

    bvals, bvecs = series.values(
        'DiffusionBValue', 'DiffusionGradientOrientation', dims=dims)
    array, header = series.array(dims, pixels_first=True, first_volume=True)

    fit_image = [
        {
            'func': models.DWI.Lin().fit,
            'bvals': bvals[z,:],
            'bvecs': np.stack(bvecs[z,:]),
        } 
    for z in range(array.shape[2])]

    return _mdr(series, array, header, fit_image, study)


def DCE(folder):

    desc = "DCE"
    dims = ['SliceLocation', 'AcquisitionTime']

    series, study = input_series(folder, desc, export_study)
    if series is None:
        raise RuntimeError('Cannot perform MDR on DCE: not all series are there.')
    
    time, aif = load_aif(folder)
    array, header = series.array(dims, pixels_first=True, first_volume=True)
    
    fit_image = {
        'func': dcmri.pixel_2cfm_linfit,
        'aif': aif,
        'time': time,
        'baseline': 15,
    }

    return _mdr(series, array, header, fit_image, study)


def _mdr(series, array, header, fit_image, study, force_2d=True):

    series.message('Setting up MDR..')
    fit_coreg = {
        'spacing': header[0,0].PixelSpacing,
        'MaximumStepLength': 0.1,
        "FinalGridSpacingInPhysicalUnits": 20,
        'downsample': 2,
    }
    coreg, defo, model_fit, pars = mdreg.fit(array,
        fit_image = fit_image,
        fit_coreg = fit_coreg,
        maxit = 5,
        force_2d = force_2d,
    )

    # Save results as DICOM
    desc = series.SeriesDescription
    fit = study.new_series(SeriesDescription=desc + '_mdr_fit')
    fit.set_array(model_fit, header, pixels_first=True)
    moco = study.new_series(SeriesDescription = desc + '_mdr_moco')
    moco.set_array(coreg, header, pixels_first=True)
    defo = study.new_series(SeriesDescription = desc + '_mdr_defo')
    defo.set_array(mdreg.defo_norm(defo, 'eumip'), header[:,0], pixels_first=True)

    return fit, moco
