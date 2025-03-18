""" 
@author: Joao Periquito 
iBEAt Rename Scrpit
2022
Pulse sequence name standardization for iBEAt MR Protcol
"""
import time
import os

from itertools import chain
import pandas as pd

def check(database):

    results_path = database.path() + '_output'
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    df = pd.DataFrame([
        ['T2w',0,''],
        ['Dixon_out_phase',0,''],           #TE
        ['Dixon_in_phase',0,''],            #TE
        ['Dixon_fat',0,''],
        ['Dixon_water',0,''],
        ['PC_right_delta_magnitude',0,''],  #
        ['PC_right',0,''],
        ['PC_right_delta_phase',0,''],
        ['PC_left_delta_magnitude',0,''],
        ['PC_left',0,''],
        ['PC_left_delta_phase',0,''],
        ['T2starm_pancreas_magnitude',0,''],    #Echo train length
        ['T2starm_pancreas_phase',0,''],
        ['T2starm_pancreas_T2star',0,''],
        ['T1w_magnitude',0,''],
        ['T1w_phase',0,''],
        ['T1m_magnitude',0,''],         #number of TIs
        ['T1m_phase',0,''],
        ['T1m_moco',0,''],
        ['T1m_T1',0,''],
        ['T2m_magnitude',0,''],         #
        ['T2m_phase',0,''],
        ['T2m_moco',0,''],
        ['T2m_T2',0,''],
        ['T2starm_magnitude',0,''], #echo train length
        ['T2starm_phase',0,''],
        ['T2starm_T2star',0,''],
        ['DTI',0,''],               # bvalues
        ['IVIM',0,''],              # b vales
        ['MT_OFF',0,''],            # pulse prepararion
        ['MT_ON',0,''],
        ['ASL_RBF_moco',0,''],      # inversion delay
        ['DCE',0,''],               # time resolution (difference between acqui time)
        ['Dixon_post_contrast_out_phase',0,''],
        ['Dixon_post_contrast_in_phase',0,''],
        ['Dixon_post_contrast_fat',0,''],
        ['Dixon_post_contrast_water',0,'']
        ], columns=['MRI Sequence','Checked','Notes'])

    list_of_series = database.series()
    list_of_series_description = []
    for series in (list_of_series):
        list_of_series_description.append(series['SeriesDescription'])
    
    # Loop through the rows and update the second column based on the condition
    dims = ['AcquisitionTime']
    for index, row in df.iterrows():
        if row['MRI Sequence'] in list_of_series_description:
            df.at[index, 'Checked'] = 1

            if row['MRI Sequence'] == 'Dixon_out_phase':
                series = database.series(SeriesDescription='Dixon_out_phase')
                if series[0]['EchoTime'] == 1.34:
                    continue
                else:
                    df.at[index, 'Notes'] = 'Echo time (1.34ms) = ' + str(series[0]['EchoTime'])
                    df.at[index, 'Checked'] = 2
            
            if row['MRI Sequence'] == 'Dixon_in_phase':
                series = database.series(SeriesDescription='Dixon_in_phase')
                if series[0]['EchoTime'] == 2.57:
                    continue
                else:
                    df.at[index, 'Notes'] = 'Echo time (2.57ms) = ' + str(series[0]['EchoTime'])
                    df.at[index, 'Checked'] = 2

            if row['MRI Sequence'] == 'PC_right_delta_magnitude':
                series = database.series(SeriesDescription='PC_right_delta_magnitude')
                if series[0]['EchoTime'] == 2.74:
                    continue
                else:
                    df.at[index, 'Notes'] = 'Echo time (2.74ms) = ' + str(series[0]['EchoTime'])
                    df.at[index, 'Checked'] = 2        

            if row['MRI Sequence'] == 'PC_left_delta_magnitude':
                series = database.series(SeriesDescription='PC_left_delta_magnitude')
                if series[0]['EchoTime'] == 2.74:
                    continue
                else:
                    df.at[index, 'Notes'] = 'Echo time (2.74ms) = ' + str(series[0]['XXXXX'])
                    df.at[index, 'Checked'] = 2        

            if row['MRI Sequence'] == 'T2starm_pancreas_magnitude':
                series = database.series(SeriesDescription='T2starm_pancreas_magnitude')
                if series[0]['EchoTrainLength'] == 12:
                    df.at[index, 'Notes'] = 'Correct'
                    continue
                else:
                    df.at[index, 'Notes'] = 'Echo train length (12) = ' + str(series[0]['EchoTrainLength'])
                    df.at[index, 'Checked'] = 2

            if row['MRI Sequence'] == 'T1w_magnitude':
                series = database.series(SeriesDescription='T1w_magnitude')
                if series[0]['EchoTime'] == 4.92:
                    df.at[index, 'Notes'] = 'Correct'
                    continue
                else:
                    df.at[index, 'Notes'] = 'Echo time (4.92) = ' + str(series[0]['EchoTime'])
                    df.at[index, 'Checked'] = 2

            if row['MRI Sequence'] == 'T1m_magnitude':
                series = database.series(SeriesDescription='T1m_magnitude')
                InversionTimes = series[0].values('InversionTime', dims=dims)
                if len(InversionTimes) > 28:
                    df.at[index, 'Notes'] = 'Correct'
                    continue
                else:
                    df.at[index, 'Notes'] = 'Number of inversion times (< 28) = ' + str(len(InversionTimes))
                    df.at[index, 'Checked'] = 2

            if row['MRI Sequence'] == 'T2m_magnitude':
                series = database.series(SeriesDescription='T2m_magnitude')
                AcquisitiomTimes = series[0].values('AcquisitionTime', dims=dims)
                if len(AcquisitiomTimes) == 55:
                    df.at[index, 'Notes'] = 'Correct'
                    continue
                else:
                    df.at[index, 'Notes'] = 'Number of scans seems wrong (50) = ' + str(len(AcquisitiomTimes))
                    df.at[index, 'Checked'] = 2

            if row['MRI Sequence'] == 'T2starm_magnitude':
                series = database.series(SeriesDescription='T2starm_magnitude')
                if series[0]['EchoTrainLength'] == 12:
                    df.at[index, 'Notes'] = 'Correct'
                    continue
                else:
                    df.at[index, 'Notes'] = 'Echo train length (12) = ' + str(series[0]['EchoTrainLength'])
                    df.at[index, 'Checked'] = 2

            if row['MRI Sequence'] == 'IVIM':
                series = database.series(SeriesDescription='IVIM')
                shape = series[0].shape(dims=['SliceLocation', 'InstanceNumber'])

                if shape[1] == 30:
                    df.at[index, 'Notes'] = 'Correct'
                    continue
                else:
                    df.at[index, 'Notes'] = 'b-values length (30) = ' + str(shape[1])
                    df.at[index, 'Checked'] = 2

            if row['MRI Sequence'] == 'DTI':
                series = database.series(SeriesDescription='DTI')
                bvecs = series[0].values('DiffusionGradientOrientation', dims=['SliceLocation', 'InstanceNumber'])
                if bvecs.shape[1] == 146:
                    df.at[index, 'Notes'] = 'Correct'
                    continue
                else:
                    df.at[index, 'Notes'] = 'Number of diffusion directions (144) = ' + str(len(bvecs))
                    df.at[index, 'Checked'] = 2               

    def color_rule(val):
        return ['background-color: red' if x == 0 else 'background-color: orange' if x == 2 else 'background-color: green' for x in val]

    iBEAt_column = df.style.apply(color_rule, axis=1, subset=['Checked'])
    iBEAt_column.to_excel(os.path.join(results_path,'iBEAt_checklist.xlsx'), engine='openpyxl', index=False)

    

def Philips_rename(series):
        
    SeqName = series["SeriesDescription"]
    
    if SeqName is None:
        return
    
    if SeqName == 'T1w_abdomen_dixon_cor_bh':
        TE = series.EchoTime

        inphase = series.subseries(EchoTime=TE[0])
        inphase.SeriesDescription = 'Dixon_in_phase'

        outphase = series.subseries(EchoTime=TE[1])
        outphase.SeriesDescription = 'Dixon_out_phase'
        
        return 'Dixon'
    
    if SeqName == 'PC_RenalArtery_Right_EcgTrig_fb_120':

        magnitude = series.subseries(ImageType=['ORIGINAL', 'PRIMARY', 'M_FFE', 'M', 'FFE'])
        magnitude.SeriesDescription = 'PC_right_magnitude'

        m_pca = series.subseries(ImageType=['ORIGINAL', 'PRIMARY', 'M_PCA', 'M', 'PCA'])
        m_pca.SeriesDescription = 'PC_right_mpca'

        phase = series.subseries(ImageType=['ORIGINAL', 'PRIMARY', 'PHASE CONTRAST M', 'P', 'PCA'])
        phase.SeriesDescription = 'PC_right_phase'

        return 'PC_right'
    
    if SeqName == 'PC_RenalArtery_Left_EcgTrig_fb_120':

        magnitude = series.subseries(ImageType=['ORIGINAL', 'PRIMARY', 'M_FFE', 'M', 'FFE'])
        magnitude.SeriesDescription = 'PC_left_magnitude'

        m_pca = series.subseries(ImageType=['ORIGINAL', 'PRIMARY', 'M_PCA', 'M', 'PCA'])
        m_pca.SeriesDescription = 'PC_left_mpca'

        phase = series.subseries(ImageType=['ORIGINAL', 'PRIMARY', 'PHASE CONTRAST M', 'P', 'PCA'])
        phase.SeriesDescription = 'PC_left_phase'

        return 'PC_left'

    if SeqName == 'T1map_kidneys_cor-oblique_mbh fa35':
        
        magnitude = series.subseries(ImageType=['ORIGINAL', 'PRIMARY', 'M_FFE', 'M', 'FFE'])
        magnitude.SeriesDescription = 'T1m_magnitude'

        T1map = series.subseries(ImageType=['ORIGINAL', 'PRIMARY', 'T1 MAP', 'T1', 'UNSPECIFIED'])
        T1map.SeriesDescription = 'T1m_T1'

        return 'T1m'
    
    if SeqName == 'T2star_map_pancreas_tra_mbh':

        magnitude = series.subseries(ImageType=['ORIGINAL', 'PRIMARY', 'M_FFE', 'M', 'FFE'])
        magnitude.SeriesDescription = 'T2starm_pancreas_magnitude'

        phase = series.subseries(ImageType=['ORIGINAL','PRIMARY','PHASE MAP', 'P', 'FFE'])
        phase.SeriesDescription = 'T2starm_pancreas_phase'

        return 'T2starm_pancreas'
    
    if SeqName == 'T2star_map_kidneys_cor-oblique_mbh':

        magnitude = series.subseries(ImageType=['ORIGINAL', 'PRIMARY', 'M_FFE', 'M', 'FFE'])
        magnitude.SeriesDescription = 'T2starm_magnitude'

        phase = series.subseries(ImageType=['ORIGINAL','PRIMARY','PHASE MAP', 'P', 'FFE'])
        phase.SeriesDescription = 'T2starm_phase'

        return 'T2starm'

    if SeqName == 'T2map_mGRASE_kidneys_cor-oblique_mbh_TEn*10':
        
        magnitude = series.subseries(ImageType=['ORIGINAL', 'PRIMARY', 'M_SE', 'M', 'SE'])
        magnitude.SeriesDescription = 'T2m_magnitude'

        T2map = series.subseries(ImageType=['ORIGINAL', 'PRIMARY', 'T2 MAP', 'T2', 'UNSPECIFIED'])
        T2map.SeriesDescription = 'T2m_T2'

        R2map = series.subseries(ImageType=['ORIGINAL', 'PRIMARY', 'R2_UNSPECIFIED', 'R2', 'UNSPECIFIED'])
        R2map.SeriesDescription = 'T2m_R2'

        return 'T2m'
    
    if SeqName == 'DTI TEST 4 NSA1 b100 add':
        return 'DTI'
    
    if SeqName == 'IVIM _kidneys_cor-oblique_fb gradient file needed':
        return 'IVIM'
    
    if SeqName == 'MT_ON_kidneys_cor-oblique_bh off resonance':
        
        magnitude = series.subseries(ImageType=['ORIGINAL', 'PRIMARY', 'M_FFE', 'M', 'FFE'])
        magnitude.SeriesDescription = 'MT_ON'

        phase = series.subseries(ImageType=['ORIGINAL','PRIMARY','PHASE MAP', 'P', 'FFE'])
        phase.SeriesDescription = 'MT_ON_phase'
        
        return 'MT_ON_mag_and_phase'
    
    if SeqName == 'MT_OFF_kidneys_cor-oblique_bh':
        
        magnitude = series.subseries(ImageType=['ORIGINAL', 'PRIMARY', 'M_FFE', 'M', 'FFE'])
        magnitude.SeriesDescription = 'MT_OFF'

        phase = series.subseries(ImageType=['ORIGINAL','PRIMARY','PHASE MAP', 'P', 'FFE'])
        phase.SeriesDescription = 'MT_OFF_phase'
        
        return 'MT_OFF_mag_and_phase'
    
    if SeqName == 'DCE_kidneys_cor-oblique_fb_wet_pulse':
        return 'DCE'
    
    if SeqName == 'T1w_abdomen_post_contrast_dixon_cor_bh':
        TE = series.EchoTime

        inphase = series.subseries(EchoTime=TE[0])
        inphase.SeriesDescription = 'Dixon_post_contrast_in_phase'

        outphase = series.subseries(EchoTime=TE[1])
        outphase.SeriesDescription = 'Dixon_post_contrast_out_phase'
        
        return 'Dixon_post_contrast'

            

def Siemens_rename(series): 
    """
    The sequence names in Leeds have been removed by the anonymisation
    procedure and must be recovered from other attributes
    """
    SeqName = series["SequenceName"]

    if SeqName is None:
        return
    
    if SeqName == '*tfi2d1_115':
        return

    if SeqName == '*tfi2d1_192':
        if series["FlipAngle"] > 30:
            return 'localizer_bh_fix'
        else: 
            return 'localizer_bh_ISO'

    if SeqName == '*h2d1_320':
        return 'T2w'

    if SeqName == '*fl3d2':
        sequence = 'Dixon'
        imType = series["ImageType"]
        if imType[3] == 'OUT_PHASE' or imType[4] == 'OUT_PHASE': 
            return sequence + '_out_phase'
        if imType[3] == 'IN_PHASE'  or imType[4] == 'IN_PHASE': 
            return sequence + '_in_phase'
        if imType[3] == 'FAT'       or imType[4] == 'FAT': 
            return sequence + '_fat'
        if imType[3] == 'WATER'     or imType[4] == 'WATER': 
            return sequence + '_water'

    if SeqName == '*fl2d1r4': 
        if series["ImagePositionPatient"][0] < 0: #(smallest left, )
            return 'PC_right'
        else: 
            return 'PC_left'

    if SeqName == '*fl2d1_v120in':
        imType = series["ImageType"]
        if series["ImagePositionPatient"][0] < 0:
            sequence = 'PC_right'
        else:
            sequence = 'PC_left'
        if imType[2] == 'P': 
            return sequence + '_delta_phase'
        if imType[2] == 'MAG': 
            return sequence + '_delta_magnitude'
        
    if SeqName == '*fl2d12':
        imType = series["ImageType"]
        if series["InPlanePhaseEncodingDirection"] == 'COL':
            sequence = 'T2starm_pancreas'
        else:
            sequence = 'T2starm'
        if imType[2] == 'M': 
            return sequence + '_magnitude'
        if imType[2] == 'P': 
            return sequence + '_phase'
        if imType[2] == 'T2_STAR MAP': 
            return sequence + '_T2star'

    if SeqName == '*fl2d1':
        imType = series["ImageType"]
        sequence = 'T1w'
        if imType[2] == 'M': 
            return sequence + '_magnitude'
        if imType[2] == 'P': 
            return sequence + '_phase'

    if SeqName == '*tfl2d1r106': 
        imType = series["ImageType"]
        sequence = 'T1m'
        res = list(chain.from_iterable(i if isinstance(i, list) else [i] for i in imType))
        if res[2] == 'T1 MAP': 
            return sequence + '_T1'
        if res[3] == 'MOCO': 
            return sequence + '_moco'
        if res[2] == 'M': 
            return sequence + '_magnitude'
        if res[2] == 'P': 
            return sequence + '_phase'

    if SeqName == '*tfl2d1r96':
        imType = series["ImageType"]
        sequence = 'T2m'
        if imType[-1] == 'T2': 
            return sequence + '_T2'
        if imType[-1] == 'MOCO': 
            return sequence + '_moco'
        if imType[2] == 'M': 
            return sequence + '_magnitude'
        if imType[2] == 'P': 
            return sequence + '_phase'

    if SeqName[:5] == '*ep_b' or SeqName[0][:5] == '*ep_b':
        if len(series.files()) < 1000:
            return 'IVIM'
        else:
            return 'DTI'

    if SeqName == '*fl3d1':
        if series["ScanOptions"] == 'PFP': 
            return 'MT_OFF'
        else:
            return 'MT_ON'

    if SeqName == '*tfi2d1_154': 
        return 'ASL_planning'
    
    if SeqName == 'tgse3d1_512': 
        return 'ASL'
    
    if SeqName == 'WIP_tgse3d1_512': 
        return 'ASL'
    
    if SeqName == '*tfl2d1_16': 
        return 'DCE'
    
    if SeqName == 'RAVE3d1': 
        return 'RAVE_kidneys_fb'
    
    if SeqName == '*fl3d2': 
        return 'Dixon'

def GE_rename(series): 
    """
    The sequence names in Leeds have been removed by the anonymisation
    procedure and must be recovered from other attributes
    """
    SeqName = series["SeriesDescription"]
    print(SeqName)

    if SeqName == 'InPhase: T1_abdomen_dixon_cor_bh':
        return 'Dixon_in_phase'
    if SeqName == 'OutPhase: T1_abdomen_dixon_cor_bh':
        return 'Dixon_out_phase'
    if SeqName == 'WATER: T1_abdomen_dixon_cor_bh':
        return 'Dixon_water'
    if SeqName == 'FAT: T1_abdomen_dixon_cor_bh':
        return 'Dixon_fat'
    if SeqName == 'InPhase: T1_abdomen_post_contrast_dixon_cor_bh':
        return 'Dixon_post_contrast_in_phase'
    if SeqName == 'OutPhase: T1_abdomen_post_contrast_dixon_cor_bh':
        return 'Dixon_post_contrast_out_phase'
    if SeqName == 'WATER: T1_abdomen_post_contrast_dixon_cor_bh':
        return 'Dixon_post_contrast_water'
    if SeqName == 'FAT: T1_abdomen_post_contrast_dixon_cor_bh':
        return 'Dixon_post_contrast_fat'

def all_series(folder):

    DCE_count = 0
    ASL_count = 0
    all_series = folder.series()
    Manufacturer = all_series[0]['Manufacturer']

    for i, series in enumerate(all_series):
        folder.progress(i+1, len(all_series), 'Renaming...')

        if Manufacturer == 'SIEMENS':
            imDescription = Siemens_rename(series)
            if imDescription is None:
                continue
            series.SeriesDescription = imDescription

            if imDescription == 'DCE':
                DCE_count = 1
            if DCE_count == 1 and imDescription[0:5] == 'Dixon':
                series.SeriesDescription = imDescription.split('_')[0] + '_post_contrast' + imDescription.split('Dixon')[-1]
            elif imDescription == 'ASL' and ASL_count == 0:
                series.SeriesDescription = imDescription + '_M0_moco'
                ASL_count = 1
            elif imDescription == 'ASL' and ASL_count == 1:
                series.SeriesDescription = imDescription + '_PW_moco'
                ASL_count = 2
            elif imDescription == 'ASL' and ASL_count == 2:
                series.SeriesDescription = imDescription + '_RBF_moco'
                ASL_count = 3
            elif imDescription == 'ASL' and ASL_count == 3:
                series.SeriesDescription = imDescription + '_control_moco'
                ASL_count = 4
            elif imDescription == 'ASL' and ASL_count == 4:
                series.SeriesDescription = imDescription + '_label0_moco'
                ASL_count = 0
        elif Manufacturer == 'Philips':
            imDescription = Philips_rename(series)
            if imDescription is None:
                continue
            series.SeriesDescription = imDescription

        elif Manufacturer == 'GE MEDICAL SYSTEMS':
            imDescription = GE_rename(series)
            if imDescription is None:
                continue
            series.SeriesDescription = imDescription


