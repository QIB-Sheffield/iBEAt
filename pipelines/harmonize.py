import numpy as np
import dbdicom
import utilities.standardize_subject_name as standardize_subject_name

# Note from older code:
# For T1-mapping, Siemens uses the field 'InversionTime' 
# but Philips uses (0x2005, 0x1572). Need to include a 
# harmonization step for Philips.


def mt(database):
    # Merge MT ON and MT OFF sequences and remove the originals

    # Find the series
    mt_on = database.series(SeriesDescription="MT_ON")
    mt_off = database.series(SeriesDescription="MT_OFF")

    # If one is missing, or there are multiple, do not proceed
    if len(mt_on)!=1 or len(mt_off)!=1:
        return
    
    # Merge the two sequences into one
    try:
        study = mt_on[0].parent()
        mt = study.new_series(SeriesDescription='MT')
        dbdicom.merge(mt_on+mt_off, mt)
    except:
        return
    
    # If successfully merged, remove the originals
    mt_on[0].remove()
    mt_off[0].remove()

def t1_t2_merge(database):

    t1 = database.series(SeriesDescription="T1m_magnitude")
    t2 = database.series(SeriesDescription="T2m_magnitude")

    if len(t1)!=1 or len(t2)!=1:
        return
    
    try:
        study = t1[0].parent()
        t1_t2 = study.new_series(SeriesDescription='T1m_T2m_magnitude')
        dbdicom.merge(t1+t2, t1_t2)
    except:
        return


def ivim(database):
    desc = 'IVIM'
    series = database.series(SeriesDescription=desc)[0]

    if series.instance().Manufacturer=='SIEMENS':
        # The b-values and bvectors are incorrectly stored in the header, so must be hardwired.
        # First create an array of b-values and b-vectors ordered by slice location and acquisition time.
        # for each slice, 10 b-values are acquired 3 times, once in each direction (-x, -y, z).
        dims = ('SliceLocation', 'InstanceNumber')
        nslices = 30
        nbvals = 10
        bvals = [0, 10, 20, 30, 50, 80, 100, 200, 300, 600.0]
        bvals = np.array(nslices*[bvals + bvals + bvals])
        bvecs_z = [[-1,0,0]]*nbvals + [[0,-1,0]]*nbvals + [[0,0,1]]*nbvals
        bvecs = np.zeros((nslices,3*nbvals), dtype=object)
        for z in range(nslices):
            for b in range(3*nbvals):
                bvecs[z,b] = bvecs_z[b]
        # Write b-values and b-vectors in the correct DICOM data element
        series.set_values((bvals,bvecs), ('DiffusionBValue', 'DiffusionGradientOrientation'), dims=dims)

def dti(database):
    desc = "DTI"
    series = database.series(SeriesDescription=desc)[0]

    if series.instance().Manufacturer=='SIEMENS':
        # bvalues and b-vectors are correctly stored but in a private data tag
        # Copy the values to the standard DICOM data tag
        bvals, bvecs = series.values((0x19, 0x100c), (0x19, 0x100e)) 
        series.set_values((bvals,bvecs), ('DiffusionBValue', 'DiffusionGradientOrientation'))


def t2(database):
    desc = "T2m_magnitude"
    series = database.series(SeriesDescription=desc)[0]

    if series.instance().Manufacturer=='SIEMENS':
        TE = [[0,30,40,50,60,70,80,90,100,110,120]]*5
        series.set_values(np.array(TE), 'InversionTime',  dims=('SliceLocation', 'InstanceNumber'))


def dce(database):
    desc = "DCE"
    series = database.series(SeriesDescription=desc)[0]

    if series.instance().Manufacturer=='SIEMENS':
        # Split into two separate series, one for the aorta and one for the kidneys
        # Then delete the original
        try:
            split_series = series.split_by('ImageOrientationPatient')
        except:
            # If the series has already been split, there is nothing to be done.
            return
        locs = split_series[0].unique('SliceLocation')
        if len(locs) == 1:
            split_series[0].SeriesDescription = "DCE_aorta_axial_fb"
            split_series[1].SeriesDescription = desc
        else:
            split_series[0].SeriesDescription = desc
            split_series[1].SeriesDescription = "DCE_aorta_axial_fb"
        series.remove()

        	
def pc_left(database):
    desc = [
        'PC_left_delta_phase',
    ]   
    for d in desc:
        series = database.series(SeriesDescription=d) 
        if series == []:
            pass
        series = series[-1] # selected the last if there are multiple
        
        if series.instance().Manufacturer=='SIEMENS':
            venc = 120 # cm/sec # read from DICOM header?
            phase = series.pixel_values(dims='TriggerTime')
            velocity = phase * venc/4096 # cm/sec
            vel = series.copy(SeriesDescription=d[:-11] + 'velocity')
            vel.set_pixel_values(velocity, dims='TriggerTime')

def pc_right(database):
    desc = [
        'PC_right_delta_phase',
    ]   
    for d in desc:
        series = database.series(SeriesDescription=d) 
        if series == []:
            pass
        series = series[-1] # selected the last if there are multiple
        
        if series.instance().Manufacturer=='SIEMENS':
            venc = 120 # cm/sec # read from DICOM header?
            phase = series.pixel_values(dims='TriggerTime')
            velocity = phase * venc/4096 # cm/sec
            vel = series.copy(SeriesDescription=d[:-11] + 'velocity')
            vel.set_pixel_values(velocity, dims='TriggerTime')

def subject_name_internal(database,dataset):

    series_list = database.series()

    if dataset[0] == 7:
        subject_name = database.PatientName
        if len(subject_name) != 7:
            correct_name = database.PatientName[3:7]+ "_" + database.PatientName[7:]
            database.PatientName = correct_name
            database.save()
            return


    subject_name = series_list[0]['PatientID']
    if len(subject_name) != 7 or subject_name[4] != "_":

        correct_name = standardize_subject_name.subject_hifen(subject_name)
        #print(correct_name)
        if correct_name != 0:
            if dataset[0] == 3 and dataset [1] == 3:
                correct_name = correct_name + "_followup"
            elif dataset[0] == 2 and dataset [1] == 0:
                correct_name = correct_name + "_followup"
            database.PatientName = correct_name
            database.save()
            return

        correct_name = standardize_subject_name.subject_underscore(subject_name)
        print(correct_name)
        if correct_name != 0:
            if dataset[0] == 3 and dataset [1] == 3:
                correct_name = correct_name + "_followup"
            elif dataset[0] == 2 and dataset [1] == 0:
                correct_name = correct_name + "_followup"
            database.PatientName = correct_name
            database.save()
            return
        
        correct_name = standardize_subject_name.subject_seven_digitd(subject_name)
        print(correct_name)
        if correct_name != 0:
            if dataset[0] == 3 and dataset [1] == 3:
                correct_name = correct_name + "_followup"
            elif dataset[0] == 2 and dataset [1] == 0:
                correct_name = correct_name + "_followup"
            database.PatientName = correct_name
            database.save()
            return

def subject_name(database):

    series_list = database.series()
    subject_name = series_list[0]['PatientID']
    if len(subject_name) != 7 or subject_name[4] != "_":

        correct_name = standardize_subject_name.subject_hifen(subject_name)
        #print(correct_name)
        if correct_name != 0:
            database.PatientName = correct_name
            database.save()
            return

        correct_name = standardize_subject_name.subject_underscore(subject_name)
        print(correct_name)
        if correct_name != 0:
            database.PatientName = correct_name
            database.save()
            return
        
        correct_name = standardize_subject_name.subject_seven_digitd(subject_name)
        print(correct_name)
        if correct_name != 0:
            database.PatientName = correct_name
            database.save()
            return

def all_series(database):

    pc_left(database)
    pc_right(database)
    t2(database)
    mt(database)
    dti(database)
    ivim(database)
    dce(database)
    


