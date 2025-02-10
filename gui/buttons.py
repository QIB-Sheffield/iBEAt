import datetime
from wezel.gui import Action, Menu
from wezel.displays import TableDisplay, MatplotLibDisplay
from wezel.plugins.pyvista import SurfaceDisplay


from pipelines import (
    fetch_AI_model,
    fetch_Drive_mask,
    rename, 
    harmonize, 
    fetch_AI_model,
    fetch_Drive_mask,
    segment, 
    measure, 
    mdr, 
    export, 
    mapping,
    align,
    roi_fit,
)
import utilities.upload as upload


# Needs to be in a central location
#weights = 'C:\\Users\\steve\\Dropbox\\Data\\dl_models\\UNETR_kidneys_v1.pth'


def _never(app):
    return False

def _if_a_database_is_open(app): 
    return app.database() is not None



## HARMONIZATION



def rename_all_series(app):
    folder = app.database()
    rename.all_series(folder)
    rename.check(folder)
    app.refresh()

def harmonize_pc(app):
    folder = app.database()
    harmonize.pc(folder)
    app.refresh()

def harmonize_t2(app):
    folder = app.database()
    harmonize.t2(folder)
    app.refresh()

def harmonize_mt(app):
    folder = app.database()
    harmonize.mt(folder)
    app.refresh()

def harmonize_dti(app):
    folder = app.database()
    harmonize.dti(folder)
    app.refresh()

def harmonize_ivim(app):
    folder = app.database()
    harmonize.ivim(folder)
    app.refresh()

def harmonize_dce(app):
    folder = app.database()
    harmonize.dce(folder)
    app.refresh()

def harmonize_all(app):
    harmonize_pc(app) 
    harmonize_t2(app)
    harmonize_mt(app)
    harmonize_dti(app)
    harmonize_ivim(app)
    harmonize_dce(app)

action_rename = Action("Rename series..", on_clicked=rename_all_series, is_clickable=_if_a_database_is_open)
action_harmonize_pc = Action("Harmonize PC..", on_clicked=harmonize_pc, is_clickable=_if_a_database_is_open)
action_harmonize_t2 = Action("Harmonize T2..", on_clicked=harmonize_t2, is_clickable=_if_a_database_is_open)
action_harmonize_mt = Action("Harmonize MT..", on_clicked=harmonize_mt, is_clickable=_if_a_database_is_open)
action_harmonize_dti = Action("Harmonize DTI..", on_clicked=harmonize_dti, is_clickable=_if_a_database_is_open)
action_harmonize_ivim = Action("Harmonize IVIM..", on_clicked=harmonize_ivim, is_clickable=_if_a_database_is_open)
action_harmonize_dce = Action("Harmonize DCE..", on_clicked=harmonize_dce, is_clickable=_if_a_database_is_open)
action_harmonize_all = Action("Harmonize all..", on_clicked=harmonize_all, is_clickable=_if_a_database_is_open)


menu_harmonize = Menu('Harmonize..')
menu_harmonize.add(action_harmonize_pc)
menu_harmonize.add(action_harmonize_t2)
menu_harmonize.add(action_harmonize_mt)
menu_harmonize.add(action_harmonize_dti)
menu_harmonize.add(action_harmonize_ivim)
menu_harmonize.add(action_harmonize_dce)
menu_harmonize.add_separator()
menu_harmonize.add(action_harmonize_all)


## SEGMENTATION


def fetch_dl_models(app):
    folder = app.database()
    fetch_AI_model.dl_models(folder)
    app.refresh()

def fetch_kidney_masks(app):
    folder = app.database()
    fetch_Drive_mask.kidney_masks(folder)
    app.refresh()

def segment_kidneys(app):
    database = app.database()
    kidneys = segment.kidneys(database)
    for kidney in kidneys:
        app.addWidget(SurfaceDisplay(kidney), title=kidney.label())
    app.refresh()

def segment_renal_sinus_fat(app):
    database = app.database()
    sinus_fats = segment.renal_sinus_fat(database)
    if sinus_fats is not None:
        for sinus_fat in sinus_fats:
            app.addWidget(SurfaceDisplay(sinus_fat), title=sinus_fat.label())
    app.refresh()

def segment_aorta_on_dce(app):
    database = app.database()
    aorta_mask = segment.aorta_on_dce(database)
    app.display(aorta_mask)
    app.refresh()

def segment_renal_artery(app):
    database = app.database()
    ra_masks = segment.renal_artery(database)
    for mask in ra_masks:
        app.display(mask)
    app.refresh()

def whole_kidney_canvas(app):
    folder = app.database()
    clusters = segment.compute_whole_kidney_canvas(folder)
    if clusters is not None:
        app.display(clusters)
    app.refresh()

def export_segmentations(app):
    database = app.database()
    export.kidney_masks_as_dicom(database)
    export.kidney_masks_as_png(database)
    export.aif_as_png(database)
    app.refresh()

def all_segmentation_steps(app):
    answer = app.dialog.question('This is going to take a while. Do you want to continue?')
    if 'Yes' != answer:
        return
    fetch_dl_models(app)
    fetch_kidney_masks(app)
    segment_kidneys(app)
    segment_renal_sinus_fat(app)
    segment_aorta_on_dce(app)
    segment_renal_artery(app)
    whole_kidney_canvas(app)
    export_segmentations(app)


action_all_segmentation_steps = Action("All above segmentation steps..", on_clicked=all_segmentation_steps, is_clickable=_if_a_database_is_open)
action_fetch_dl_models = Action("Fetch deep learning models..", on_clicked=fetch_dl_models, is_clickable=_if_a_database_is_open)
action_fetch_kidney_masks = Action("Fetch kidney masks..", on_clicked=fetch_kidney_masks, is_clickable=_never)
action_segment_kidneys = Action("Auto-segment kidneys...", on_clicked=segment_kidneys, is_clickable=_if_a_database_is_open)
action_renal_sinus_fat = Action("Segment renal sinus fat..", on_clicked=segment_renal_sinus_fat, is_clickable=_if_a_database_is_open)
action_aorta_on_dce = Action("Segment aorta on DCE..", on_clicked=segment_aorta_on_dce, is_clickable=_if_a_database_is_open)
action_renal_artery = Action("Segment renal artery on PC..", on_clicked=segment_renal_artery, is_clickable=_if_a_database_is_open)
action_whole_kidney_canvas = Action("Calculate segmentation canvas..", on_clicked=whole_kidney_canvas, is_clickable=_if_a_database_is_open)
action_export_kidney_segmentations = Action("Export segmentations..", on_clicked=export_segmentations, is_clickable=_if_a_database_is_open)

menu_segment = Menu('Segment..')
menu_segment.add(action_fetch_dl_models)
menu_segment.add(action_fetch_kidney_masks)
menu_segment.add(action_segment_kidneys)
menu_segment.add(action_renal_sinus_fat)
menu_segment.add(action_aorta_on_dce)
menu_segment.add(action_renal_artery)
menu_segment.add_separator()
menu_segment.add(action_whole_kidney_canvas)
menu_segment.add_separator()
menu_segment.add(action_export_kidney_segmentations)
menu_segment.add_separator()
menu_segment.add(action_all_segmentation_steps)





## MODEL-DRIVEN MOTION CORRECTION




def mdreg_t1(app):
    database = app.database()
    results = mdr.T1(database)
    for series in results:
        app.display(series)
    app.refresh()

def mdreg_t2star(app):
    database = app.database()
    results = mdr.T2star(database)
    for series in results:
        app.display(series)
    app.refresh()

def mdreg_t2(app):
    database = app.database()
    results = mdr.T2(database)
    for series in results:
        app.display(series)
    app.refresh()

def mdreg_dti(app):
    database = app.database()
    results = mdr.DTI(database)
    for series in results:
        app.display(series)
    app.refresh()

def mdreg_ivim(app):
    database = app.database()
    results = mdr.IVIM(database)
    for series in results:
        app.display(series)
    app.refresh()

def mdreg_mt(app):
    database = app.database()
    results = mdr.MT(database)
    for series in results:
        app.display(series)
    app.refresh()

def mdreg_dce(app):
    database = app.database()
    results = mdr.DCE(database)
    for series in results:
        app.display(series)
    app.refresh()

def mdreg_export(app):
    database = app.database()
    export.mdreg(database)
    app.refresh()

def all_mdreg(app):
    answer = app.dialog.question('This is going to take a while. Do you want to continue?')
    if 'Yes' != answer:
        return
    mdreg_t1(app)
    mdreg_t2(app)
    mdreg_t2star(app)
    mdreg_mt(app)
    mdreg_dti(app)
    mdreg_ivim(app)
    mdreg_dce(app)
    mdreg_export(app)


action_all_mdreg = Action("All motion corrections..", on_clicked=all_mdreg, is_clickable=_if_a_database_is_open)
action_mdreg_t1 = Action("T1 motion correction..", on_clicked=mdreg_t1, is_clickable=_if_a_database_is_open)
action_mdreg_t2 = Action("T2 motion correction..", on_clicked=mdreg_t2, is_clickable=_if_a_database_is_open)
action_mdreg_t2star = Action("T2* motion correction..", on_clicked=mdreg_t2star, is_clickable=_if_a_database_is_open)
action_mdreg_mt = Action("MT motion correction..", on_clicked=mdreg_mt, is_clickable=_if_a_database_is_open)
action_mdreg_dti = Action("DTI motion correction..", on_clicked=mdreg_dti, is_clickable=_if_a_database_is_open)
action_mdreg_ivim = Action("IVIM motion correction..", on_clicked=mdreg_ivim, is_clickable=_if_a_database_is_open)
action_mdreg_dce = Action("DCE motion correction..", on_clicked=mdreg_dce, is_clickable=_if_a_database_is_open)
action_mdreg_export = Action("Exporting motion corrections as gif..", on_clicked=mdreg_export, is_clickable=_if_a_database_is_open)


menu_mdreg = Menu('Motion correction..')
menu_mdreg.add(action_mdreg_t1)
menu_mdreg.add(action_mdreg_t2)
menu_mdreg.add(action_mdreg_t2star)
menu_mdreg.add(action_mdreg_mt)
menu_mdreg.add(action_mdreg_dti)
menu_mdreg.add(action_mdreg_ivim)
menu_mdreg.add(action_mdreg_dce)
menu_mdreg.add_separator()
menu_mdreg.add(action_mdreg_export)
menu_mdreg.add_separator()
menu_mdreg.add(action_all_mdreg)



## MAPPING


def map_T1(app):
    database = app.database()
    results = mapping.T1(database)
    for series in results:
        app.display(series)
    app.refresh()

def map_T2(app):
    database = app.database()
    results = mapping.T2(database)
    for series in results:
        app.display(series)
    app.refresh()

def map_T2star(app):
    database = app.database()
    results = mapping.T2star(database)
    for series in results:
        app.display(series)
    app.refresh()

def map_MT(app):
    database = app.database()
    MTR = mapping.MT(database)
    app.display(MTR)
    app.refresh()

def map_DTI(app):
    database = app.database()
    results = mapping.DTI(database)
    for series in results:
        app.display(series)
    app.refresh()

def map_IVIM(app):
    database = app.database()
    results = mapping.IVIM(database)
    for series in results:
        app.display(series)
    app.refresh()

def map_DCE(app):
    database = app.database()
    results = mapping.DCE(database)
    for series in results:
        app.display(series)
    app.refresh()

def map_export(app):
    database = app.database()
    export.mapping(database)
    app.refresh()

def map_all(app):
    answer = app.dialog.question('This is going to take a while. Do you want to continue?')
    if 'Yes' != answer:
        return
    map_T1(app)
    map_T2(app)
    map_T2star(app)
    map_MT(app)
    map_DTI(app)
    map_IVIM(app)
    map_DCE(app)
    map_export(app)

action_map_all = Action("Map all parameters..", on_clicked=map_all, is_clickable=_if_a_database_is_open)
action_map_T1 = Action("T1 mapping..", on_clicked=map_T1, is_clickable=_if_a_database_is_open)
action_map_T2 = Action("T2 mapping..", on_clicked=map_T2, is_clickable=_if_a_database_is_open)
action_map_T2star = Action("T2* mapping..", on_clicked=map_T2star, is_clickable=_if_a_database_is_open)
action_map_MT = Action("MTR mapping..", on_clicked=map_MT, is_clickable=_if_a_database_is_open)
action_map_DTI = Action("DTI mapping..", on_clicked=map_DTI, is_clickable=_if_a_database_is_open)
action_map_IVIM = Action("IVIM mapping..", on_clicked=map_IVIM, is_clickable=_if_a_database_is_open)
action_map_DCE = Action("DCE mapping..", on_clicked=map_DCE, is_clickable=_if_a_database_is_open)
action_map_export = Action("Exporting maps as gif..", on_clicked=map_export, is_clickable=_if_a_database_is_open)

menu_map = Menu('Parameter mapping..')
menu_map.add(action_map_T1)
menu_map.add(action_map_T2)
menu_map.add(action_map_T2star)
menu_map.add(action_map_MT)
menu_map.add(action_map_DTI)
menu_map.add(action_map_IVIM)
menu_map.add(action_map_DCE)
menu_map.add_separator()
menu_map.add(action_map_export)
menu_map.add_separator()
menu_map.add(action_map_all)


# ALIGNMENT


def align_dixon(app):
    database = app.database()
    results = align.dixon(database)
    for series in results:
        app.display(series)
    app.refresh()

def align_T1(app):
    database = app.database()
    results = align.t1(database)
    for series in results:
        app.display(series)
    app.refresh()

def align_T2(app):
    database = app.database()
    results = align.t2(database)
    for series in results:
        app.display(series)
    app.refresh()

def align_T2star(app):
    database = app.database()
    results = align.t2star(database)
    for series in results:
        app.display(series)
    app.refresh()

def align_MT(app):
    database = app.database()
    results = align.mt(database)
    for series in results:
        app.display(series)
    app.refresh()

def align_IVIM(app):
    database = app.database()
    results = align.ivim(database)
    for series in results:
        app.display(series)
    app.refresh()

def align_DTI(app):
    database = app.database()
    results = align.dti(database)
    for series in results:
        app.display(series)
    app.refresh()

def align_DCE(app):
    database = app.database()
    results = align.dce(database)
    for series in results:
        app.display(series)
    app.refresh()

def align_ASL(app):
    database = app.database()
    results = align.asl(database)
    for series in results:
        app.display(series)
    app.refresh()

def export_alignment(app):
    database = app.database()
    export.alignment(database)
    app.refresh()

def align_all(app):
    answer = app.dialog.question('This is going to take a while. Do you want to continue?')
    if 'Yes' != answer:
        return
    align_dixon(app)
    align_T1(app)
    align_T2(app)
    align_T2star(app)
    align_MT(app)
    align_DTI(app)
    align_IVIM(app)
    align_DCE(app)
    align_ASL(app)
    export_alignment(app)

action_align_all = Action("Align all sequences..", on_clicked=align_all, is_clickable=_if_a_database_is_open)
action_align_dixon = Action("Align DIXON..", on_clicked=align_dixon, is_clickable=_if_a_database_is_open)
action_align_T1 = Action("Align T1..", on_clicked=align_T1, is_clickable=_if_a_database_is_open)
action_align_T2 = Action("Align T2..", on_clicked=align_T2, is_clickable=_if_a_database_is_open)
action_align_T2star = Action("Align T2*..", on_clicked=align_T2star, is_clickable=_if_a_database_is_open)
action_align_MT = Action("Align MTR..", on_clicked=align_MT, is_clickable=_if_a_database_is_open)
action_align_DTI = Action("Align DTI..", on_clicked=align_DTI, is_clickable=_if_a_database_is_open)
action_align_IVIM = Action("Align IVIM..", on_clicked=align_IVIM, is_clickable=_if_a_database_is_open)
action_align_DCE = Action("Align DCE..", on_clicked=align_DCE, is_clickable=_if_a_database_is_open)
action_align_ASL = Action("Align ASL..", on_clicked=align_ASL, is_clickable=_if_a_database_is_open)
action_align_export = Action("Exporting alignments as gif..", on_clicked=export_alignment, is_clickable=_if_a_database_is_open)

menu_align = Menu('Align scans..')
menu_align.add(action_align_dixon)
menu_align.add(action_align_T1)
menu_align.add(action_align_T2)
menu_align.add(action_align_T2star)
menu_align.add(action_align_MT)
menu_align.add(action_align_DTI)
menu_align.add(action_align_IVIM)
menu_align.add(action_align_DCE)
menu_align.add(action_align_ASL)
menu_align.add_separator()
menu_align.add(action_align_export)
menu_align.add_separator()
menu_align.add(action_align_all)





# MEASURE



def measure_kidney_volumetrics(app):
    database = app.database()
    features = measure.kidney_volumetrics(database)
    app.addWidget(TableDisplay(features), 'Kidney - Volume features')
    app.refresh()

def measure_sinus_fat_volumetrics(app):
    database = app.database()
    features = measure.sinus_fat_volumetrics(database)
    app.addWidget(TableDisplay(features), 'Renal sinus fat - Volume features')
    app.refresh()

def fill_gaps(app):
    database = app.database()
    results = align.fill_gaps(database)
    for series in results:
        app.display(series)
    app.refresh()

def cortex_medulla(app):
    folder = app.database()
    clusters = segment.cortex_medulla(folder)
    if clusters is not None:
        app.display(clusters)
    app.refresh()

def measure_t1_maps(app):
    database = app.database()
    features = measure.t1_maps(database)
    app.addWidget(TableDisplay(features), 'T1-mapping')
    app.refresh()

def measure_t2_maps(app):
    database = app.database()
    features = measure.t2_maps(database)
    app.addWidget(TableDisplay(features), 'T2-mapping')
    app.refresh()

def measure_t2star_maps(app):
    database = app.database()
    features = measure.tstar_maps(database)
    app.addWidget(TableDisplay(features), 'T2*-mapping')
    app.refresh()

def measure_mt_maps(app):
    database = app.database()
    features = measure.mt_maps(database)
    app.addWidget(TableDisplay(features), 'MT-mapping')
    app.refresh()

def measure_ivim_maps(app):
    database = app.database()
    features = measure.ivim_maps(database)
    app.addWidget(TableDisplay(features), 'IVIM-mapping')
    app.refresh()

def measure_dti_maps(app):
    database = app.database()
    features = measure.dti_maps(database)
    app.addWidget(TableDisplay(features), 'DTI-mapping')
    app.refresh()

def measure_dce_maps(app):
    database = app.database()
    features = measure.dce_maps(database)
    app.addWidget(TableDisplay(features), 'DCE-mapping')
    app.refresh()

def measure_asl_maps(app):
    database = app.database()
    features = measure.asl_maps(database)
    app.addWidget(TableDisplay(features), 'ASL-mapping')
    app.refresh()

def measure_all(app):
    answer = app.dialog.question('This is going to take a while. Do you want to continue?')
    if 'Yes' != answer:
        return
    measure_kidney_volumetrics(app)
    measure_sinus_fat_volumetrics(app)
    fill_gaps(app)
    cortex_medulla(app)
    measure_t1_maps(app)
    measure_t2_maps(app)
    measure_t2star_maps(app)
    measure_mt_maps(app)
    measure_ivim_maps(app)
    measure_dti_maps(app)
    measure_dce_maps(app)
    measure_asl_maps(app)

action_all_measurements = Action("All measurements..", on_clicked=measure_all, is_clickable=_if_a_database_is_open)
action_measure_kidney_volumetrics = Action("Measure kidney volumetrics..", on_clicked=measure_kidney_volumetrics, is_clickable=_if_a_database_is_open)
action_measure_sinus_fat_volumetrics = Action("Measure sinus fat volumetrics..", on_clicked=measure_sinus_fat_volumetrics, is_clickable=_if_a_database_is_open)
action_fill_gaps = Action("Fill slice gaps..", on_clicked=fill_gaps, is_clickable=_if_a_database_is_open)
action_cortex_medulla = Action("Cortex-medulla separation..", on_clicked=cortex_medulla, is_clickable=_if_a_database_is_open)
action_t1_maps = Action("Save T1 pixel values..", on_clicked=measure_t1_maps, is_clickable=_if_a_database_is_open)
action_t2_maps = Action("Save T2 parameters..", on_clicked=measure_t2_maps, is_clickable=_if_a_database_is_open)
action_t2star_maps = Action("Save T2* parameters..", on_clicked=measure_t2star_maps, is_clickable=_if_a_database_is_open)
action_mt_maps = Action("Save MT parameters..", on_clicked=measure_mt_maps, is_clickable=_if_a_database_is_open)
action_ivim_maps = Action("Save IVIM parameters..", on_clicked=measure_ivim_maps, is_clickable=_if_a_database_is_open)
action_dti_maps = Action("Save DTI parameters..", on_clicked=measure_dti_maps, is_clickable=_if_a_database_is_open)
action_dce_maps = Action("Save DCE parameters..", on_clicked=measure_dce_maps, is_clickable=_if_a_database_is_open)
action_asl_maps = Action("Save ASL parameters..", on_clicked=measure_asl_maps, is_clickable=_if_a_database_is_open)

menu_measure = Menu('Pixel values..')
menu_measure.add(action_measure_kidney_volumetrics)
menu_measure.add(action_measure_sinus_fat_volumetrics)
menu_measure.add_separator()
menu_measure.add(action_fill_gaps)
menu_measure.add(action_cortex_medulla)
menu_measure.add_separator()
menu_measure.add(action_t1_maps)
menu_measure.add(action_t2_maps)
menu_measure.add(action_t2star_maps)
menu_measure.add(action_mt_maps)
menu_measure.add(action_ivim_maps)
menu_measure.add(action_dti_maps)
menu_measure.add(action_dce_maps)
menu_measure.add(action_asl_maps)
menu_measure.add_separator()
menu_measure.add(action_all_measurements)


# ROI FITS


def roi_fit_T1(app):
    database = app.database()
    figs = roi_fit.T1(database)
    for fig in figs:
        app.addWidget(MatplotLibDisplay(fig), 'T1 ROI analysis')
    app.refresh()

def roi_fit_T2(app):
    database = app.database()
    figs = roi_fit.T2(database)
    for fig in figs:
        app.addWidget(MatplotLibDisplay(fig), 'T2 ROI analysis')
    app.refresh()

def roi_fit_T2star(app):
    database = app.database()
    figs = roi_fit.T2star(database)
    for fig in figs:
        app.addWidget(MatplotLibDisplay(fig), 'T2* ROI analysis')
    app.refresh()

def roi_fit_PC(app):
    database = app.database()
    figs = roi_fit.PC(database)
    for fig in figs:
        app.addWidget(MatplotLibDisplay(fig), 'PC analysis')
    app.refresh()

def roi_fit_DCE(app):
    database = app.database()
    figs = roi_fit.dce(database)
    for fig in figs:
        app.addWidget(MatplotLibDisplay(fig), 'DCE ROI analysis')
    app.refresh()

def roi_fit_DCE_cm(app):
    database = app.database()
    figs = roi_fit.dce_cm(database)
    for fig in figs:
        app.addWidget(MatplotLibDisplay(fig), 'DCE ROI analysis (Cortex-medulla)')
    app.refresh()


def roi_fit_all(app):
    roi_fit_T1(app)
    roi_fit_T2(app)
    roi_fit_T2star(app)
    roi_fit_PC(app)
    roi_fit_DCE(app)
    roi_fit_DCE_cm(app)

action_roi_fit_all = Action("ALL ROI fits..", on_clicked=roi_fit_all, is_clickable=_if_a_database_is_open)
action_roi_fit_t1 = Action("ROI fit T1..", on_clicked=roi_fit_T1, is_clickable=_if_a_database_is_open)
action_roi_fit_t2 = Action("ROI fit T2..", on_clicked=roi_fit_T2, is_clickable=_if_a_database_is_open)
action_roi_fit_t2star = Action("ROI fit T2*..", on_clicked=roi_fit_T2star, is_clickable=_if_a_database_is_open)
action_roi_fit_pc = Action("ROI fit PC..", on_clicked=roi_fit_PC, is_clickable=_if_a_database_is_open)
action_roi_fit_dce = Action("ROI fit DCE..", on_clicked=roi_fit_DCE, is_clickable=_if_a_database_is_open)
action_roi_fit_dce_cm = Action("ROI fit DCE (Cortex-Medulla)..", on_clicked=roi_fit_DCE_cm, is_clickable=_if_a_database_is_open)


menu_roifit = Menu('ROI analysis..')

menu_roifit.add(action_roi_fit_t1)
menu_roifit.add(action_roi_fit_t2)
menu_roifit.add(action_roi_fit_t2star)
menu_roifit.add(action_roi_fit_pc)
menu_roifit.add(action_roi_fit_dce)
menu_roifit.add(action_roi_fit_dce_cm)
menu_roifit.add_separator()
menu_roifit.add(action_roi_fit_all)



# OTHER STEPS


def _upload(app):
    folder = app.database()
    path = folder.path()
    filename_log = path + datetime.datetime.now().strftime('%Y%m%d_%H%M_') + "log_segmentation.txt"
    upload.main(path, filename_log)
    app.refresh()


action_upload = Action("Upload to drive..", on_clicked=_upload, is_clickable=_if_a_database_is_open)





