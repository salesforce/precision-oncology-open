"""Harcoded constants and functions to support RTOG data loading.


Steps to take if there are updates to the variable listings or excel files:
    1. Update rtog_variable_listings dict
    2. Update rtog_endpoints, if more endpoints have been added.
    3. Update rtog_unknown_class_X, if this changes the 'unknown class' value for any predictor variables.
    4. Update rtog_default_class_y, if this changes the 'default class' value for any endpoint variables.
    5. Update rtog_text_fields, if there are new text fields.
    6. Update rtog_field_mapping, if any names need to be standardized to existing conventions.
    7. Update rtog_categorical_fields, if any are categorical.
    8.

"""
import pint
import numpy as np

################################## Slide-Case Correspondence################################
_slideInfo_9408 = [
    '/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9408/11.10.2020/9408_11.10.2020 DeIDed Slide Information.xlsx',
    '/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9408/11.23.2020/9408_11.23.2020 DeIDed Slide Information.xlsx',
    '/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9408/12.8.2020/9408_12.8.2020 DeIDed Slide Information.xlsx',
    '/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9408/12.22.2020/9408_12.22.2020 DeIDed Slide Information.xlsx',
    '/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9408/1.12.2021/9408_1.12.2021 DeIDed Slide Information.xlsx',
    '/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9408/1.26.2021/9408_1.26.2021 DeIDed Slide Information.xlsx',
    '/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9408/2.16.2021/9408_2.16.2021 DeIDed Slide Information.xlsx',
    '/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9408/3.2.2021/9408_3.2.2021 DeIDed Slide Information.xlsx',
    '/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9408/3.16.2021/9408_3.16.2021 DeIDed Slide Information.xlsx',
    '/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9408/3.30.2021/9408_3.30.2021 DeIDed Slide Information.xlsx',
    '/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9408/4.13.2021/9408_4.13.2021 DeIDed Slide Information.xlsx',
    '/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9408/4.27.2021/9408_4.27.2021 DeIDed Slide Information.xlsx',
    '/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9408/5.11.2021/9408_5.11.2021 DeIDed Slide Information.xlsx',
    '/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9408/5.13.2021 all rescans/9408_5.13.2021 DeIDed Slide Information.xlsx',
    "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9408/6.9.2021/9408_6.9.2021 DeIDed Slide Information.xlsx",
    "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9408/7.28.2021/9408_7.28.2021 DE-ID'ed Export.xlsx",
    "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9408/8.3.2021 Histo/9408_8.3.2021-Deided- Histo Export.xlsx"
]

_slideInfo_9202 = [
    "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9202/1.12.2021/9202_1.12.2021 DeIDed Slide Information updated 2.4.2021.xlsx",
    "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9202/1.26.2021/9202_1.26.2021 DeIDed Slide Information updated 2.4.2021.xlsx",
    "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9202/10.21.2020/9202_10.21.2020 DeID Slide Information.xlsx" ,
    "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9202/10.5.2020/9202_10.5.2020 DeIDed Slide Information Updated 10.21.20.xlsx",
    "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9202/11.24.2020/9202_11.24.2020 DeIDed Slide Information updated 12.9.2020.xlsx",
    "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9202/11.9.2020/9202_11.9.2020 DeIDed Slide Information.xlsx",
    "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9202/12.22.2020/9202_12.22.2020 DeIDed Slide Information.xlsx",
    "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9202/12.8.2020/9202_12.8.2020 DeIDed Slide Information.xlsx",
    "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9202/2.16.2021/9202_2.16.2020 DeIDed Slide Information.xlsx",
    "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9202/3.16.2021/9202_3.16.2020 DeIDed Slide Information updated 4.5.2021.xlsx",
    "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9202/4.27.2021/9202_4.27.2021 DeIDed Slide Information.xlsx",
    "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9202/5.11.2021 all rescans/9202_5.11.2021 DeIDed Slide Information.xlsx",
    "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9202/5.25.2021/9202_5.25.2021 DeIDed Slide Information.xlsx",
    "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9202/9.14.2020/9202_9.14.2020 DeIDed Slide Information Updated 10.21.20.xlsx",
    "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9202/8.3.2021/9202_Upload_8.3.2021 De-ided.xlsx",
    "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9202/8.3.2021 Histo/9202 Histoslides_Deided-8.3.2021.xlsx"
]

_slideInfo_9413 = [
    "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9413/10.22.2020/9413_10.22.2020 DeIDed Slide Information.xlsx",
    "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9413/10.5.2020/9413_10.5.2020 DeIDed Slide Information Updated 10.21.20.xlsx",
    "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9413/11.24.2020/9413_11.24.2020 DeIDed Slide Information.xlsx",
    "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9413/11.9.2020 RC slides/9413_11.9.2020 DeIDed Slide Information.xlsx",
    "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9413/8.25.2020/8.25.2020 DeIDed Slide Information Updated 10.21.2020.xlsx",
    "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9413/5.11.2021 3 rescans/9413_5.11.2021 DeIDed Slide Information.xlsx",
    "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9413/6.9.2021/9413_6.9.2021 DeIDed Slide Information.xlsx",
    "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9413/7.28.2021 Histo/De-ided-9413_7.28.2021 Histology Export.xlsx" 
]

_slideInfo_9910 = [
    "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9910/6.9.2021/9910_6.9.2021 DeIDed Slide Information.xlsx",
    "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9910/6.21.2021/9910_6.21.2021 DeIDed Slide Information.xlsx",
    "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9910/7.21.2021 Blurry/9910_7.21.2021_Blurry DeIDed Slide Information.xlsx",
    "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9910/7.28.2021/Deided-9910_7.28.2021 Export.xlsx",
    "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9910/8.10.2021/9910-8.11.2021_deid-upload.xlsx"
]


_slideInfo_0126 = [
    "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-0126/6.30.2021/0126_6.30.2021 DeIDed Slide Information.xlsx",
    "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-0126/7.21.2021/0126_7.21.2021 DeIDed Slide Information.xlsx",
    "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-0126/7.21.2021 Blurry/0126_7.21.2021 Blurry DeIDed Slide Information.xlsx",
    "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-0126/7.28.2021/0126_7.28.2021 DE-ID'ed Export.xlsx",
    "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-0126/8.3.2021/0126_8.3.21 De-id-Export.xlsx",
    "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-0126/8.10.2021/0126-8.10.2021-Deided-upload.xlsx"
]

slide_case_correspondence_files = {
    '9202' : _slideInfo_9202,
    '9408' : _slideInfo_9408,
    '9413' : _slideInfo_9413,
    '9910' : _slideInfo_9910,
    '0126' : _slideInfo_0126
}


#################################  RTOG Pre-Processing #################################

# These variables should be removed from data matrix X prior to fitting a model.
rtog_confounding_variables = [
    'cn_deidentified',
    'rt_end_years',
    'rt_start_years',
    'randomization_year',
    'orchiectomy',
    'orchiectomy_years',
    'pelvis_start_years',
    'pelvis_end_years',
    'pelvis_total_dose',
    'pelvis_fractions',
    'prostate_start_years',
    'prostate_end_years',
    'prostate_total_dose',
    'prostate_fraction',
    'prostate_dose',
    'rt_end_reason',
    'rt_end_years',
    'rt_fractions',
    'rt_modality',
    'rt_start_years',
    'rt_total_dose',
    'sn',
    'baseline_serum',
    'baseline_serum_nmol_l',
    'baseline_serum_unit',
    'rx_orig',
    'energy_of_beam',
]


#################################  RTOG Clinical Data Loading #################################

# These may need updating when human error is discovered in RTOG files.
gcp_baseline_paths = {
    "9202" : "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/NRG Statistics/RTOG 9202/Baseline_and_results_9202.xlsx",
    "9413" : "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/NRG Statistics/RTOG 9413/Baseline_and_results_9413.xlsx",
    "9408" : "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/NRG Statistics/RTOG 9408/Baseline_and_results_9408_no_dup.xlsx",
#   "9408" : "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/NRG Statistics/RTOG 9408/Baseline_and_results_9408.xlsx",
    "9910" : "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/NRG Statistics/RTOG 9910/Baseline_and_efficacy_9910.xlsx",
    "0126" : "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/NRG Statistics/RTOG 0126/Baseline_RTTreat_efficacy_0126.xlsx",
}

"""Expected variable ranges in the RTOG variable listings.
Values Key:
'c' - continuous number
'i' - integer number
's' - string
{1,2,'a','b'} - set values
['c', {'a'}] - e.g. continous numbers or 'a'

Variables must match what is parsed by the RTOG class.
The raw Variable sheets may have 'inter_cardio'
But RTOG class will parse them to be 'intercurrent cardio'
The values below must use the RTOG version ('intercurrent cardio')

The variables must NOT match the variables after RTOG standardization
E.g. RTOG Class will convert race to be in {1,2,3,4,5}
But 9202 is in {1,2,3,4,5,6,7,8,9}
Keep the 9202 version.
"""
rtog_variable_listings = {
    '9202' : {
        'sn' : [{'9202', 9202}],
        'cn_deidentified' : ['i'],
        'rx' : [{1,2}],
        'randomization_year' : ['i'],
        'age' : ['i'],
        'kps' : ['i'],
        'race' : [{1,2,3,4,5,6,7,8,9}],
        'prior_cancer' : [{1,2,9}],
        'intercurrent_cardio' : [{0,1,2,9}],
        'intercurrent_diabetes' : [{0,1,2,9}],
        'intercurrent_hypertension' : [{0,1,2,9}],
        'intercurrent_other' : [{0,1,2,9}],
        'intercurrent_other_specified' : ['s'],
        'gleason_primary' : [{1,2,3,4,5,9}],
        'gleason_secondary' : [{1,2,3,4,5,9}],
        'gleason' : [{2,3,4,5,6,7,8,9,10}],
        'baseline_psa' : ['c'],
        'psa' : ['c'],
        'psa_years' : ['c'],
        'nstage' : [{0,1,2,3,9}],
        'tstage' : [set(range(17))], # Assume 9 = Tx (not given)
        'baseline_serum' : ['c'],
        'baseline_serum_nmol_l' : ['c'],
        'salvage_type' : [{1,2,3,4,5,6}],
        'salvage_comment' : ['s'],
        'salvage_years' : ['c'],
        'survival' : [{0,1}],
        'survival_years' : ['c'],
        'cod' : [{1,2,3,4,5,9}],
        'distant_mets' : [{0,1,2}],
        'distant_mets_years' : ['c'],
        'biochemical_failure' : [{0,1,2}],
        'biochemical_failure_years' : ['c'],
        'local_failure' : [{0,1,2}],
        'local_failure_years' : ['c'],
        'disease_free_survival' : [{0,1}],
        'disease_free_survival_years' : ['c'],
        'disease_specific_survival' : [{0,1,2}],
        'disease_specific_survival_years' : ['c'],
        'serum_value' : ['c'],
        'serum_unit' : ['s'],
        'serum_years' : ['c'],
        'ae_time_frame' : ['s'],
        'ae_cat' : ['s'],
        'ae_grade' : ['c'],
        'ae_date' : ['s'],
        'beam_whole_pelvis' : [{1,2,3,4,5,6}],
        'pelvis_dose' : ['c'],
        'pelvis_fraction' : ['c'],
        'rt_pelvis_start_date' : ['c'],
        'rt_pelvis_end_date' : ['c'],
        'beam_whole_prostate' : [{1,2,3,4,5,6}],
        'prostate_dose' : ['c'],
        'prostate_fraction' : ['c'],
        'rt_prostate_start_date' : ['c'],
        'rt_prostate_end_date' : ['c'],
        'beam_whole_para_aortics' : [{1,2,3,4,5,6}],
        'para_aortics_dose' : ['c'],
        'para_aortics_fraction' : ['c'],
        'rt_para_aortics_start_date' : ['c'],
        'rt_para_aortics_end_date' : ['c'],
        'rt_complete' : [{0,1,2,9}],
        'rt_disc_reason' : [{0,1,2,3,4,5}],
        'slide_id' : ['i'],
        'usi_short' : ['s'],
        'usi_specimen' : ['s'],
        'block_id' : ['c'],
    },
    '9408' : {
        'sn' : [{'9408', 9408}],
        'usi' : ['s'],
        'rx' : [{1, 2}],
        'randomization_year' : ['c'],
        'randomization_date' : ['d'], #Not in var listing, but in data.
        'age' : ['c'],
        'kps' : ['c'],
        'race' : [{1,2,3,4,5,6,98,99,}],
        'prior_cancer' : [{1,2,9}],
        'inter_cardio' : [{0,1,2,9}], # 0 not in original varlist, but its in other studies.
        'inter_diabetes' : [{0,1,2,9}],# 0 not in original varlist, but its in other studies.
        'inter_hyperextension' : [{0,1,2,9}],# 0 not in original varlist, but its in other studies.
        'inter_other' : [{0,1,2,9}],# 0 not in original varlist, but its in other studies.
        'inter_other_specified' : ['s'],
        'gleason_primary' : [{1,2,3,4,5,9}],
        'gleason_secondary' : [{1,2,3,4,5,9}],
        'gleason_combined' : [{2,3,4,5,6,7,8,9,10,99}],
        'baseline_psa' : ['c'],
        'tstage' : [{0,1,2,3,4,5,6,7,8,10,11,12,13,14}],
        'nstage' : [{0,1,2,3,9}],
        'mstage' : [{0,1,2,3,4,9}],
        'prior_surgery' : [{1,2,9}],
        'prior_rt' : [{1,2,9}],
        'prior_hormones' : [{1,2,9}],
        'prior_chemo' : [{1,2,9}],
        'baseline_serum' : ['c'],
        'baseline_serum_unit' : [{1,2,3,4}],
        'pelvis_total_dose' : ['c'],
        'pelvis_start_years' : ['c'],
        'pelvis_end_years' : ['c'],
        'pelvis_fractions' : ['c'],
        'prostate_total_dose' : ['c'],
        'prostate_start_years' : ['c'],
        'prostate_end_years' : ['c'],
        'prostate_fractions' : ['c'],
        'rt_end_reason' : [{0,1,2,3,4,5}],
        'energy_of_beam' : [{1,2,3,4,5}],
        'nonprotocol_years' : ['c'],
        'nonprotocol_type' : ['s'],
        'nonprotocol_details' : ['s'],
        'survival' : [{0,1}],
        'survival_years' : ['c'],
        'cod' : [{1,2,3,4,5,9}],
        'disease_free_survival' : [{0,1}],
        'dfs_years' : ['c'],
        'distant_met' : [{0,1,2}],
        'distant_met_years' : ['c'],
        'biochemical_failure' : [{0,1,2}], # The '2' is not in variable listing, but that's likely a typo.
        'biochemical_failure_years' : ['c'],
        'local_failure' : [{0,1,2}],# 2 not in original varlist, but its in other studies.
        'local_failure_years' : ['c'],
        'disease_specific_survival' : [{0,1,2}],
        'disease_specific_survival_years' : ['c'],
        'ae_time_frame' : [{1,2,3}],
        'ae_cat' : ['s'],
        'ae_grade' : ['c'],
        'ae_years' : ['c'],
        'cn_deidentified' : ['i'],
        'psa' : ['c'],
        'psa_years' : ['c'],
        'form' : ['s'],
        'serum_value' : ['c'],
        'serum_unit' : [{1,2,3,4}],
        'serum_years' : ['c'],
    },
    '9413' : {
        'sn' : [{'9413',9413}],
        'cn_deidentified' : ['i'],
        'rx' : [{1,2,3,4}],
        'randomization_year' : ['c'],
        'age' : ['c'],
        'kanofsky' : ['c'],
        'node_risk' : [{1,1.5,2}],#Added 1.5. Not in var listing.
        'race' : [{1,2,3,4,5,6,98,99}],
        'prior_cancer' : [{0,1,2,9}],
        'inter_cardio' : [{0,1,2,9}],
        'inter_diabetes' : [{0,1,2,9}],
        'inter_hyperextension' : [{0,1,2,9}],
        'inter_other' : [{0,1,2,9}],
        'gleason_primary' : [{1,2,3,4,5,9}],
        'gleason_combined' : [{2,3,4,5,6,7,8,9,10,99}],
        'baseline_psa' : ['c'],
        'tstage' : [{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}],
        'nstage' : [{0,1,2,3,9}],
        'prior_surgery' : [{1,2,9}],
        'prior_surgery_years' : ['c'],
        'prior_surgery_type' : ['s'],
        'prior_rt' : [{1,2,9}],
        'prior_rt_years' : ['c'],
        'prior_hormones' : [{1,2,9}],
        'prior_homrones_years' : ['c'],
        'orchiectomy' : [{1,2,9}],
        'orchiectomy_years' : ['c'],
        'prior_chemo' : [{1,2,9}],
        'prior_chemo_years' : ['c'],
        'primor_chemo_years' : ['s'],
        'baseline_serum' : ['c'],
        'baseline_serum_unit' : [{'NMOL/L','NG/DL','NG/ML','Unk'}],
        'rt_total_dose' : ['c'],
        'rt_start_years' : ['c'],
        'rt_end_years' : ['c'],
        'rt_fractions' : ['c'],
        'rt_end_reason' : [{0,1,2,3,4,5}],
        'energy_of_beam' : [{1,2,3,4,5}],
        'nonprotocol_years' : ['c'],
        'nonprotocol_type' : ['s'],
        'nonprotocol_details' : ['s'],
        'survival' : [{0,1}],
        'survival_years' : ['c'],
        'cod' : [{1,2,3,4}],
        'distant_mets' : [{0,1,2}],
        'distant_mets_years' : ['c'],
        'biochemical_failure' : [{0,1,2}],
        'biochemical_failure_years' : ['c'],
        'phoenix_bf' : [{0,1,2}],
        'phoenix_bf_years' : ['c'],
        'dfs_protocol' : [{0,1}],
        'dfs_protocol_years' : ['c'],
        'dfs_phoenix' : [{0,1}],
        'dfs_phoenix_years' : ['c'],
        'csf_protocol' : [{0,1,2}], #Added the 2. Not in var listing. Probably competing event.
        'csf_protocol_years' : ['c'],
        'csf_phoenix' : [{0,1,2}],# Added the 2. Not in var listing.
        'csf_phoenix_years' : ['c'],
        'ae_time_frame' : [{1,2,3}],
        'ae_cat' : ['s'],
        'ae_grade' : ['c'],
        'ae_date_' : ['c'],
        'psa' : ['c'],
        'psa_unit' : [{1,2,3,4,5,6,7}],
        'psa_years' : ['c'],
        'form' : ['s'],
        'serum_value' : ['c'],
        'serum_unit' : [{1,2,3,4}],
        'serum_years' : ['c'],
        'slide_id' : ['c'],
        'usi_short' : ['s'],
        'usi_specimen' : ['s'],
        'block_id' : ['c'],
    },
    '9910' : {
        'sn' : [{'9910',9910}],
        'cn_deidentified' : ['i'],
        'rx' : [{1,2}],
        'randomization_year' : ['c'],
        'age' : ['c'],
        'race' : [{1,2,3,4,5,6,98,99}],
        'prior_cancer' : [{0,1,2,9}],
        'inter_cardio' : [{0,1,2,9}],
        'inter_diabetes' : [{0,1,2,9}],
        'inter_hyperextension' : [{0,1,2,9}],
        'inter_other' : [{0,1,2,9}],
        'primary_gleason' : ['c'],
        'secondary_gleason' : ['c'],
        'gleason' : [{2,3,4,5,6,7,8,9,10}],
        'baseline_psa' : ['c'],
        'tstage' : [{0,1,2,3,4,5,6,7,10,11,12,13,9}],
        'nstage' : [{0,1,9}],
        'prior_surgery' : [{0,1,2,9}],
        'prior_surgery_type' : ['s'],
        'prior_rt' : [{0,1,2,9}],
        'prior_hormones' : [{0,1,2,9}],
        'prior_hormone_years' : ['c'],
        'orchiectomy' : [{0,1,2,9}],
        'prior_chemo' : [{0,1,2,9}],
        'prior_chemo_agent' : ['s'],
        'rt_total_dose' : ['c'],
        'rt_start_years' : ['c'],
        'rt_end_years' : ['c'],
        'rt_fractions' : ['c'],
        'rt_end_reason' : [{1,2,3,4,5,6,7,8,9}],
        'pelvic_rt' : ['s'],
        'nodal_site1' : [set(range(1,21))],
        'nodal_site2' : [set(range(1,21))],
        'nodal_site3' : [set(range(1,21))],
        'nodal_site4' : [set(range(1,21))],
        'nodal_site5' : [set(range(1,21))],
        'nodal_rt_fx1' : ['c'],
        'nodal_rt_fx2' : ['c'],
        'nodal_rt_fx3' : ['c'],
        'nodal_rt_fx4' : ['c'],
        'nodal_rt_fx5' : ['c'],
        'nodal_rt_dose1' : ['c'],
        'nodal_rt_dose2' : ['c'],
        'nodal_rt_dose3' : ['c'],
        'nodal_rt_dose4' : ['c'],
        'nodal_rt_dose5' : ['c'],
        'hormone' : ['s'],
        'hormone_start_years' : ['c'],
        'hormone_end_years' : ['c'],
        'hormone_total_dose' : ['c'],
        'hormone_end_reason' : [{1,2,3,4,5,6,7,8,9}],
        'survival' : [{0,1}],
        'survival_years' : ['c'],
        'cod' : ['s'],
        'disease_specific_survival' : [{0,1,2}],
        'disease_specific_years' : ['c'],
        'disease_free_survival' : [{0,1}],
        'disease_free_survival_years' : ['c'],
        'distant_metastasis' : [{0,1,2}],
        'distant_metastasis_years' : ['c'],
        'local_progression' : [{0,1,2}],
        'local_progression_years' : ['c'],
        'phoenix_failure' : [{0,1,2}],
        'phoenix_failure_years' : ['c'],
        'biochemical_failure' : [{0,1,2}],
        'biochemical_failure_years' : ['c'],
        'non_protocol_hormone' : [{0,1,2}],
        'non_protocol_hormone_years' : ['c'],
        'ae_cat' : ['s'],
        'ae_grade' : ['c'],
        'ae_date_' : ['c'],
        'psa' : ['c'],
        'psa_years' : ['c'],
        'form' : ['s'],
    },
    '0126' : {
        'sn' : [{'0126',126}],
        'cn_deidentified' : ['c'],
        'rx' : [{1,2}],
        'randomization_year' : ['c'],
        'age' : ['c'],
        'race' : [{1,2,3,4,5,6,9}],
        'ethnicity' : [{1,2,9}],
        'zubrod' : ['c'],
        'prior_cancer' : [{0,1,2,9}],
        'primary_gleason' : ['c'],
        'secondary_gleason' : ['c'],
        'gleason' : [{2,3,4,5,6,7,8,9,10}],
        'baseline_psa' : ['c'],
        't_stage' : [{1,2,3,4,5,6,7}],
        'n_stage' : [{0,1,9}],
        'm_stage' : [{0,1,9}],
        'prior_surgery' : [{0,1,2,9}],
        'prior_surgery_years' : ['c'],
        'prior_rt' : [{0,1,2,9}],
        'prior_hormones' : [{0,1,2,9}],
        'prior_hormones_years' : ['c'],
        'orchiectomy' : [{0,1,2,9}],
        'orchiectomy_years' : ['c'],
        'prior_chemo' : [{0,1,2,9}],
        'prior_chemo_years' : ['c'],
        'rt_modality' : [{1,2}],
        'rt_total_dose' : ['c'],
        'rt_start_years' : ['c'],
        'rt_end_years' : ['c'],
        'rt_fractions' : ['c'],
        'rt_end_reason' : [{1,2,3,4,5,6,7,8,9,98,99}],#98,99 not in var listing. adding.
        'nonprotocol_years' : ['c'],
        'nonprotocol_type' : ['s'],
#       'nonprotocol_type' : [{1,2,3,4,5,6,7,8,9}],
        'survival' : [{0,1}],
        'survival_years' : ['c'],
        'cod' : ['s'],
        'disease_specific_survival' : [{0,1,2}],
        'disease_specific_years' : ['c'],
        'disease_free_survival' : [{0,1}],
        'disease_free_survival_years' : ['c'],
        'distant_metastasis' : [{0,1,2}],
        'distant_metastasis_years' : ['c'],
        'local_progression' : [{0,1,2}],
        'local_progression_years' : ['c'],
        'astro_bf' : [{0,1,2}],
        'astro_bf_years' : ['c'],
        'phoenix_bf' : [{0,1,2}],
        'phoenix_bf_years' : ['c'],
        'ae_cat' : ['s'],
        'ae_grade' : ['c'],
        'ae_timeframe' : [{1,2}],
        'ae_years' : ['c'],
        'psa' : ['c'],
        'psa_years' : ['c'],
        'form' : ['s'],
        'serum_value' : ['c'],
        'serum_unit' : [{1,2,3}],
        'serum_years' : ['c'],
    }
}

# Study endpoints (e.g. DM). These are hardcoded so that these vars can be eliminating in creating (X,y) training matrices.
rtog_endpoints = {
    '9202' : [
        'biochemical_failure',
        'biochemical_failure_years',
        'cod',
        'distant_met',
        'distant_met_years',
        'disease_free_survival',
        'disease_free_survival_years',
        'disease_specific_survival',
        'disease_specific_survival_years',
        'local_failure',
        'local_failure_years',
        'survival',
        'survival_years',
        'salvage_type',
        'salvage_years',
        'salvage_comment',
       ],
    '9413' : [
        'biochemical_failure',
        'biochemical_failure_years',
        'phoenix_biochemical_failure',
        'phoenix_biochemical_failure_years',
        'cod',
        'distant_met',
        'distant_met_years',
        'survival',
        'survival_years',
        'nonprotocol_years',
        'nonprotocol_type',
        'nonprotocol_details',
        'disease_free_survival_protocol',
        'disease_free_survival_protocol_years',
        'disease_free_survival_phoenix',
        'disease_free_survival_phoenix_years',
        'disease_specific_survival',
        'disease_specific_survival_years',
        'disease_specific_survival_protocol',
        'disease_specific_survival_protocol_years',
    ],
    '9408' : [
        'biochemical_failure',
        'biochemical_failure_years',
        'cod',
        'distant_met',
        'distant_met_years',
        'disease_specific_survival', #note: original var is dsm_status. This is to standardize with 9202.
        'disease_specific_survival_years',
        'disease_free_survival',
        'disease_free_survival_years',
        'local_failure',
        'local_failure_years',
        'survival',
        'survival_years',
        'nonprotocol_years',
        'nonprotocol_type',
        'nonprotocol_details',
    ],
    '9910' : [
        'survival',
        'survival_years',
        'cod',
        'disease_specific_survival',
        'disease_specific_survival_years',
        'disease_free_survival',
        'disease_free_survival_years',
        'distant_met',
        'distant_met_years',
        'local_progression',
        'local_progression_years',
        'phoenix_biochemical_failure', #note: in this study, phoenix_failure is a better version of biochem failure. A more recent definition.
        'phoenix_biochemical_failure_years',
        'biochemical_failure',
        'biochemical_failure_years',
        'non_protocol_hormone',
        'non_protocol_hormone_years',
    ],
    '0126' : [
        'survival',
        'survival_years',
        'cod',
        'disease_specific_survival',
        'disease_specific_survival_years',
        'disease_free_survival',
        'disease_free_survival_years',
        'distant_met',
        'distant_met_years',
        'local_progression',
        'local_progression_years',
        'astro_biochemical_failure',
        'astro_biochemical_failure_years',
        'phoenix_biochemical_failure', #note: in this study, phoenix_failure is a better version of biochem failure. A more recent definition.
        'phoenix_biochemical_failure_years',
        'biochemical_failure',
        'biochemical_failure_years',
        'nonprotocol_years',
        'nonprotocol_type',
    ],
}

# Mapping to convert multi-class problems into binary problems. Study-specific.
rtog_binary_mapping = {
    '9202' : {
        'distant_met': {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'distant_met_5year': {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'distant_met_10year': {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'distant_met_15year': {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'distant_met_25year': {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'biochemical_failure' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'biochemical_failure_5year' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'biochemical_failure_10year' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'biochemical_failure_15year' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'biochemical_failure_25year' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'disease_specific_survival' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'disease_specific_survival_5year' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'disease_specific_survival_10year' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'disease_specific_survival_15year' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'disease_specific_survival_25year' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'survival_5year' : {1:1, 0:0}, # 0 - Alive, 1 - dead
        'survival_10year' : {1:1, 0:0}, # 0 - Alive, 1 - dead
        'survival_15year' : {1:1, 0:0}, # 0 - Alive, 1 - dead
        'survival_25year' : {1:1, 0:0}, # 0 - Alive, 1 - dead
    },
    '9408' : {
        'distant_met': {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'distant_met_5year': {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'distant_met_10year': {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'distant_met_15year': {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'distant_met_25year': {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'biochemical_failure' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'biochemical_failure_5year' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'biochemical_failure_10year' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'biochemical_failure_15year' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'biochemical_failure_25year' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'disease_specific_survival' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'disease_specific_survival_5year' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'disease_specific_survival_10year' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'disease_specific_survival_15year' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'disease_specific_survival_25year' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'survival_5year' : {1:1, 0:0}, # 0 - Alive, 1 - dead
        'survival_10year' : {1:1, 0:0}, # 0 - Alive, 1 - dead
        'survival_15year' : {1:1, 0:0}, # 0 - Alive, 1 - dead
        'survival_25year' : {1:1, 0:0}, # 0 - Alive, 1 - dead
    },
    '9413' : {
        'distant_met': {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'distant_met_5year': {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'distant_met_10year': {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'distant_met_15year': {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'distant_met_25year': {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'biochemical_failure' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'biochemical_failure_5year' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'biochemical_failure_10year' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'biochemical_failure_15year' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'biochemical_failure_25year' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'disease_specific_survival' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'disease_specific_survival_5year' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'disease_specific_survival_10year' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'disease_specific_survival_15year' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'disease_specific_survival_25year' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'survival_5year' : {1:1, 0:0}, # 0 - Alive, 1 - dead
        'survival_10year' : {1:1, 0:0}, # 0 - Alive, 1 - dead
        'survival_15year' : {1:1, 0:0}, # 0 - Alive, 1 - dead
        'survival_25year' : {1:1, 0:0}, # 0 - Alive, 1 - dead
    },
    '9910' : {
        'distant_met': {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'distant_met_5year': {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'distant_met_10year': {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'distant_met_15year': {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'distant_met_25year': {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'biochemical_failure' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'biochemical_failure_5year' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'biochemical_failure_10year' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'biochemical_failure_15year' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'biochemical_failure_25year' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'disease_specific_survival' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'disease_specific_survival_5year' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'disease_specific_survival_10year' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'disease_specific_survival_15year' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'disease_specific_survival_25year' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'survival_5year' : {1:1, 0:0}, # 0 - Alive, 1 - dead
        'survival_10year' : {1:1, 0:0}, # 0 - Alive, 1 - dead
        'survival_15year' : {1:1, 0:0}, # 0 - Alive, 1 - dead
        'survival_25year' : {1:1, 0:0}, # 0 - Alive, 1 - dead
    },
    '0126' : {
        'distant_met': {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'distant_met_5year': {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'distant_met_10year': {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'distant_met_15year': {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'distant_met_25year': {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'biochemical_failure' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'biochemical_failure_5year' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'biochemical_failure_10year' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'biochemical_failure_15year' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'biochemical_failure_25year' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'disease_specific_survival' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'disease_specific_survival_5year' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'disease_specific_survival_10year' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'disease_specific_survival_15year' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'disease_specific_survival_25year' : {1:1, 0:0, 2:0}, # 0 - censored, 1 - Failure, 2 - competing Event
        'survival_5year' : {1:1, 0:0}, # 0 - Alive, 1 - dead
        'survival_10year' : {1:1, 0:0}, # 0 - Alive, 1 - dead
        'survival_15year' : {1:1, 0:0}, # 0 - Alive, 1 - dead
        'survival_25year' : {1:1, 0:0}, # 0 - Alive, 1 - dead
    },
}

# For predictor variables (X) imputation:
# boosting performs better when data can be intelligently imputed.
# Meaning it is better to have a 'nan' than an 'Unknown' class.
# The below dictionary identifies the unknown classes for predictor variables, so that they can be set to nans.
# the function standardize_unknown_values_in_predictor_variables sets them to nans.
rtog_unknown_class_X = {
    '9202' : {
        'kps' : [9],
        'race' : [8,9],
        'prior_cancer' : [9, 0],
        'intercurrent_cardio' : [9,0],
        'intercurrent_diabetes' : [9,0],
        'intercurrent_hypertension' : [9,0],
        'intercurrent_other' : [9,0],
        'gleason_primary' : [9],
        'gleason_secondary' : [9],
        'nstage' : [9],
        'cod' : [9],
    },
    '9408' : {
        'kps' : [9],
        'race' : [99],
        'prior_cancer' : [9, 0],
        'intercurrent_cardio' : [9,0],
        'intercurrent_diabetes' : [9,0],
        'intercurrent_hypertension' : [9,0],
        'intercurrent_other' : [9,0],
        'gleason_primary' : [9],
        'gleason_secondary' : [9],
        'gleason_combined' : [99],
        'nstage' : [9],
        'mstage' : [9],
        'prior_surgery' : [9],
        'prior_rt' : [9],
        'prior_hormones' : [9],
        'prior_chemo' : [9],
        'cod' : [9],
    },
    '9413' : {
        'kps' : [9],
        'prior_cancer' : [9, 0],
        'intercurrent_cardio' : [9,0],
        'intercurrent_diabetes' : [9,0],
        'intercurrent_hypertension' : [9,0],
        'intercurrent_other' : [9,0],
        'gleason_combined' : [99],
        'gleason_primary' : [9],
        'prior_surgery' : [9],
        'prior_rt' : [9],
        'prior_hormones' : [9],
        'prior_chemo' : [9],
        'cod' : [4],
        'race' : [98,99],
    },
    '9910' : {
        'race' : [98, 99],
        'prior_cancer' : [0, 9],
        'intercurrent_cardio' : [9,0],
        'intercurrent_diabetes' : [9,0],
        'intercurrent_hypertension' : [9,0],
        'intercurrent_other' : [9,0],
        'gleason_primary' : [9],
        'gleason_secondary' : [9],
        'prior_surgery' : [0,9],
        'prior_rt' : [0,9],
        'prior_hormones' : [0,9],
        'orchiectomy' : [0,9],
        'prior_chemo' : [0,9],
        'rt_end_reason' : [8,9],
    },
    '0126' : {
        'race' : [9],
        'ethnicity' : [9],
        'gleason_primary' : [9],
        'gleason_secondary' : [9],
        'prior_cancer' : [0,9],
        'prior_surgery' : [0,9],
        'prior_rt' : [0,9],
        'prior_hormones' : [0,9],
        'orchiectomy' : [0,9],
        'prior_chemo' : [0,9],
        'rt_end_reason' : [8,9,98,99],
    }
}

# For endpoint variable (y) imputation:
# Each variable is assigned a default class. This is the class that is used to replace nans, in the raw data.
# This can be a new class (e.g. -1), if the options are non-complete. E.g. 'cod' in 9202 has values 1-9 for various causes of death, but no 'didn't die' class.
# This can be an existing class (e.g. 0), if the options are complete. E.g. 'distant_met' in 9202 has value of 0=censored, meaning neither DM nor competing event occured.
rtog_default_class_y = {
    '9202' : {
        'survival' : -1, #default class: not given
        'cod' : 0, #default class standardized to: 0: Other/ Unknown
        'distant_met' : 0, #default class: Censored
        'biochemical_failure' : 0, #default class: Censored
        'local_failure' : 0, #default class: Censored
        'disease_free_survival' : -1, #default class: not given
        'disease_specific_survival' : 0, #default class: Censored
    },
    '9413' : {'survival' : -1, #default class: not given
        'cod' : 0, #default class standardized to: 0: Other/ Unknown
        'distant_met' : 0, #default class: Censored
        'biochemical_failure' : 0, #default class: Censored
        'disease_free_survival' : 0, #default class; Censored.
        'disease_specific_survival' : 0, #default class: Censored
    },
    '9408' : {'survival' : -1, #default class: not given
        'cod' : 0, #default class standardized to: 0: Other/ Unknown
        'distant_met' : 0, #default class: Censored
        'biochemical_failure' : 0, #default class: Censored
        'local_failure' : 0, #default class: Censored
        'disease_specific_survival' : 0, #default class: Censored
    },
    '9910' : {
        'survival' : -1, #default class: not given
        'cod' : 0, #default class standardized to: 0: Other/ Unknown
        'disease_specific_survival' : 0,
        'disease_free_survival' : -1,
        'distant_met' : 0, #default class: Censored
        'local_progression' : 0,
        'biochemical_failure' : 0, #default class: Censored
        'non_protocol_hormone' : 0,
    },
    '0126' : {
        'survival' : -1, #default class: not given
        'cod' : 0, #default class standardized to: 0: Other/ Unknown
        'disease_specific_survival' : 0,
        'disease_free_survival' : -1,
        'distant_met' : 0, #default class: Censored
        'local_progression' : 0,
        'biochemical_failure' : 0, #default class: Censored
        'non_protocol_hormone' : 0,
    },
}

# Unstructured text fields.
rtog_text_fields = {
    '9202' : [
        'intercurrent_other_specified',
        'salvage_comment',
    ],
    '9408' : [
        'intercurrent_other_specified',
        'nonprotocol_type',
        'nonprotocol_details',
        'randomization_date',
    ],
    '9413' : [
        'prior_surgery_type',
    ],
    '9910' : [
        'prior_surgery_type',
        'prior_chemo_agent',
        'pelvic_rt',
        'cod', # This is converted to a categorical variable by the initializer
#       'cn_deidentified', #This is converted to a categorical variable by the initializer
    ],
    '0126' : [
#       'ae_cat',
#       'form',
    ],
}

# Mapping to account for discrepancies in field names across studies.
rtog_field_mapping = {
    "inter_cardio" : "intercurrent_cardio",
    "inter_diabetes" : "intercurrent_diabetes",
    "inter_hypertension" : "intercurrent_hypertension",
    "inter_hyperextension" : "intercurrent_hypertension",
    "inter_other" : "intercurrent_other",
    "intercurent_other" : "intercurrent_other",
    "inter_other_specified" : "intercurrent_other_specified",
    "intercurent_other_specified" : "intercurrent_other_specified",
    "distant_mets" : "distant_met",
    "distant_metastasis" : "distant_met",
    "distant_mets_years" : "distant_met_years",
    "distant_metastasis_years" : "distant_met_years",
    "disease_specific_years" : "disease_specific_survival_years",
    "dsm_status" : "disease_specific_survival",
    "dsm_statuts" : "disease_specific_survival",
    "dsm_years" : "disease_specific_survival_years",
    "dfs_years" : "disease_free_survival_years",
    "dfs_protocol" : "disease_free_survival_protocol",
    "dfs_protocol_years" : "disease_free_survival_protocol_years",
    "dfs_phoenix" : "disease_free_survival_phoenix",
    "dfs_phoenix_years" : "disease_free_survival_phoenix_years",
    "csf_protocol" : "disease_specific_survival_protocol",
    "csf_protocol_years" : "disease_specific_survival_protocol_years",
    "csf_phoenix" : "disease_specific_survival",
    "csf_phoenix_years" : "disease_specific_survival_years",
    "kanofsky" : "kps",
    "primary_gleason" : "gleason_primary",
    "secondary_gleason" : "gleason_secondary",
    "t_stage" : "tstage",
    "m_stage" : "mstage",
    "n_stage" :"nstage",
    "prostate_fractions" : "prostate_fraction",
    "primary_gleason" : "gleason_primary",
    "secondary_gleason" : "gleason_secondary",
    "randomized_year" : "randomization_year",
    "astro_bf" : "astro_biochemical_failure",
    "astro_bf_years" : "astro_biochemical_failure_years",
    "phoenix_bf" : "phoenix_biochemical_failure",
    "phoenix_bf_years" : "phoenix_biochemical_failure_years",
    "phoenix_failure" : "phoenix_biochemical_failure",
    "phoenix_failure_years" : "phoenix_biochemical_failure_years",
    "prior_hormones_years" : "prior_hormone_years",

}
# Add additional fields to rtog_variable_listings to include
# name adjustments according to rtog_field_mapping
for sn, fields in rtog_variable_listings.items():
    new_fields = {}
    for fn, fvals in fields.items():
        if fn in rtog_field_mapping:
            new_fields[rtog_field_mapping[fn]] = fvals
    fields.update(new_fields)

# Catboost needs to know which input data fields are categorical vs numerical.
rtog_categorical_fields = {
    #9202
    "sn",
    "cn_deidentified",
    "rx",
    "race",
    "prior_cancer",
    "intercurrent_cardio",
    "intercurrent_diabetes",
    "intercurrent_hypertension",
    "intercurrent_other",
    "salvage_type",
    "survival"
    "cod",
    "distant_met",
    "biochemical_failure",
    "local_failure",
    "disease_free_survival",
    "disease_specific_survival",
    "rt_complete",
    "rt_disc_reason",
    "pelvic_rt",

    #9408
    "prior_surgery",
    "prior_rt",
    "prior_hormones",
    "prior_chemo",
    "baseline_serum_unit",
    "rt_end_reason",
    "ae_time_frame",
    "ae_cat",
    "serum_unit",

    #9413
    "node_risk",
    "orchiectomy",
    "psa_unit",

    #9910
    "hormone_end_reason",

    #0126
    "ethnicity",
    "nonprotocol_type",
    "local_progression",
    "astro_biochemical_failure",
    "phoenix_biochemical_failure",
    "ae_timeframe",
}


def is_categorical(list_of_fields):
    """Returns boolean vector s.t. if list_of_fields[i] is categorical, then b[i] == 1, 0 otherwise.
    """
    b = [l in rtog_categorical_fields for l in list_of_fields]
    return b


def merge(rtog_iterable):
    df = pd.concat([item.df for item in rtog_iterable], axis=0)
    rtog = RTOG()
    rtog.df = df
    return rtog


def encode_X_as_one_hot(X, col_indices=[]):
    """Converts X[col_indices] into one-hot vector representations
    Args:
        X(numpy.array): Feature matrix
        col_indices(list): the indices to expand

    E.g.
    X = [[0, 1],
         [1, 1],
         [3, 2],
        ]
    col_indices = [0]
    This will return:
    array([[1., 0., 0., 1.],
           [0., 1., 0., 1.],
           [0., 0., 1., 2.]])
    """
    if len(col_indices) == 0:
        return X
    assert len(X.shape) == 2
    encoded_x = None
    for i in range(X.shape[1]):
        feature = X[:,i]
        if i in col_indices:
            label_encoder = LabelEncoder()
            feature = label_encoder.fit_transform(feature)
            feature = feature.reshape(X.shape[0], 1)
            onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
            feature = onehot_encoder.fit_transform(feature)
        else:
            feature = feature.reshape(X.shape[0], 1)

        if encoded_x is None:
            encoded_x = feature
        else:
            encoded_x = np.concatenate((encoded_x, feature), axis=1)

    return encoded_x


def serum_values_to_ng_dl(value, units):
    """Converts the value to ng/dl. "units" must be one of "nmol/l" or "ng/ml" or "ng/dl" or "unit/nos"
    e.g.
    value = 23.00
    units = 'nmol/l'
    serum_values_to_ng_dl(value, units)
    # prints 663.3660000000002 nanogram/deciliter

    value = 2.80
    units = 'ng/ml'
    serum_values_to_ng_dl(value, units)
    # prints 280.00000000000006 nanogram/deciliter


    """
    ureg = pint.UnitRegistry()
    testosterone_molar_mass = 288.42 #g/mol

    if units.lower() == 'nmol/l':
        value = value * ureg.nmol / ureg.liter
        value = value.to('ng/dl', 'chemistry', mw=testosterone_molar_mass * ureg('g/mol')).magnitude
    elif units.lower() == 'ng/ml':
        value = value * ureg.ngram / ureg.mliter
        value = value.to('ng/dl').magnitude
    elif units.lower() == 'ng/dl':
        pass
    elif units.lower() == 'unit/nos':
        print("Units of type unit/nos, settin value to nan")
        value = np.nan
    else:
        raise ValueError("Unit type {} not supported".format(units))

    return(value)


def drop_confounding_variables(df):
    drop_cols =[var for var in rtog_confounding_variables if var in df.columns]
    return df.drop(columns=drop_cols)
