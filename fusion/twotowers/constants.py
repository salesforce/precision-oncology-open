import pathlib

PROJECT_DATA_DIR = pathlib.Path('/export/medical-ai/ucsf')
TEST_SETS_DIR = PROJECT_DATA_DIR / 'test_sets'
#ORIGINAL_EMR = TEST_SETS_DIR / 'distant_met_5year_0.20pertrial_seed3_nccn=0.677.pkl' # original data
ORIGINAL_EMR = TEST_SETS_DIR / '/export/medical-ai/ucsf/test_sets/distant_met_5year_0.20pertrial_seed2_nccn=0.701.pkl'  # with new test set
PARSED_EMR = TEST_SETS_DIR / 'distant_met_5year_0.20pertrial_seed3_nccn=0.677_parsed.pkl.3' # original
#PARSED_EMR = TEST_SETS_DIR / 'distant_met_5year_0.20pertrial_seed3_nccn=0.677_parsed.pkl.2'
#PARSED_EMR = TEST_SETS_DIR / 'distant_met_5year_0.20pertrial_seed3_nccn=0.677_parsed.pkl'
#PARSED_EMR = TEST_SETS_DIR / 'distant_met_5year_0.20pertrial_seed3_nccn=0.677_parsed.pkl.4'   # with new test set 
PARSED_EMR_2 = TEST_SETS_DIR / 'distant_met_5year_0.20pertrial_seed3_nccn=0.677_parsed.pkl.5'   # with new test set but original test split

SSL_RTOG_DIR = PROJECT_DATA_DIR / 'ssl_rtog'
MOCO_FEATURE_QUILT = {
    'r50': SSL_RTOG_DIR / 'moco/model_R50_b=256_lr=0.03_pg4plus/features/RTOG-{}_quilts/',
    'r50_imagenet': SSL_RTOG_DIR / 'model_R50_b=256_lr=0.03_pg4plus_imagenet_pretrained-800/features/RTOG-{}_quilts/' }

STUDY_NUMS = ['9202', '9408', '9413', '9910', '0126']


# labels to binary mapping
TARGETS = {
    'distant_met_5year': ['distant_met_5year'],
    'year_any': ['15year_any', '10year_any', '5year_any'],
    'task_any': ['survival_any', 'biochemical_failure_any', 'distant_met_any', 'biochemical_failure_any'],
    'all': [
        'biochemical_failure_15year', 'biochemical_failure_10year', 'biochemical_failure_5year', 
        'distant_met_15year', 'distant_met_10year', 'distant_met_5year',
        'disease_specific_survival_10year', 'disease_specific_survival_15year', 'disease_specific_survival_5year',
        'survival_15year', 'survival_10year', 'survival_5year'],
}


# TODO: re-use from rtog_constants
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
    "astro_bf",
    "phoenix_bf",
    "ae_timeframe",
}


# labels to binary mapping
BINARY_LABELS = {
    'biochemical_failure_15year': True, 
    'biochemical_failure_10year': True,
    'biochemical_failure_5year': True,
    #'biochemical_failure': True,
    #'biochemical_failure_years': False,
    'distant_met_15year': True,
    'distant_met_10year': True,
    'distant_met_5year': True,
    'distant_met': True,
    'distant_met_years': False,
    'disease_specific_survival_10year': True,
    'disease_specific_survival_15year': True,
    'disease_specific_survival_5year': True,
    'disease_specific_survival': True,
    'disease_specific_survival_years': False,
    'survival_15year': True,
    'survival_10year': True,
    'survival_5year': True,
    #'survival': True,
    'survival_years': False,
}