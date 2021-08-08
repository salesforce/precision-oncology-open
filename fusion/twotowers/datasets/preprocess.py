"""
TODO: rtog_constants.py:  /export/medical_ai/ -> /export/medical-ai/
Adapted from: https://github.com/MetaMind/precision_oncology/blob/master/clinical_data_classifier/Clinical%20Data%20Classifier.ipynb
"""

import pickle
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.getcwd())
sys.path.append('../clinical_data_classifier')

from rtog_constants import drop_confounding_variables, is_categorical
from rtog_helper import RTOG, rtog_from_study_number
from rtog_image_helper import ImageLoader 


from twotowers.constants import *
from sklearn.preprocessing import LabelEncoder

np.random.seed(6)



def convert_tabnet_format(df, categorical_names):
    """Fills numerical nans with the column mean, and label-encodes categorical variables"""

    df_tn = df.copy()
    for col in df_tn.columns:
        if col not in categorical_names:
            df_tn[col].fillna(df_tn[col].mean(), inplace=True)
        else:
            l_enc = LabelEncoder()
            df_tn[col] = df_tn[col].fillna('nan')
            df_tn[col] = l_enc.fit_transform(df_tn[col].values)
    df_tn.fillna(0, inplace=True)
    return df_tn


def main():

    # TODO
    p1 = pickle.load(open('/export/medical-ai/ucsf/test_sets/distant_met_5year_0.20pertrial_seed3_nccn=0.677_parsed.pkl.3', 'rb'))['df_full']
    train_id = p1[p1.split != 'test']['id'].tolist()
    test_id = p1[p1.split == 'test']['id'].tolist()

    # read in original dataframes
    original_emr = pickle.load(open(ORIGINAL_EMR, 'rb'))
    df_X_full = original_emr['df_X_full']
    X_train = original_emr['X_train']

    # TODO
    df_X_full['id'] = df_X_full.apply(lambda x: f'{x.sn}_{x.cn_deidentified}',axis=1)
    df_X_full['test_set'] = df_X_full['id'].apply(lambda x: 1 if x in test_id else 0)

    # define columns to keep from 
    img_feature_cols = [f'f{i}' for i in range(128)]
    feature_cols = [c for c in X_train.columns if c not in img_feature_cols]
    identifier_cols = ['sn', 'cn_deidentified', 'test_set']

    # create copy from full dataframe
    df_X = df_X_full[feature_cols].copy()

    # dropping rx and pelvic_rt for prognostic
    df_X = df_X.drop(columns=['rx', 'pelvic_rt'])
    feature_cols = df_X.columns.tolist()

    # Determine which variables are categorical, and convert them into strings for Catboost
    categorical = is_categorical(df_X.columns)
    categorical = np.where(categorical)[0]
    categorical_names = df_X.columns[categorical]
    numerical = np.array(list(set(range(len(df_X.columns))) - set(categorical)))
    numerical_names = df_X.columns.values[numerical]
    for nn in numerical_names:
        assert nn not in categorical_names
    assert len(numerical) + len(categorical) == len(df_X.columns)

    # Catboost requires categorical variables to be string or integer. Float and nans must be converted to strings.
    categorical_dims = {}
    for col in categorical_names:
        df_X[col] = df_X[col].astype(str)
        categorical_dims[col] = len(df_X[col].unique())
    cat_idxs = np.array(list(categorical))
    cat_dims = np.array(list(categorical_dims.values()))
    cat_emb_dims = list(map(int, np.round(1.6 * cat_dims ** 0.56)))
    df_X = convert_tabnet_format(df_X, categorical_names)

    # combine identifiers with features
    df_id = df_X_full[identifier_cols].copy()
    df_X = pd.concat([df_X, df_id], axis=1)

    # Add quilt path
    for sn in STUDY_NUMS:
        feat_dir = str(MOCO_FEATURE_QUILT['r50']).format(sn)
        # Add the feature quilts if available
        if os.path.exists(feat_dir):
            feature_quilt_paths = sorted([os.path.join(feat_dir,i) for i in os.listdir(feat_dir) if '.pkl' in i])
            cn_deids = np.array([int(i.split("/")[-1].split('_')[0]) for i in feature_quilt_paths])

        for p, cn in zip(feature_quilt_paths, cn_deids):
            #df_X.loc[(df_X.sn == int(sn)) & (df_X.cn_deidentified == cn), ['quilt_path']] = p
            df_X.loc[(df_X.sn == int(sn)) & (df_X.cn_deidentified == cn), ['featquilt']] = p


    # load featurized images 
    loader = ImageLoader()
    loader.load_feature_matrix(df_X)    
    df_X = df_X.rename(columns={'featquilt':'quilt_path'})

    # get labels
    df = df_X
    for y_var, mb in BINARY_LABELS.items():
        rtogs_y = []
        for sn in STUDY_NUMS: 

            print(sn, y_var)

            rtog = rtog_from_study_number(sn, standardize=True)
            rx, ry, rm = rtog.get_Xy(y_var=y_var, make_binary=mb)
            rtogs_y.append(ry.df)
        df_y = pd.concat(rtogs_y, sort=True)
        df = pd.concat([df, df_y], axis=1)

    # aggregate labels
    df['survival_any'] = (df['survival_15year'] == 1) | (df['survival_10year'] == 1) | (df['survival_5year'] == 1)
    df['biochemical_failure_any'] = (df['biochemical_failure_15year'] == 1) | (df['biochemical_failure_10year'] == 1) | (df['biochemical_failure_5year'] == 1)
    df['distant_met_any'] = (df['distant_met_15year'] == 1) | (df['distant_met_10year'] == 1) | (df['distant_met_5year'] == 1)
    df['disease_specific_survival_any'] = (df['disease_specific_survival_15year'] == 1) | (df['disease_specific_survival_10year'] == 1) | (df['disease_specific_survival_5year'] == 1)

    df['15year_any'] = (df['distant_met_15year'] == 1) | (df['biochemical_failure_15year'] == 1) | (df['disease_specific_survival_15year'] == 1) | (df['survival_15year'] == 1)
    df['10year_any'] = (df['distant_met_10year'] == 1) | (df['biochemical_failure_10year'] == 1) | (df['disease_specific_survival_10year'] == 1) | (df['survival_10year'] == 1)
    df['5year_any'] = (df['distant_met_5year'] == 1) | (df['biochemical_failure_5year'] == 1) | (df['disease_specific_survival_5year'] == 1) | (df['survival_5year'] == 1)

    df['survival_any'] = df['survival_any'].astype(int)
    df['biochemical_failure_any'] = df['biochemical_failure_any'].astype(int)
    df['distant_met_any'] = df['distant_met_any'].astype(int)
    df['biochemical_failure_any'] = df['biochemical_failure_any'].astype(int)
    df['15year_any'] = df['15year_any'].astype(int)
    df['10year_any'] = df['10year_any'].astype(int)
    df['5year_any'] = df['5year_any'].astype(int)

    # create validation split for quilt
    idxs = np.arange(len(df))
    df = df.reset_index()
    idxs = df[(df.test_set == 0) & (~df.quilt_path.isna())].index.tolist()
    np.random.shuffle(idxs)
    val_len = int(len(idxs) * 0.2)
    df['split'] = 'train'
    df.loc[df.test_set == 1, 'split'] = 'test'
    df.loc[idxs[:val_len], 'split'] = 'valid'

    # create ID
    df['id'] = df.apply(lambda x: f"{x.sn}_{x.cn_deidentified}", axis=1) 

    # save  
    parsed_emr = {}
    parsed_emr['cat_idxs'] = cat_idxs 
    parsed_emr['cat_dims'] = cat_dims
    parsed_emr['cat_emb_dims'] = cat_emb_dims
    parsed_emr['input_dim'] = len(feature_cols)
    parsed_emr['features'] = feature_cols
    parsed_emr['df_full'] = df 

    # create 5 fold validation split for train
    cols_to_keep = feature_cols + img_feature_cols + ['distant_met_5year', 'quilt_path', 'split', 'id']
    for idx, split in zip([1,0], ['test', 'train']): 
        df_split = df[df.test_set == idx]

        if split == 'train': 
            # create 5 fold splits
            idxs = np.arange(len(df_split))
            np.random.shuffle(idxs)
            for split_num, split_idxs in enumerate(np.array_split(idxs, 5)):
                split_num += 1
                df_split[f'Split{split_num}'] = 0
                df_split[f'Split{split_num}'].iloc[split_idxs] = 1
                cols_to_keep += [f'Split{split_num}']
        
        parsed_emr[f'df_{split}'] = df_split[cols_to_keep]

    pickle.dump(parsed_emr, open(PARSED_EMR_2, 'wb'))
                
if __name__ == '__main__': 
    main()