import pickle
import numpy as np
import pandas as pd
import os 
import sys
sys.path.append(os.getcwd())

from twotowers.constants import *

PREFIX = "/export/medical-ai/ucsf/ssl_rtog/moco/"
FEATURES_DIR = PREFIX + "model_R50_b=256_lr=0.03_pg4plus/features/RTOG-{}_quilts/"
STUDY_NUMS = ['9202', '9408', '9413', '9910', '0126']

def main():

    original_emr = pickle.load(open(ORIGINAL_EMR, 'rb'))
    df_X_full = original_emr['df_X_full']
    X_train = original_emr['X_train']
    X_test = original_emr['X_test']

    img_feature_cols = [f'f{i}' for i in range(128)]
    X_train.drop(columns=img_feature_cols, inplace=True) 
    X_test.drop(columns=img_feature_cols, inplace=True)


    nan_cols = X_train.columns[X_train.eq('nan').any()].tolist() + ['rx']
    X_train = X_train.fillna('NONE')
    X_train = X_train.replace({'nan': None})
    X_train[nan_cols] = X_train[nan_cols].apply(pd.to_numeric, axis = 1)
    X_train = X_train.fillna('NONE')

    X_test = X_test.fillna('NONE')
    X_test = X_test.replace({'nan': None})
    X_test[nan_cols] = X_test[nan_cols].apply(pd.to_numeric, axis = 1)
    X_test = X_test.fillna('NONE')
    

    df_X_full = df_X_full.fillna('NONE')
    df_X_full = df_X_full.replace('nan', 'NONE')

    index_columns = X_train.columns.tolist()
    to_keep_columns = index_columns + ['cn_deidentified', 'sn']

    print(f'X_trian before merge: {X_train.shape}')
    X_train = X_train.set_index(index_columns)
    df_X_full = df_X_full.set_index(index_columns)
    X_train = X_train.join(df_X_full, how='left')
    df_X_full = df_X_full.reset_index()
    X_train = X_train.reset_index()
    print(f'X_trian after merge: {X_train.shape}\n')

    print(f'X_test before merge: {X_test.shape}')
    X_test = X_test.set_index(index_columns)
    df_X_full = df_X_full.set_index(index_columns)
    X_test = X_test.join(df_X_full, how='left')
    df_X_full = df_X_full.reset_index()
    X_test = X_test.reset_index()
    print(f'X_test after merge: {X_test.shape}\n')

    import pdb; pdb.set_trace()

    # Load Data
    for sn in STUDY_NUMS:

        feat_dir = FEATURES_DIR.format(sn)

        # Add the feature quilts if available
        if os.path.exists(feat_dir):
            feature_quilt_paths = sorted([os.path.join(feat_dir,i) for i in os.listdir(feat_dir) if '.pkl' in i])
            cn_deids = np.array([int(i.split("/")[-1].split('_')[0]) for i in feature_quilt_paths])
            print("Loaded {} feature-quilts\n{}\n".format(len(feature_quilt_paths), feat_dir))
            import pdb; pdb.set_trace()

            #studies[sn].df['featquilt'] = ''
            #for id, path in zip(cn_deids, feature_quilt_paths):
            #    studies[sn].df.loc[studies[sn].df['cn_deidentified'] == id, 'featquilt'] = path
        else: 
            print(f'Features dir for study {sn} does not exist: {feat_dir}')




if __name__ == '__main__':
    main()