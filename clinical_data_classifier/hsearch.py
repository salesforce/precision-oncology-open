"""Script to hyperparameter search a model on a given task and testset

Example call:
python hsearch.py \
        --outcome=distant_met \
        --outcome_period=5 \
        --testset="testset3" \
        --quilt_dir="/export/medical_ai/ucsf/ssl_rtog/moco/model_R50_b=256_lr=0.03_pg4plus/features/RTOG-{}_quilts/" \
        --results_dir="./tmp" \
        --num_trials=3 \
"""
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import numpy as np
import shap
from experimental_utils import CORAL
from sklearn.model_selection import KFold
import random
import argparse
import re
import os
from datetime import datetime

from utils import plot_ss_curve
from rtog_helper import RTOG, rtog_from_study_number

from rtog_image_helper import ImageLoader
from histopathology_image_helper import *
from rtog_constants import drop_confounding_variables
from rtog_constants import is_categorical

from tqdm import tqdm
from catboost import Pool
from catboost import CatBoostClassifier
import sys
print(sys.argv)

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--outcome', default='distant_met', type=str,
                    help='outcome. e.g. distant_met')
parser.add_argument('--outcome_period', default='5', type=str,
                    help='outcome time period in years. e.g. 5')
parser.add_argument('--testset', default='testset2',
                    type=str,
                    choices=['testset2', 'testset3', 'RTOG-9202', 'RTOG-9413', 'RTOG-9408', 'RTOG-9910', 'RTOG-0126'],
                    help='The testset. Options are testset3, RTOG-9202, RTOG-9413, RTOG-9408, RTOG-9910, RTOG-0126')
parser.add_argument('--quilt_dir', type=str, default="/export/medical_ai/ucsf/ssl_rtog/moco/model_R50_b=256_lr=0.03_pg4plus/features/RTOG-{}_quilts/",
                    help='The directory containing directories of the form RTOG-{}_quilt, containing the quilts for each study')
parser.add_argument('--results_dir', default="/tmp", type=str,
                    help='Dir to output results. File output name is of the form outcomeyear_testset_quiltdir_num-quilts_datetime-generated.txt')
parser.add_argument('--num_trials', default=100, type=int,
                    help='Number of random parameter combinations to try.')
# parser.add_argument('--y_var', default='distant_met_5year', type=str,
#                     help='outcome. e.g. distant_met_5year')
args = parser.parse_args()

print(args)
# import IPython
# IPython.embed()


def printc(df):
    """helper function. prints all the columns of a pandas dataframe."""
    prev = pd.options.display.max_columns
    prev_r = pd.options.display.max_rows
    pd.options.display.max_columns = None
    pd.options.display.max_rows = 20
    display(df)
    pd.options.display.max_columns = prev
    pd.options.display.max_rows = prev_r


def random_catboost_params():
    """returns random params."""
    depth = random.choice(range(2,10))
    lr = 0.1 ** (np.random.uniform() * 5) # sample between 0.1 and 0.00001
    return depth, lr


# define a specific catboost model
catboost_depth, catboost_lr = random_catboost_params()
def make_catboost():
    params_clinimage = {
        'iterations':3000,
        'early_stopping_rounds':3000,
        'learning_rate':0.003, #0.008
        'depth':5,
        'verbose':False,
        'thread_count':4,
        'one_hot_max_size':10, # our data's categorical features have max of 6
        'train_dir' : '/tmp/catboost_info',
        'eval_metric':'AUC',
    }
    params = params_clinimage
    params['depth'], params['learning_rate'] = random_catboost_params()
    model = CatBoostClassifier(**params)
    return model


def index_into(df_data, df_index, col='sn', vals=[]):
    assert(type(vals) == list)
    if not vals:
        return df_data
    bool_vec = False * np.zeros(len(df_index[col]))
    for v in vals:
        bool_v = (df_index[col] == v)
        if not any(bool_v):
            print("Warning: {} has no values".format(v))
        bool_vec = bool_vec | bool_v
    return df_data[bool_vec].copy()


def compute_auc(gt, probs):
    fpr, tpr, thresholds = metrics.roc_curve(gt, probs, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print("AUC={:0.3f}".format(auc))
    acc_bal = max([np.mean((ss, sp)) for ss, sp in zip(tpr, 1-fpr)])
    print("Best balanced accuracy: {:0.3f}".format(acc_bal))
    return auc


# set endpoint
y_var_list = [
    'distant_met',
    'distant_met_5year',
    'distant_met_10year',
    'biochemical_failure_5year',
    'biochemical_failure_10year',
    'disease_specific_survival_10year',
    'survival_10year',
]
y_var = args.outcome + "_{}year".format(args.outcome_period)
if y_var not in y_var_list:
    raise ValueError("outcome_period not supported: {}. Options {}".format(y_var, y_var_list))
else:
    print("Selected outcome: {}".format(y_var))

#Load NRG Data
studies = {}
features_dir = args.quilt_dir
for sn in ['9202', '9408', '9413', '9910', '0126']:
    print(" ----- Loading RTOG-{} ----- ".format(sn))
    studies[sn] = rtog_from_study_number(sn, standardize=True)
    feat_dir = features_dir.format(sn)

    # Add the feature quilts if available
    if os.path.exists(feat_dir):
        feature_quilt_paths = sorted([os.path.join(feat_dir,i) for i in os.listdir(feat_dir) if '.pkl' in i])
        cn_deids = np.array([int(i.split("/")[-1].split('_')[0]) for i in feature_quilt_paths])
        print("Loaded {} feature-quilts from {}".format(len(feature_quilt_paths), feat_dir))

        studies[sn].df['featquilt'] = ''
        for id, path in zip(cn_deids, feature_quilt_paths):
            studies[sn].df.loc[studies[sn].df['cn_deidentified'] == id, 'featquilt'] = path
    print(" -------------------------- ")

# Merge NRG data into two pd.DataFrames (df_X, df_y)
rtogs_X = []
rtogs_y = []
for sn, rtog in studies.items():
    rx, ry, rm = rtog.get_Xy(y_var=y_var, make_binary=True)
    rtogs_X.append(rx.df)
    rtogs_y.append(ry.df)
df_X_full = pd.concat(rtogs_X, sort=True)
df_y = pd.concat(rtogs_y, sort=True)

# # Add Image Features
loader = ImageLoader()
loader.load_feature_matrix(df_X_full)
df_X = df_X_full.drop(columns='featquilt')

# Drop bad variables
df_X = drop_confounding_variables(df_X)

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

# Display Results
print("\nX Matrix:")
print("Nan fraction: {}".format(np.sum(np.sum(df_X.isnull())) / np.prod(df_X.shape)))
print(df_X)
print("y matrix")
print(df_y)

if args.testset == 'testset2':
    # Use fixed test set.
    train_size=0.8
    train_size=0.8
    np.random.seed(2)

    X_test = []
    X_train = []
    y_test = []
    y_train = []

    for sn in df_X_full['sn'].unique():
        sn_y = df_y[df_X_full['sn'] == sn].copy()
        sn_x = df_X[df_X_full['sn'] == sn].copy()
        idxs = np.array(range(len(sn_y)))
        np.random.shuffle(idxs)
        cutoff = int(len(sn_y) * train_size)
        X_train.append(sn_x.iloc[idxs[:cutoff]])
        y_train.append(sn_y.iloc[idxs[:cutoff]])
        X_test.append(sn_x.iloc[idxs[cutoff:]])
        y_test.append(sn_y.iloc[idxs[cutoff:]])
    X_test = pd.concat(X_test)
    y_test = pd.concat(y_test)
    X_train = pd.concat(X_train)
    y_train = pd.concat(y_train)

    np.random.seed() #unset random seed

elif args.testset == 'testset3':
    # Use fixed test set.
    train_size=0.8
    train_size=0.8
    np.random.seed(3)

    X_test = []
    X_train = []
    y_test = []
    y_train = []

    for sn in df_X_full['sn'].unique():
        sn_y = df_y[df_X_full['sn'] == sn].copy()
        sn_x = df_X[df_X_full['sn'] == sn].copy()
        idxs = np.array(range(len(sn_y)))
        np.random.shuffle(idxs)
        cutoff = int(len(sn_y) * train_size)
        X_train.append(sn_x.iloc[idxs[:cutoff]])
        y_train.append(sn_y.iloc[idxs[:cutoff]])
        X_test.append(sn_x.iloc[idxs[cutoff:]])
        y_test.append(sn_y.iloc[idxs[cutoff:]])
    X_test = pd.concat(X_test)
    y_test = pd.concat(y_test)
    X_train = pd.concat(X_train)
    y_train = pd.concat(y_train)

    np.random.seed() #unset random seed

elif args.testset.split('-')[0] == 'RTOG':
    studies_ = [9202, 9413, 9408, 126, 9910]

    studies_test = [int(args.testset.split('-')[1])]
    studies_train = list(set(studies_) - set(studies_test))

    X_test_full = index_into(df_X_full, df_X_full, col='sn', vals=studies_test)
    X_test = index_into(df_X, df_X_full, col='sn', vals=studies_test)
    y_test = index_into(df_y, df_X_full, col='sn', vals=studies_test)

    X_train = index_into(df_X, df_X_full, col='sn', vals=studies_train)
    y_train = index_into(df_y, df_X_full, col='sn', vals=studies_train)

else:
    raise ValueError("Testset unknown: {}".format(args.testset))


train_pool = Pool(X_train, y_train, cat_features=categorical)
validate_pool = Pool(X_test, y_test, cat_features=categorical)

print("Train set characteristics:")
for val in np.unique(y_train):
    print("y_train={}: {} instances".format(val, np.sum(y_train == val)))
print()

print("Test set characteristics:")
for val in np.unique(y_test):
    print("y_test={}: {} instances".format(val, np.sum(y_test == val)))
print()
print(X_test)
print(y_test)

results = []
for i in range(args.num_trials):
    print("Trial {}".format(i))
    model = make_catboost()
    print(model.get_params())
    model.fit(X_train,
              y_train,
              cat_features=categorical,
              eval_set=(X_test, y_test),
              plot=False
             ) #chooses best model for this eval set.
    auc = compute_auc(y_test, model.predict_proba(X_test)[:,1])
    results.append((auc, model.get_params()))

results = sorted(results, key=lambda x: x[0])
print(results)

if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)
results_file = os.path.join(args.results_dir,
                            '{}_{}_{}_{}.txt'.format(y_var, args.testset, re.sub("[/. ]", "", args.quilt_dir), re.sub("[. ]", "", str(datetime.now())))
                            )
with open(results_file, 'w') as f:
    prefix = ""
    for r in results:
        f.write(prefix)
        f.write(str(r))
        prefix = "\n"
