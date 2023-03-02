"""Computes Table of AUCs
Endpoints x timeframe
e.g.
DM, PSA, PCSS, OS x 5,10,15 yr
"""

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import metrics
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import random

from rtog_helper import accuracy_score_balanced
from rtog_helper import RTOG



if __name__ == "__main__":

    # User-defined variables
    train_size=0.8
    study_number = '9202'
    study_number = '9408'
    impute_x = "default_class"
    oversample_function = RandomOverSampler()
    hyperparameter_tune = False
    optimization_dict = {
        'max_depth': [6,8,10], #default: 6
        'n_estimators': [50,100,200],
        'min_child_weight' : [1,2], #default: 1
    }
    y_vars = [
        'distant_met',
        'biochemical_failure',
        'survival',
        'disease_specific_survival'
    ]
    time_frames_years = [5,10,15]

    # Calculate AUC for each y_var at each time point
    study_path = RTOG.gcp_baseline_paths[study_number]
    rtog = RTOG(filename=study_path, study_number=study_number, file_type='excel')
    results = {}
    print("----- Variables -----")
    print("train_size={}".format(train_size))
    print("study_number={}".format(study_number))
    print("impute_x={}".format(impute_x))
    print("hyperparameter_tune={}".format(hyperparameter_tune))
    if hyperparameter_tune:
        print("Optimization dict:")
        print(optimization_dict)

    # Drop potentially confounding variables:
    rtog = rtog.drop(columns='cn_deidentified')

    print("----- Begin AUC Compute -----")
    for y_var0 in y_vars:
        results[y_var0] = {}
        for yr in time_frames_years:
            y_var = y_var0 + "_{}year".format(yr)
            rtog_X, rtog_y = rtog.get_Xy(y_var=y_var, make_binary=True, impute_x=impute_x)
            rtog_X = rtog_X.drop(columns=[c for c in rtog_X.df.columns if 'salvage' in c])
            X = rtog_X.df.to_numpy()
            y = rtog_y.df.to_numpy().flatten()
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, stratify=y)
            X_train, y_train = oversample_function.fit_resample(X_train, y_train)

            model = XGBClassifier()
            if hyperparameter_tune:
                model = GridSearchCV(model, optimization_dict, scoring='accuracy', verbose=1)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            predictions = [round(value) for value in y_pred]
            accuracy = accuracy_score(y_test, predictions)
            accuracy_balanced, per_class_sensitivities = accuracy_score_balanced(y_test, predictions)

            y_probs = model.predict_proba(X_test)[:,1]
            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_probs, pos_label=1)
            auc = metrics.auc(fpr, tpr)

            print("y-var: {}, auc: {:0.3f}, len(y_test)={}".format(y_var, auc, len(y_test)))
            results[y_var0][yr] = auc
    print("----- End AUC Compute -----\n")

    # Print Table results
    print("----- Results -----")
    df = pd.DataFrame.from_dict(results)
    print(df)
