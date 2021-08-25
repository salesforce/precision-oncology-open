import catboost
from tqdm import tqdm
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import KFold
from rtog_constants import is_categorical


def compute_best_auc_by_data_ablation(model, train_pool, X_train, y_train, validate_pool, max_removal=2000, batch_size=250, verbose=True):
    """Sequentially removes low-quality data from train_pool, to maximize AUC
    """
    indices, scores = model.get_object_importance(
        validate_pool,
        train_pool,
        update_method='SinglePoint', # The fastest and least accurate. 'AllPoints' is slowest and most accurate.
        importance_values_sign='Positive' # Positive values means that the optimized metric
                                          # value is increase because of given train objects.
                                          # So here we get the indices of bad train objects.
    )
    def _train_and_return_score(train_indices):
        model.fit(X_train[train_indices], y_train[train_indices], cat_features=categorical, eval_set=validate_pool)
        metric_value = model.eval_metrics(validate_pool, ['AUC'])['AUC'][-1]
        return metric_value

    train_indices = np.full(X_train.shape[0], True)
    auc = _train_and_return_score(train_indices)
    if verbose:
        s = 'AUC on validation datset when {} harmful objects from train are dropped: {}'
        print(s.format(0, auc))
    auc_values = [auc]
    for batch_start_index in range(0, max_removal, batch_size):
        train_indices[indices[batch_start_index:batch_start_index + batch_size]] = False
        auc = _train_and_return_score(train_indices)
        if verbose:
            s = 'AUC on validation datset when {} harmful objects from train are dropped: {}'
            print(s.format(batch_start_index + batch_size, auc))
        auc_values.append(auc)

    # Re-train model on the best ablated set.
    idx_best = np.argmax(auc_values)
    train_indices = np.full(X_train.shape[0], True)
    train_indices[indices[0 : idx_best * batch_size]] = False
    auc = _train_and_return_score(train_indices)
    return max(auc_values)


def average_auc(X, y, num_runs=20, model_func=CatBoostClassifier,
                cat_dims=[], data_ablation=False, verbose=False, domain_shift=False, ablation_fraction=0.05):
    """Prints the average auc +- std, averaged over num_runs
    Assume df_y is binary
    Assume positive label for y is 1
    """
    print(ablation_fraction)
    assert len(np.unique(df_y.values) == 2)
    auc_values = []
    if data_ablation:
        print("Ablating low-quality data. be patient..")

    for i in tqdm(range(num_runs)):
        # Split into train/test, load model
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, stratify=y)
        if domain_shift:
            X_train = CORAL(X_train, X_test, categorical=cat_dims)
        X_train, y_train = oversample_function.fit_resample(X_train, y_train) # Unclear if this helps
        model = model_func()

        # Maybe ablate X_train's low-quality data points
        if data_ablation:
            # Generalizable data ablation
            X_train_p, y_train_p = filter_lowvalue_data(X_train, y_train, model, cat_dims, fraction=ablation_fraction)
            model.fit(X_train_p, y_train_p, cat_features=cat_dims, eval_set=(X_test, y_test))

            y_probs = model.predict_proba(X_test)[:,1]
            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_probs, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            auc_values.append(auc)
        else:
            model.fit(X_train, y_train, cat_features=cat_dims, eval_set=(X_test, y_test))
            y_probs = model.predict_proba(X_test)[:,1]
            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_probs, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            auc_values.append(auc)

        print("Run {} AUC: {}".format(i, auc))

    print("AUC ({} runs): {:.3f} +- {:.3f}".format(num_runs,np.mean(auc_values), np.std(auc_values)))
    return auc_values


def score_dataset(X, y, model, n_splits=5, categorical=[]):
    """Cross-validates (X,y) to evaluate the quality of its data points given the model

    Args:
        D (numpy.array): 2D matrix of data points, N x F, with N points of F features.
        model (catboost model): easiest to pass with make_catboost() function
    """
    assert len(y) == X.shape[0]
    S = np.zeros(X.shape[0])
    train_fraction = 0.8
    kf = KFold(n_splits=n_splits, shuffle=False)
    print("Scoring dataset")
    for train_ind, val_ind in tqdm(kf.split(X)):
        R, r = X.iloc[train_ind], y.iloc[train_ind]
        V, v = X.iloc[val_ind], y.iloc[val_ind]
        model.fit(R, r, cat_features=categorical, eval_set=(V, v))
        idxs, scores = model.get_object_importance(
            Pool(V, v, cat_features=categorical),
            Pool(R, r, cat_features=categorical),
            update_method='SinglePoint', # The fastest and least accurate. 'AllPoints' is slowest and most accurate.
            importance_values_sign='Positive' # Positive values means that the optimized metric
                                              # value is increase because of given train objects.
                                              # So here we get the indices of bad train objects.
        )
        S[train_ind[idxs]] += scores
    return S


def clip_topk(X, y, scores, k=20):
    """Removes the (X,y) points with the highest k scores
    """
    assert X.shape[0] == len(scores)
    assert X.shape[0] == len(y)
    assert k < len(y)
    scores = -np.array(scores)

    indx = np.argsort(scores)[:-k]
    indx_clipped = np.argsort(scores)[-k:]
    X_keep, y_keep = X.iloc[indx], y.iloc[indx]
    X_clipped, y_clipped = X.iloc[indx_clipped], y.iloc[indx_clipped]
    return (X_keep, y_keep), (X_clipped, y_clipped)


def filter_lowvalue_data(X, y, model, categorical, fraction=0.10):
    """ Filter the worst fraction from X, y, using a catboost model. Categorical are the categorical feature indices.
    """
    assert len(y) == X.shape[0]
    if int(fraction * X.shape[0]) == 0:
        print("Warning: fraction {} too low for dataset size {}.".format(fraction, X.shape))
        return X, y

    scores = score_dataset(X, y, model, n_splits=5, categorical=categorical)
    (X_kept, y_kept), (_, _) = clip_topk(X, y, scores, k=int(fraction * X.shape[0]))
    return X_kept, y_kept


def filter_for_rows_with_image_data(x, y):
    im_feats = ["f{}".format(i) for i in range(128)]
    kept_x, kept_y = [], []
    for i in range(len(x)):
        row_x = x.iloc[i]
        row_y = y.iloc[i]
        vals = row_x[im_feats]
        if all(vals == 0.0):
            continue
        else:
            kept_x.append(row_x)
            kept_y.append(row_y)
    return pd.DataFrame(kept_x), pd.DataFrame(kept_y)


def filter_for_columns_of_image_features(x):
    cols = x.columns
    im_feats = ["f{}".format(i) for i in range(128)]
    cols = set(cols) - set(im_feats)
    cols = list(cols)
    return x.drop(columns=cols)


def filter_for_columns_of_image_features_and_nccn_vars(x):
    cols = x.columns
    feats = ['baseline_psa', 'gleason_combined', 'gleason_primary', 'gleason_secondary', 'tstage']
    feats.extend(["f{}".format(i) for i in range(128)])
    cols = set(cols) - set(feats)
    cols = list(cols)
    x = x.drop(columns=cols)
    cat_vars = is_categorical(x.columns)
    cat_vars = np.where(cat_vars)[0]
    return x, cat_vars


def filter_out_columns_of_image_features(x):
    im_feats = ["f{}".format(i) for i in range(128)]
    return x.drop(columns=im_feats)


def drop_low_shap_features(df_X, model, N=5):
    """Computes absolute values of shapley scores for each feature(column) of df_X, and drops the lowest N.
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(df_X)
    cols_to_drop = df_X.columns[np.argsort(np.mean(np.abs(shap_values.values), axis=0))][:N]
    print("Dropping feature columns: {}".format(cols_to_drop))
    return df_X.drop(columns=cols_to_drop)


class NCCN(object):
    def __init__(self):
#       self.probs_per_risk_group_5yrDM = np.array([0.05, 0.10, 0.20]) # Low, Medium, High Risk
#       self.probs_per_risk_group_5yrDM = np.array([0.01, 0.03, 0.12]) # Low, Medium, High Risk
#       self.probs_per_risk_group_10yrDM = np.array([0.02, 0.06, 0.12]) # Low, Medium, High Risk
#       self.probs_per_risk_group_5yrPSA = np.array([0.05, 0.15, 0.25]) # Low, Medium, High Risk
#       self.probs_per_risk_group_10yrPSA = np.array([0.10, 0.20, 0.30]) # Low, Medium, High Risk
        self.risk_group_probs = {
            'distant_met_5year' : np.array([0.01, 0.03, 0.12]),
            'distant_met_10year' : np.array([0.02, 0.06, 0.12]),
            'distant_met_15year' : np.array([0.03, 0.09, 0.12]), # note: this isn't in RTOG papers.
            'distant_met_25year' : np.array([0.03, 0.09, 0.12]), # note: this isn't in RTOG papers.
            'biochemical_failure_5year' : np.array([0.05, 0.15, 0.25]),
            'biochemical_failure_10year' : np.array([0.10, 0.20, 0.30]),
            'disease_specific_survival_10year' : 1-np.array([0.99, 0.96, 0.90]),
            'survival_10year' : 1-np.array([0.67, 0.66, 0.65]),
        }


    def low_risk(self, df):
        """Returns 1 if the patients are low-risk, zero otherwise
        """
        return pd.DataFrame(
            (df['baseline_psa'] < 10).values &
            (df['gleason_combined'] <= 6).values &
            (df['tstage'] <= 2).values,
            columns=['low_risk']
        )


    def high_risk(self, df):
        """Returns 1 if the patients are high-risk, zero otherwise
        """
        return pd.DataFrame(
            (df['baseline_psa'] > 20).values |
            (df['gleason_combined'] >= 8).values |
            (df['tstage'] >= 3).values,
            columns=['high_risk']
        )


    def intermediate_risk(self, df):
        """Returns 1 if the patients are intermediate risk, zero otherwise
        """
        df_low = self.low_risk(df)
        df_high = self.high_risk(df)
        df_int = pd.DataFrame(
            ~df_low.values & ~df_high.values,
            columns=['inter_risk']
        )
        return df_int

    def risk_group(self, df):
        risk_group = np.concatenate([
            self.low_risk(df).values.reshape(-1,1),
            self.intermediate_risk(df).values.reshape(-1,1),
            self.high_risk(df).values.reshape(-1,1),
            ], axis=1)
        return risk_group

    def predict_proba(self, df, outcome='distant_met_5year'):
        risk_group = self.risk_group(df)
        probs = np.sum(risk_group * self.risk_group_probs[outcome], axis=1)
        return probs
