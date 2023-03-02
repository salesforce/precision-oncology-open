"""Storage for a number of functions used for experimentation.
"""


from scipy.linalg import fractional_matrix_power as matrix_power
from catboost import CatBoostClassifier
import numpy as np

# Define a specific catboost model
def make_catboost(params='params_fast'):
    """Return catboost model w/pre-specified parameters.
    """
    menu = dict(
    params_fast = {'iterations':150, 'learning_rate':0.1, 'depth':2, 'verbose':False, 'thread_count':4, 'one_hot_max_size':10, 'train_dir' : '/tmp/catboost_info', 'eval_metric':'AUC',},
    params_clin = {'iterations':4500, 'early_stopping_rounds':400,  'learning_rate':0.01, 'depth':4, 'verbose':False, 'thread_count':4, 'one_hot_max_size':10, 'train_dir' : '/tmp/catboost_info','eval_metric':'AUC',},
    params_clinimage = {'iterations':8000, 'early_stopping_rounds':1000, 'learning_rate':0.003, 'depth':5, 'l2_leaf_reg' : 10, 'verbose':False, 'thread_count':4, 'one_hot_max_size':10, 'train_dir' : '/tmp/catboost_info','eval_metric':'AUC',},
    params_clinimage_multiclass = {'iterations':8000, 'early_stopping_rounds':1000, 'learning_rate':0.003, 'depth':5, 'l2_leaf_reg' : 10, 'verbose':False, 'thread_count':4, 'one_hot_max_size':10, 'train_dir' : '/tmp/catboost_info', 'eval_metric' : 'Accuracy'},
    params_clinimage2 = {'iterations':15000, 'early_stopping_rounds':6000, 'learning_rate':0.0007, 'depth':6, 'verbose':False, 'thread_count':4, 'one_hot_max_size':10, 'train_dir' : '/tmp/catboost_info','eval_metric':'AUC',},
    params_image = {'iterations':4500, 'early_stopping_rounds':400,  'learning_rate':0.03, 'depth':2, 'verbose':False, 'thread_count':4, 'one_hot_max_size':10, 'train_dir' : '/tmp/catboost_info','eval_metric':'AUC',},
#   params_hsearch_image = {'early_stopping_rounds': 3000, 'eval_metric': 'AUC', 'train_dir': '/tmp/catboost_info', 'one_hot_max_size': 10, 'verbose': False, 'thread_count': 4, 'depth': 6, 'learning_rate': 0.0054927033844122015, 'iterations': 3000},
    params_hsearch_image = {'early_stopping_rounds': 300, 'eval_metric': 'AUC', 'train_dir': '/tmp/catboost_info', 'one_hot_max_size': 10, 'verbose': False, 'thread_count': 4, 'depth': 6, 'learning_rate': 0.22731900285470796, 'iterations': 3000},

    #Hsearch. AUC = 0.763, using meanpooled image data.
    params_hsearch_5yrdm = {'early_stopping_rounds': 300, 'eval_metric': 'AUC', 'train_dir': '/tmp/catboost_info', 'one_hot_max_size': 10, 'verbose': False, 'thread_count': 4, 'depth': 2, 'learning_rate': 0.3412988615159519, 'iterations': 3000},
    params_hsearch_10yrdm = {'early_stopping_rounds': 300, 'eval_metric': 'AUC', 'train_dir': '/tmp/catboost_info', 'one_hot_max_size': 10, 'verbose': False, 'thread_count': 4, 'depth': 9, 'learning_rate': 0.03263624719116104, 'iterations': 3000},
    params_hsearch_5yrPSA = {'early_stopping_rounds': 300, 'eval_metric': 'AUC', 'train_dir': '/tmp/catboost_info', 'one_hot_max_size': 10, 'verbose': False, 'thread_count': 4, 'depth': 9, 'learning_rate': 0.044791205905705365, 'iterations': 3000},
    params_hsearch_10yrPSA = {'early_stopping_rounds': 3000, 'eval_metric': 'AUC', 'train_dir': '/tmp/catboost_info', 'one_hot_max_size': 10, 'verbose': False, 'thread_count': 4, 'depth': 8, 'learning_rate': 0.014632309809680622, 'iterations': 3000},

    # Hsearch Pathology Ablation
    params_hsearch_5yrdm_path_clin = {'early_stopping_rounds': 3000, 'eval_metric': 'AUC', 'train_dir': '/tmp/catboost_info', 'one_hot_max_size': 10, 'verbose':False, 'thread_count':4, 'depth':5, 'learning_rate':0.24875295849212564, 'iterations':3000},
    params_hsearch_5yrdm_path_nccn = {'early_stopping_rounds': 3000, 'eval_metric': 'AUC', 'train_dir': '/tmp/catboost_info', 'one_hot_max_size': 10, 'verbose': False, 'thread_count': 4, 'depth': 2, 'learning_rate': 0.0006663231153979403, 'iterations': 3000},

    # distant_met_5year optimal params per study
    params_9408 = {'early_stopping_rounds': 300, 'eval_metric': 'AUC', 'train_dir': '/tmp/catboost_info', 'one_hot_max_size': 10, 'verbose': False, 'thread_count': 4, 'depth': 9, 'learning_rate': 0.7979003375034982, 'iterations': 3000},
    params_9202 = {'early_stopping_rounds': 300, 'eval_metric': 'AUC', 'train_dir': '/tmp/catboost_info', 'one_hot_max_size': 10, 'verbose': False, 'thread_count': 4, 'depth': 3, 'learning_rate': 0.048986745689979516, 'iterations': 3000},

    # Params for predictive biomarkers. Note that only 9408 and 9202 are valid trials for this.
    params_9202_pred_dm = {'iterations':15000, 'early_stopping_rounds':2000, 'learning_rate':0.0007, 'depth':6, 'verbose':False, 'thread_count':4, 'one_hot_max_size':10, 'train_dir' : '/tmp/catboost_info','eval_metric':'AUC',},
    params_9408_pred_dm = {'early_stopping_rounds': 600, 'eval_metric': 'AUC', 'train_dir': '/tmp/catboost_info', 'one_hot_max_size': 10, 'verbose': False, 'thread_count': 4, 'depth': 9, 'learning_rate': 0.7979003375034982, 'iterations': 3000},
    )

    print("Making Catboost: {}".format(params))
    params = menu[params]
    model = CatBoostClassifier(**params)
    return model


def CORAL(X_train, X_test, categorical=[]):
    """Returns X_source*, domain-shifted to covariance-match X_target.

    Assumps both are in the form N x F, with N entries and F features.
    """
    assert X_train.shape[1] == X_test.shape[1]
    def fix(x):
        x = x.astype(np.float)
        x[np.isnan(x)] = 0
        return x

    # We perform correlation shift only on numerical features, excluding categorical ones.
    categorical = set(categorical)
    numerical = set(range(X_train.shape[1])) - categorical
    categorical, numerical = np.array(list(categorical)), np.array(list(numerical))
    X_source = fix(X_train[:, numerical])
    X_target = fix(X_test[:, numerical].copy())

    C_s = np.cov(X_source.T) + np.eye(X_source.shape[1])
    C_t = np.cov(X_target.T) + np.eye(X_target.shape[1])
    D_s = np.matmul(X_source, matrix_power(C_s, -0.5))
    D_ss = np.matmul(D_s, matrix_power(C_t, 0.5))

    X_train_shifted = X_train.copy()
    X_train_shifted[:, numerical] = D_ss

    assert np.all(X_train[:, categorical] == X_train_shifted[:, categorical])
    return X_train_shifted


# This doesn't seem to help
def filter_nanfrac_rowwise(X, y, nanthresh):
    """Returns the data with row-wise nanfraction in X that is greater than nanthresh.
    """
    if type(X) == pd.DataFrame:
        assert type(y) == np.DataFrame
        nanfrac_rowwise = np.sum(X.isnull(), axis=1) / X.shape[1]
        Xf = X[nanfrac_rowwise > nanthresh]
        yf = y[nanfrac_rowwise > nanthresh]
    elif type(X) == np.ndarray:
        assert type(y) == np.ndarray
        nanfrac_rowwise = np.sum(pd.DataFrame(X).isnull(), axis=1) / X.shape[1]
        Xf = X[nanfrac_rowwise > nanthresh, :]
        yf = y[nanfrac_rowwise > nanthresh]
    else:
        raise TypeError("Type not supported. X={}, y={}".format(type(X), type(y)))
    return Xf, yf

# This doesn't seem to help
def filterOut_nanfrac_rowwise(X, y, nanthresh):
    """Returns the data with row-wise nanfraction in X that is less than or equal to nanthresh.
    """
    if type(X) == pd.DataFrame:
        assert type(y) == np.DataFrame
        nanfrac_rowwise = np.sum(X.isnull(), axis=1) / X.shape[1]
        Xf = X[nanfrac_rowwise <= nanthresh]
        yf = y[nanfrac_rowwise <= nanthresh]
    elif type(X) == np.ndarray:
        assert type(y) == np.ndarray
        nanfrac_rowwise = np.sum(pd.DataFrame(X).isnull(), axis=1) / X.shape[1]
        Xf = X[nanfrac_rowwise <= nanthresh, :]
        yf = y[nanfrac_rowwise <= nanthresh]
    else:
        raise TypeError("Type not supported. X={}, y={}".format(type(X), type(y)))
    return Xf, yf

#Didn't help
def filterOut_missing(df_X, df_y, cols=[]):
    """Filters out row-entries that are missing any values specified in cols.
    """
    idx_keep = 1 ^ np.any(df_X[cols].isnull(), axis=1)
    return df_X[idx_keep].copy(), df_y[idx_keep].copy()



# -------- Experiment to segregate data by numerical quantiles and categorical quantiles ---- didn't help

# def df_slice(df, var, lb, ub):
#     d = df.copy()
#     d = d[d[var] <= ub]
#     d = d[d[var] > lb]
#     dr = df.copy()
#     dr = pd.concat([dr[dr[var] > ub], dr[dr[var] <= lb]], axis=0)
#     return d, dr


# def quantile(df, var, n, q, return_remainder=True):
#     """Return the horizontal slice of df corresponding to the nth quartile of df[var]
#     Args:
#         df(pandas.Dataframe): the dataframe
#         var(string): one of the columns of df
#         n(int): in the range of [1,4]
#         q(float): the quantile fraction. E.g. 0.25 gives 4 pieces, 0.1 gives 10, etc.
#     """
#     lb = df[var].quantile(q=q * (n-1))
#     if n == 1: # Subtract epsilon to ensure the n=1 grabs the lower-bound value
#         lb -= 0.00001
#     ub = df[var].quantile(q=q * n)
#     dfd, dfdr = df_slice(df, var, lb, ub)
#     if return_remainder:
#         return dfd, dfdr, (lb, ub)
#     else:
#         return dfd, (lb, ub)


# def decile_iterator(df, var, return_remainder=True):
#     """Yields the horizontal slices of df, split into the declies of df[var]
#     """
#     for i in range(1,11):
#         yield decile(df, var, i, 0.1, return_remainder=return_remainder)


# def quartile_iterator(df, var, return_remainder=True):
#     """Yields the horizontal slices of df, split into the declies of df[var]
#     """
#     for i in range(1,5):
#         yield quantile(df, var, i, 0.25, return_remainder=return_remainder)


# def median_iterator(df, var, return_remainder=True):
#     """Yields the horizontal slices of df, split into the declies of df[var]
#     """
#     for i in range(1,3):
#         yield quantile(df, var, i, 0.5, return_remainder=return_remainder)


# def runtrial(df_Xd, df_yd, df_Xdr, df_ydr, train_size=0.5, verbose=False, num_runs=20):
#     Xd = df_Xd.to_numpy()
#     yd = df_yd.to_numpy()
#     Xdr = df_Xdr.to_numpy()
#     ydr = df_ydr.to_numpy()

#     auc_values = []
#     for _ in range(num_runs):
#         # Split the decile into train/test, keep the decile-remainder in train
#         X_train, X_test, y_train, y_test = train_test_split(Xd, yd, train_size=train_size, stratify=yd)
#         X_train = np.concatenate([X_train, Xdr], axis=0)
#         y_train = np.concatenate([y_train, ydr], axis=0)
#         X_train, y_train = oversample_function.fit_resample(X_train, y_train)

#         model = make_catboost()
#         model.fit(X_train, y_train, categorical)
#         assert len(np.unique(y_test)) == 2
#         y_probs = model.predict_proba(X_test)[:,1]
#         fpr, tpr, thresholds = metrics.roc_curve(y_test, y_probs, pos_label=1)
#         auc = metrics.auc(fpr, tpr)
#         auc_values.append(auc)

#     if verbose:
#         print("X_train: \t{}, \ty_train: \t{}".format(X_train.shape, y_train.shape))
#         print("X_test: \t{}, \ty_test: \t{}, y_test_pos={} & y_test_neg={}".format(
#             X_test.shape,
#             y_test.shape,
#             sum(y_test),
#             len(y_test) - sum(y_test)
#         ))

#     return np.mean(auc), np.std(auc), sum(y_test)

# results = {}
# numerical_vars = numerical_names
# train_size = 0.3
# num_runs=20
# # numerical_vars = ['prostate_dose']
# for var in numerical_vars:
#     print("------------------ {} ------------------".format(var))
#     results[var] = {}
#     df_Xy = pd.concat([df_X, df_y], axis=1)
#     for df_Xyd, df_Xyr, r in median_iterator(df_Xy, var):
#         # Skip if the range r is meaningless.
#         # (Sometimes quantiles get messed up in the presence of too many nans.)
#         if r[0] == r[1]:
#             results[var][r] = None
#             continue

#         df_Xd = df_Xyd.drop(columns=df_y.columns)
#         df_Xdr = df_Xyr.drop(columns=df_y.columns)
#         df_yd = df_Xyd[df_y.columns]
#         df_ydr = df_Xyr[df_y.columns]

#         # Skip if insufficient data.
#         value_counts = df_Xyd[df_yd.columns.values[0]].value_counts()
#         if len(df_Xyd) < 50 or np.any(value_counts < 2) or len(value_counts) != 2:
#             results[var][r] = None
#             continue

#         mean_auc, std_auc, n_y_test_pos = runtrial(
#             df_Xd, df_yd, df_Xdr, df_ydr, train_size=train_size, num_runs=num_runs, verbose=True)
#         info = r + ("n_y_test_pos={}".format(n_y_test_pos),)
#         results[var][info] = mean_auc
#         print("{}: len={}: auc={}".format(r, len(df_Xyd), mean_auc))

#     # The nan-slice.
#     df_Xyd = df_Xy[df_Xy[var].isnull()]
#     if len(df_Xyd) < 50 or np.any(df_Xyd[df_yd.columns.values[0]].value_counts() < 2):
#         print()
#         results[var]['nan'] = None
#         continue

#     df_Xd = df_Xyd.drop(columns=df_y.columns)
#     df_yd = df_Xyd[df_y.columns]
#     mean_auc, std_auc, n_y_test_pos = runtrial(
#         df_Xd, df_yd, df_Xdr, df_ydr, train_size=train_size, num_runs=num_runs, verbose=True)
#     info = ("nan", "n_y_test_pos={}".format(n_y_test_pos),)
#     results[var][info] = mean_auc
#     print("nan: len={}: auc={}".format(len(df_Xyd), mean_auc))

#     print()

# pad5 = lambda x: [0] * (5-len(x)) + list(x)
# df_results = pd.DataFrame({key : pad5(value.values()) for key, value in results.items()})
# printc(df_results)



# def category_iterator(df_Xy, var, return_remainder=True):
#     for unique_val in df_Xy[var].unique():
#         df_Xyd = df_Xy[df_Xy[var] == unique_val]
#         df_Xyr = df_Xy[df_Xy[var] != unique_val]
#         yield df_Xyd, df_Xyr, unique_val


# results = {}
# categorical_vars = categorical_names
# train_size = 0.3
# num_runs=20
# # categorical_vars = ['race']
# for var in categorical_vars:
#     print("------------------ {} ------------------".format(var))
#     results[var] = {}
#     df_Xy = pd.concat([df_X, df_y], axis=1)
#     for df_Xyd, df_Xyr, unique_val in category_iterator(df_Xy, var, return_remainder=True):
#         df_Xd = df_Xyd.drop(columns=df_y.columns)
#         df_Xr = df_Xyr.drop(columns=df_y.columns)
#         df_yd = df_Xyd[df_y.columns]
#         df_yr = df_Xyr[df_y.columns]

#         # Skip if insufficient data.
#         value_counts = df_Xyd[df_yd.columns.values[0]].value_counts()
#         if len(df_Xyd) < 50 or np.any(value_counts < 2) or len(value_counts) != 2:
#             results[var][unique_val] = None
#             continue

#         mean_auc, std_auc, n_y_test_pos = runtrial(
#             df_Xd, df_yd, df_Xr, df_yr, train_size=train_size, num_runs=num_runs, verbose=True)
#         info = (unique_val,) + ("n_y_test_pos={}".format(n_y_test_pos),)
#         results[var][info] = mean_auc
#         print("{}: len={}: auc={}".format(unique_val, len(df_Xyd), mean_auc))

#     # The nan-slice.
#     df_Xyd = df_Xy[df_Xy[var].isnull()]
#     if len(df_Xyd) < 50 or np.any(df_Xyd[df_yd.columns.values[0]].value_counts() < 2):
#         print()
#         results[var]['nan'] = None
#         continue

#     df_Xd = df_Xyd.drop(columns=df_y.columns)
#     df_yd = df_Xyd[df_y.columns]
#     mean_auc, std_auc, n_y_test_pos = runtrial(
#         df_Xd, df_yd, df_Xr, df_yr, train_size=train_size, num_runs=num_runs, verbose=True)
#     info = ("nan", "n_y_test_pos={}".format(n_y_test_pos),)
#     results[var][info] = mean_auc
#     print("nan: len={}: auc={}".format(len(df_Xyd), mean_auc))

#     print()

# pad5 = lambda x: [0] * (11-len(x)) + list(x)
# df_results = pd.DataFrame({key : pad5(value.values()) for key, value in results.items()})
# printc(df_results)
