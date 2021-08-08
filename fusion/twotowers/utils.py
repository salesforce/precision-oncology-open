import numpy as np
import yaml
import pandas as pd

from sklearn.metrics import average_precision_score, roc_auc_score


def get_auroc(y, prob): 
    y = np.array(y)
    prob = np.array(prob)
    if np.isnan(prob).any():
        auroc = 0
    elif len(set(y)) == 1:
        auroc = 0
    else:
        auroc = roc_auc_score(y, prob)
    return auroc

def get_auprc(y, prob): 
    y = np.array(y)
    prob = np.array(prob)
    if np.isnan(prob).any():
        auprc = 0
    elif len(set(y)) == 1:
        auprc = 0
    else:
        auprc = average_precision_score(y, prob)
    return auprc

import collections

def flatten(d, parent_key='', sep='.'):
    '''flatten a nested dictionary'''
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_best_ckpt_path(ckpt_paths, ascending=False):
    """get best ckpt path from a list of ckpt paths

    ckpt_paths: JSON file with ckpt path to metric pair
    ascending: sort paths based on ascending or descending metrics
    """

    with open(ckpt_paths, 'r') as stream:
        ckpts = yaml.safe_load(stream)

    ckpts_df = pd.DataFrame.from_dict(ckpts, orient='index').reset_index()
    ckpts_df.columns = ['path', 'metric']
    best_ckpt_path = ckpts_df.sort_values('metric', ascending=ascending).head(1)['path'].item()

    return best_ckpt_path
