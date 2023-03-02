"""Helper functions to support dealing with image data.
"""
import pickle
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict
from rtog_constants import slide_case_correspondence_files
from rtog_helper import rtog_from_study_number


def load_features_probs(dirpath):
    features = {}
    probs = {}
    for filepath in os.listdir(dirpath):
        g = pickle.load(open(os.path.join(dirpath, filepath), 'rb'))
        features[filepath.split(".")[0]] = g[0]
        probs[filepath.split(".")[0]] = g[1]
    return features, probs


class CaseManager(object):
    """Helper class to map amongst (slide-id, case-id, case-features, etc.)
    """
    def __init__(self):
        self.study_numbers = ['9202', '9413', '9408', "9910", "0126"]
        self.rtogs = {sn : rtog_from_study_number(sn).df for sn in self.study_numbers}
        self.slidecase = {sn : load_slide_case_correspondence(sn, return_skipped=False) for sn in self.study_numbers}


    def __repr__(self):
        str_ = "Case Manager: studies {}".format(", ".join(self.study_numbers))
        return str_


    def get_caseid(self, slide_id, study_number):

        caseid = self.slidecase[study_number][self.slidecase[study_number]['image id'] == slide_id]['cn_deidentified']#.values[0]
        if len(caseid) > 0:
            caseid = caseid.values[0]
            return caseid
        else:
            return None


    def get_feature_by_case(self, feature_name, case_id, study_number):
        df = self.rtogs[study_number]
        feature = df[df['cn_deidentified'] == case_id][feature_name]#.values[0]
        if len(feature) > 0:
            return feature.values[0]
        else:
            return None


    def get_feature_by_slideid(self, feature_name, slide_id, study_number):
        caseid = self.get_caseid(slide_id, study_number)
        if caseid is None:
            return None
        feature = self.get_feature_by_case(feature_name, caseid, study_number)
        return feature


def load_slide_case_correspondence(study_number, return_skipped=False):
    if study_number not in slide_case_correspondence_files:
        raise ValueError('Study number not supported: {}'.format(study_number))
    filepaths = slide_case_correspondence_files[study_number]
    dfs = []
    for filepath in filepaths:
        df = pd.read_excel(filepath)
        if "Unnamed: 2" in df.columns:
            df.rename(columns=df.iloc[0], inplace = True)
            df.drop([0], inplace = True)
        if "CN_deidentified" in df.columns:
            df = df.rename(columns={"CN_deidentified":"cn_deidentified"})
        if "CN_DeIdentified" in df.columns:
            df = df.rename(columns={"CN_DeIdentified":"cn_deidentified"})
        if "CN_Deidentified" in df.columns:
            df = df.rename(columns={"CN_Deidentified":"cn_deidentified"})
        if "CN Deid" in df.columns:
            df = df.rename(columns={"CN Deid":"cn_deidentified"}) 
        if "CN-DeID'ed" in df.columns:
            df = df.rename(columns={"CN-DeID'ed":"cn_deidentified"}) 
        if "CN DE-ID" in df.columns:
            df = df.rename(columns={"CN DE-ID":"cn_deidentified"}) 
        if "CN deidentified" in df.columns:
            df = df.rename(columns={"CN deidentified":"cn_deidentified"}) 
        if "CN De-Identified" in df.columns:
            df = df.rename(columns={"CN De-Identified":"cn_deidentified"})
        if "CN De-ID" in df.columns:
            df = df.rename(columns={"CN De-ID":"cn_deidentified"})
        df.columns = df.columns.str.lower()
        dfs.append(df)
    dfs = pd.concat(dfs)

    # Merge duplicated rows
    gs = []
    skipped = []
    for g in dfs.groupby(dfs['image id']):
        g = g[1][['image id', 'cn_deidentified']]
        g = pd.concat([pd.Series(g[col].dropna().unique(), name=col) for col in g], axis=1)

        if len(g) > 1:
            skipped.append(g)
        else:
            gs.append(g)

    df_slide_case = pd.concat(gs)
    if return_skipped:
        return df_slide_case, skipped
    else:
        return df_slide_case


def print_slidecase_stats(df, prefix=""):
    cases = df['cn_deidentified'].unique()
    n = len(cases)
    mn = min(cases)
    mx = max(cases)
    images = df['image id'].unique()
    ni = len(images)
    if prefix:
        prefix += ", "
    print("{}Num cases: {} (min: {}, max: {}). Num Images: {}".format(prefix, n, mn, mx, ni))


def reduce_probs_by_sum(matrix): # matrix is N x 6, with N the number of tiles for the case
    r = np.sum(matrix, axis=0)
    r /= np.sum(r)
    return r


def reduce_probs_by_sum2(matrix): # matrix is N x 6, with N the number of tiles for the case
    r = np.sum(matrix, axis=0)
    r[:3] = 0
    r /= np.sum(r)
    return r


def reduce_probs_by_argmax(matrix):
    full_counts = np.zeros(matrix.shape[1])
    matrix = np.argmax(matrix, axis=1)
    unique, counts = np.unique(matrix, return_counts=True)
    counts = counts / np.sum(counts)
    for u, c in zip(unique, counts):
        full_counts[u] = c
    return full_counts


def reduce_probs_to_minmax(matrix):
    mn = np.min(matrix, axis=0)
    mx = np.max(matrix, axis=0)
    rv = np.concatenate([mn, mx])
    return rv

def reduce_and_append_gleason_probs(tileprobs, df_slide_to_case, df_rtog, reduction_function):
    dict_case_images = defaultdict(list)
    image2case = lambda imId, df: df[df['image id'] == imId]['cn_deidentified'].values
    skipped_images = []
    for image_id, matrix in tileprobs.items():
        caseId = image2case(int(image_id), df_slide_to_case)
        if len(caseId) == 0:
            skipped_images.append(image_id)
            continue
        assert len(caseId) == 1, "Image ID: {}, Case ID: {}".format(image_id, caseId)
        caseId = caseId[0]
        dict_case_images[caseId].append(matrix)
    print("WARNING: function reduce_and_append_gleason_probs: skipping {} images without matching case_ids.".format(len(skipped_images)))

    dict_case_gprobs = {}
    for caseId, lmats in dict_case_images.items():
        matrix = np.concatenate(lmats, axis=0)
        r = reduction_function(matrix)
        dict_case_gprobs[caseId] = r

    probs_matrix = np.vstack(list(dict_case_gprobs.values()))
    new_cols = ['cn_deidentified'] + ['gleason_probs_' + str(i) for i in range(probs_matrix.shape[1])]
    df = pd.DataFrame(columns=new_cols)
    df['cn_deidentified'] = dict_case_gprobs.keys()
    for i in range(probs_matrix.shape[1]):
        df['gleason_probs_' + str(i)] = probs_matrix[:, i]

    df_rtog_new = df_rtog.merge(df, on='cn_deidentified', how='outer')
    return df_rtog_new

