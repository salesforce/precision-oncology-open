"""Helper classes and functions with RTOG studies.
"""
import random
import pandas as pd
import numpy as np
import pickle
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import pint

# Constants defining variable and file parsing
from rtog_constants import gcp_baseline_paths, rtog_endpoints, rtog_binary_mapping, rtog_unknown_class_X
from rtog_constants import rtog_default_class_y, rtog_text_fields, rtog_field_mapping, rtog_categorical_fields

# Functions allowing RTOG data manipulation
from rtog_constants import is_categorical, merge, serum_values_to_ng_dl


def rtog_from_study_number(study_number, create_endpoints=True, standardize=False):
    """Helper function. Loads an RTOG object given the study number (str)."""
    study_path = gcp_baseline_paths[study_number]
    rtog = RTOG(filename=study_path, study_number=study_number, file_type='excel', create_endpoints=create_endpoints)
    if standardize:
        rtog.standardize_rx()
        rtog.standardize_race()
        rtog.standardize_gleason_scores()
        rtog.standardize_tstage()
        rtog.standardize_pelvic_rt()
        rtog.standardize_prostate_dose()
        rtog.standardize_rt_complete()
        rtog.standardize_biochemical_failure()
        rtog.standardize_disease_specific_survival()
        rtog.cause_of_death()
#     rtog_object.standardize_baseline_serum() # Note: this line takes a long time to run, due to unit conversions. Also Osama said the data is too noisy to use.
        rtog.standardize_unknown_values_in_predictor_variables() # note: this must be done after standardize_rt_complete, bc that re-sets some unknown vars. This replaces the 'unknown' classes with nans, so that boosting can intelligently impute.
    print("Loaded RTOG {}, Standardized={}".format(study_number, standardize))
    return rtog


class RTOG(object):
    def __init__(self, filename=None, study_number=None, file_type="excel", create_endpoints=True):
        self.filename = filename
        self.df = None
        self.study_number = study_number

        # Load Endpoints, Default Classes (for y), and Unknown Classes (for X).
        if self.study_number in rtog_endpoints:
            self.endpoints = rtog_endpoints[study_number]
        if self.study_number in rtog_default_class_y:
            self.default_class_y = rtog_default_class_y[study_number]
        if self.study_number in rtog_unknown_class_X:
            self.unknown_class_X = rtog_unknown_class_X[study_number]

        # Load Data.
        if self.filename is not None:
            if file_type == "excel":
                self.df = pd.read_excel(filename)
            elif file_type == "csv":
                self.df = pd.read_csv(filename, index_col=0)
            self._field_fix()
            self.table_sort()

            # Study-specific additional derived endpoints get hardcoded here
            if study_number == '9202':
                # Add Radiotherapy info
                gcp_path = "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/NRG Statistics/RTOG 9202/All_RT_Data_9202.xlsx"
                self.df_rt = pd.read_excel(gcp_path)
                self.df_rt.columns = self.df_rt.columns.str.lower()
                self.df_rt.rename({'pelvis_does' : 'pelvis_dose'}, axis='columns', inplace=True)
            elif study_number == '9413': #note: data lacks disease specific survival
                pass
            elif study_number == '9408':
                pass
            elif study_number == '9910':
                # Add Radiotherapy info
                gcp_path = "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/NRG Statistics/RTOG 9910/Radiation_treatment_9910.xlsx"
                self.df_rt = pd.read_excel(gcp_path)
                self.df_rt.columns = self.df_rt.columns.str.lower()
            elif study_number == "0126":
                # Add Serum info
                gcp_path = "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/NRG Statistics/RTOG 0126/All_serum_testosteron_0126.xlsx"
                self.df_serum = pd.read_excel(gcp_path)
                self.df_serum.columns = self.df_serum.columns.str.lower()
            else:
                pass

            # Replace nans with defaults in endpoint fields
            self.df = self.df.fillna(self.default_class_y)

            if create_endpoints:
                for timeframe in [5,10,15,25]:
                    self.add_distant_met_Nyr_endpoint(timeframe)
                    self.add_biochemical_failure_Nyr_endpoint(timeframe)
                    self.add_disease_specific_survival_Nyr_endpoint(timeframe)
                    self.add_survival_Nyr_endpoint(timeframe)


    def _repr_html_(self):
        return self.df._repr_html_()


    def __getitem__(self, columns):
        if type(columns) == str:
            columns = [columns]
        new_rtog = self.copy()
        new_rtog.df = new_rtog.df[columns]
        return new_rtog


    def _field_fix(self):
        """Fixes field names for uniformity and typos. Determined in rtog_constants.py
        """
        self.df = self.df.rename(columns=str.lower)
        self.df = self.df.rename(rtog_field_mapping, axis='columns')


    def table_sort(self):
        """Sorts rows and columns in ascending order.
        """
        self.df = self.df.sort_index()
        self.df = self.df.sort_index(axis=1)


    def add_biochemical_failure_Nyr_endpoint(self, years):
        """Adds column 'biochemical_failure_Nyr' to self.df

        Indicates if the cancer metastasized within N years.
        Args:
            years(int): the years.

        Column values:
            0: Censored
            1: Failure within given years
            2: Competing event (death without failure)
        """
        field_name = 'biochemical_failure'
        if self.study_number == '9202':
            failure_outside_timeframe_value = 2 # Has a 'competing events' class.
            new_field = field_name + "_{}year".format(years)
        elif self.study_number == '9408':
            failure_outside_timeframe_value = 0 # Does not have a 'competing events' class.
            new_field = field_name + "_{}year".format(years)
        elif self.study_number == '9413':
            failure_outside_timeframe_value = 2 # Has a 'competing events' class.
            new_field = field_name + "_{}year".format(years)
            field_name = 'phoenix_biochemical_failure'
        elif self.study_number == '9910':
            failure_outside_timeframe_value = 2 # Has a 'competing events' class.
            new_field = field_name + "_{}year".format(years)
            field_name = 'phoenix_biochemical_failure'
        elif self.study_number == "0126":
            failure_outside_timeframe_value = 2 # Has a 'competing events' class.
            new_field = field_name + "_{}year".format(years)
            field_name = 'phoenix_biochemical_failure'
        else:
            raise ValueError("The failure value for biochemical_failure is not set for this study: {}".format(self.study_number))
        field_name_years = field_name + "_years"
        assert field_name in self.endpoint_fields(), "{} not in endpoint fields".format(field_name)
        assert field_name_years in self.endpoint_fields() , "{} not in endpoint fields".format(field_name_years)

        # Insert new field. If it exists already, re-compute it.
        if new_field in self.df.columns:
            self.df = self.df.drop(columns=[new_field])
        idx = self.df.columns.get_loc(field_name) + 1
        new_column_vals = []
        for f, fy in zip(self.df[field_name], self.df[field_name_years]):
            if f == 0: # Default class for biochemical_failure is 0. Same for biochemical_failure_5yr.
                new_column_vals.append(0)
            if f == 2:
                new_column_vals.append(2)
            if f == 1:
                assert ~np.isnan(fy), "Found biochemical_failure=1, with biochemical_failure_years=nan. Impossible. See rtog {}".format(
                    self.study_number)
                if fy <= years:
                    new_column_vals.append(1)
                else:
                    new_column_vals.append(failure_outside_timeframe_value)
        self.df.insert(loc=idx, column=new_field, value=list(map(int, new_column_vals)))
        self.table_sort()

        # Update endpoint fields
        self._add_endpoint_field(new_field, 0)


    def add_disease_specific_survival_Nyr_endpoint(self, years):
        """Adds column 'disease_specific_survival_Nyr' to self.df

        Indicates if the patient has lived free of prostate cancer within N years.
        Note: Contrast this with disease_free_survival, which means the patient has lived free of any disease.
        Args:
            years(int): the years.

        Column values:
            0: Censored
            1: Failure within given years
            2: Competing event (death from something other than prostate cancer.)
        """
        field_name = 'disease_specific_survival'
        if self.study_number == '9202':
            failure_outside_timeframe_value = 2
#           field_name_years = "survival_years" # Stephanie confirmed we can use this value.
        elif self.study_number == '9408':
            failure_outside_timeframe_value = 2
#           field_name_years = "dsm_years" # Osama confirmed we can use this value.
        elif self.study_number == '9413':
            failure_outside_timeframe_value = 2
        elif self.study_number == '9910':
            failure_outside_timeframe_value = 2
        elif self.study_number == '0126':
            failure_outside_timeframe_value = 2
        else:
            raise ValueError("The failure_outside_timeframe_value for disease specific survival is not set for this study: {}".format(
                self.study_number))
        field_name_years = field_name + "_years"
        assert field_name in self.endpoint_fields(), "{} not in endpoint fields".format(field_name)
        assert field_name_years in self.endpoint_fields() , "{} not in endpoint fields".format(field_name_years)

        # Insert new field. If it exists already, re-compute it.
        new_field = field_name + "_{}year".format(years)
        if new_field in self.df.columns:
            self.df = self.df.drop(columns=[new_field])
        idx = self.df.columns.get_loc(field_name) + 1
        new_column_vals = []
        for dss, dfsy in zip(self.df[field_name], self.df[field_name_years]):
            if dss == 0: # Default class for distant_met is 0. Same for distant_met_5yr.
                new_column_vals.append(0)
            if dss == 2:
                new_column_vals.append(2)
            if dss == 1:
                if dfsy <= years:
                    new_column_vals.append(1)
                else:
                    new_column_vals.append(failure_outside_timeframe_value)
        try:
            self.df.insert(loc=idx, column=new_field, value=list(map(int, new_column_vals)))
        except:
            import IPython
            IPython.embed()
#       self.df.insert(loc=idx, column=new_field, value=list(map(int, new_column_vals)))
        self.table_sort()

        # Update endpoint fields
        self._add_endpoint_field(new_field, 0)


    def add_survival_Nyr_endpoint(self, years):
        """Adds column 'survival_Nyr' to self.df. Refers to overall survival.

        Args:
            years(int): the years.

        Column values:
            0: Alive, within given years.
            1: Death, within given years.
        """
        field_name = 'survival'
        field_name_years = "survival_years" # Note, that for disease_specific_survival=1, we can take the time period from disease_free_surival_years.
        assert field_name in self.endpoint_fields(), "{} not in endpoint fields".format(field_name)
        assert field_name_years in self.endpoint_fields() , "{} not in endpoint fields".format(field_name_years)

        # Insert new field. If it exists already, re-compute it.
        new_field = field_name + "_{}year".format(years)
        if new_field in self.df.columns:
            self.df = self.df.drop(columns=[new_field])
        idx = self.df.columns.get_loc(field_name) + 1
        new_column_vals = []
        for fn, fny in zip(self.df[field_name], self.df[field_name_years]):
            if fn == 0: # Default class for distant_met is 0. Same for distant_met_5yr.
                new_column_vals.append(0)
            if fn == 1:
                if fny <= years:
                    new_column_vals.append(1)
                else:
                    new_column_vals.append(0)
        self.df.insert(loc=idx, column=new_field, value=list(map(int, new_column_vals)))
        self.table_sort()

        # Update endpoint fields
        self._add_endpoint_field(new_field, 0)


    def add_distant_met_Nyr_endpoint(self, years):
        """Adds column 'distant_met_Nyr' to self.df

        Indicates if the cancer metastasized within N years.
        Args:
            years(int): the years.

        Column values:
            0: Censored
            1: Failure within given years (metastatic prostate cancer)
            2: Competing event (death from something other than prostate cancer.)
        """
        field_name = 'distant_met'
        field_name_years = field_name + "_years"
        if self.study_number == '9202':
            failure_outside_timeframe_value = 2 # Has a 'competing events' class.
        elif self.study_number == '9408':
            failure_outside_timeframe_value = 0 # Has a 'competing events' class
        elif self.study_number == '9413':
            failure_outside_timeframe_value = 2 # Has a 'competing events' class.
        elif self.study_number == '9910':
            failure_outside_timeframe_value = 2 # Has a 'competing events' class.
        elif self.study_number == '0126':
            failure_outside_timeframe_value = 2 # Has a 'competing events' class.
        else:
            raise ValueError("The failure_outside_timeframe_value for disease specific survival is not set for this study: {}".format(self.study_number))
        assert field_name in self.endpoint_fields(), "{} not in endpoint fields".format(field_name)
        assert field_name_years in self.endpoint_fields() , "{} not in endpoint fields".format(field_name_years)

        # Insert new field. If it exists already, re-compute it.
        new_field = field_name + "_{}year".format(years)
        if new_field in self.df.columns:
            self.df = self.df.drop(columns=[new_field])
        idx = self.df.columns.get_loc(field_name) + 1
        new_column_vals = []
        for dm, dmy in zip(self.df[field_name], self.df[field_name_years]):
            if dm == 0: # Default class for distant_met is 0. Same for distant_met_5yr.
                new_column_vals.append(0)
            if dm == 2:
                new_column_vals.append(2)
            if dm == 1:
                assert ~np.isnan(dmy), "Found distant_met=1, with distant_met_years=nan. Impossible. See rtog {}".format(self.study_number)
                if dmy <= years:
                    new_column_vals.append(1)
                else:
                    new_column_vals.append(failure_outside_timeframe_value)
        self.df.insert(loc=idx, column=new_field, value=list(map(int, new_column_vals)))
        self.table_sort()

        # Update endpoint fields
        self._add_endpoint_field(new_field, 0)


    def _add_endpoint_field(self, endpoint_field, default_class_y):
        if endpoint_field in self.endpoints:
            if self.default_class_y[endpoint_field] != default_class_y:
                raise ValueError("Endpoint already listed, with different default class: {}. New attempt: {}".format(
                    self.default_class_y[endpoint_field], default_class_y
                ))
            return
        self.endpoints.append(endpoint_field)
        self.default_class_y[endpoint_field] = default_class_y


    def printc(self):
        prev = pd.options.display.max_columns
        prev_r = pd.options.display.max_rows
        pd.options.display.max_columns = None
        pd.options.display.max_rows = 90
        display(self.df)
        pd.options.display.max_columns = prev
        pd.options.display.max_rows = prev_r


    def get_fields(self):
        return self.df.columns


    def set_study_number(self, number):
        if number not in rtog_endpoints:
            raise ValueError('Study number not available: {}. Options: {}'.format(number, rtogendpoints.keys()))
        self.study_number = number
        self.endpoints = rtog_endpoints[number]
        self.default_class_y = rtog_default_class_y[number]


    def copy(self):
        new_rtog = RTOG()
        new_rtog.df = self.df.copy(deep=True)
        new_rtog.filename = self.filename
        new_rtog.study_number = self.study_number
        new_rtog.endpoints = self.endpoints
        new_rtog.default_class_y = self.default_class_y
        new_rtog.unknown_class_X = self.unknown_class_X
        return new_rtog


    def drop(self, columns=''):
        new_rtog = self.copy()
        new_rtog.df = self.df.drop(columns=columns)
        return new_rtog


    def clear_columns(self, columns=[""]):
        """Sets the specified column values to empty.

        Args:
            columns(list): the names of the columns to replace.
        """
        N = len(self.df)
        new_rtog = self.copy()
        null_columns = {c : [''] * N for c in columns}
        for c, l in null_columns.items():
            new_rtog.df[c] = l
        return new_rtog


    def endpoint_fields(self):
        if not self.study_number:
            raise ValueError("Study number not set. Cannot select endpoint fields")
        return self.endpoints


    def text_fields(self):
        if not self.study_number:
            raise ValueError("Study number not set. Cannot select text fields")
        return rtog_text_fields[self.study_number]


    def get_Xy(self, y_var=None, make_binary=False):
        """Returns training/testing data, properly formatted.

        For each study, see the RTOG XXXX Variable Listings documents for reference.
        Args:
            y_var(str): the column of self.df to use as the prediction variable. E.g. y_var='cod'
                Any rows with nans are removed.
            make_binary(bool): if True, it returns a binary vector (0,1), using the class mapping
                defined above, rtog_binary_mapping.
        """
        # Set X. Don't impute. Boosting methods do this better than you can.
        rtog_X = self.drop(columns=self.endpoint_fields() + self.text_fields())
        rtog_X = rtog_X.copy()

        rtog_meta = self.copy()
        rtog_meta.df = rtog_meta.df[self.endpoint_fields()]

        # Set y. Impute to default class.
        rtog_y = self.copy()
        rtog_y = rtog_y[rtog_y.endpoint_fields()]
        if y_var:
            default_class_y = self.default_class_y[y_var]
            rtog_y = rtog_y[y_var]
            rtog_y.df = rtog_y.df.fillna(default_class_y)

        if make_binary: # Forces y to be binary, using a pre-specified mapping in the parent class.
            for c in rtog_y.df.columns:
                mapping = rtog_binary_mapping[self.study_number][c]
                rtog_y.df[c] = rtog_y.df[c].replace(mapping)

        return rtog_X, rtog_y, rtog_meta


    def generate_test_set(self, size=100, seed=None, field_to_balance=""):
        """Samples a test set, printing the class statistics of each.

        Args:
            size(int): the number of entries to sample
            seed(int): Optional. Random seed for reproducibility.
            field_to_balance(str): Optional. If set, function tries to return an equal class
                balance in this field. E.g. disease_free_survival

        Returns:
            RTOG object - the sampled test set.
        """
        if seed is not None:
            random.seed(seed)
        df = self.df.copy(deep=True)
        if field_to_balance:
            classes = df[field_to_balance].unique()
            indices = {}
            for c in classes:
                sub_df = df[df[field_to_balance] == c]
                indices[c] = list(sub_df.index)
            m = min([len(v) for _, v in indices.items()])
            for c, l in indices.items():
                if len(l) > m:
                    random.shuffle(l)
                    indices[c] = l[:m]
            idx = [elem for _, l in indices.items() for elem in l]
        else:
            idx = list(range(len(df)))
        random.shuffle(idx)
        idx = idx[:size]
        new_rtog = self.copy()
        new_rtog.df = df
        new_rtog.df = new_rtog.df.loc[idx]
        return new_rtog


    def to_csv(self, filename):
        self.df.to_csv(filename)


    def standardize_disease_specific_survival(self, drop_prior_columns=True):
        self.standardize_disease_specific_survival_events(drop_prior_columns=drop_prior_columns)
        self.standardize_disease_specific_survival_years(drop_prior_columns=drop_prior_columns)

        # If DSS-years unspecified but DSS censored, set DSS-years to 25 (assume long time).
        isnan = self.df['disease_specific_survival_years'].isnull().values
        iszero = (self.df['disease_specific_survival'] == 0).values
        self.df.loc[np.logical_and(isnan, iszero), 'disease_specific_survival_years'] = 25


    def standardize_disease_specific_survival_events(self, drop_prior_columns=True):
        """Merges variants of DSS, prioritizing phoenix, and naming everything disease_specific_survival
        Args:
            drop_prior_columns(bool): If True, drops the original columns.
        """
        bcr_fields = [f for f in self.df.columns if 'disease_specific_survival' in f]
        e_bcr_fields = np.array([f for f in bcr_fields if 'year' not in f])
        idx_sort = []
        idx_sort.append(np.where(['phoenix' in e for e in e_bcr_fields])[0])
        idx_sort.append(np.where(['disease_specific_survival' == e for e in e_bcr_fields])[0])
        idx_sort = np.array([i[0] for i in idx_sort if len(i) > 0])
        e_bcr = self.df[e_bcr_fields[idx_sort]]
        new_values = e_bcr[e_bcr.columns[0]]
        for i in range(1,len(e_bcr.columns)):
            next_best = e_bcr[e_bcr.columns[i]][new_values.isnull()].values.copy()
            new_values = new_values.fillna(pd.Series(next_best))
        self.df = self.df.assign(disease_specific_survival=new_values)


    def standardize_disease_specific_survival_years(self, drop_prior_columns=True):
        """Merges variants of BCR, prioritizing phoenix, and naming everything disease_specific_survival
        Args:
            drop_prior_columns(bool): If True, drops the original columns.
        """
        bcr_fields = [f for f in self.df.columns if 'disease_specific_survival' in f]
        e_bcr_fields = np.array([f for f in bcr_fields if 'years' in f])
        idx_sort = []
        idx_sort.append(np.where(['phoenix' in e for e in e_bcr_fields])[0])
        idx_sort.append(np.where(['disease_specific_survival_years' == e for e in e_bcr_fields])[0])
        idx_sort = np.array([i[0] for i in idx_sort if len(i) > 0])
        e_bcr = self.df[e_bcr_fields[idx_sort]]
        new_values = e_bcr[e_bcr.columns[0]]
        for i in range(1,len(e_bcr.columns)):
            next_best = e_bcr[e_bcr.columns[i]][new_values.isnull()].values.copy()
            new_values = new_values.fillna(pd.Series(next_best))
        self.df = self.df.assign(disease_specific_survival_years=new_values)


    def standardize_biochemical_failure(self, drop_prior_columns=True):
        self.standardize_biochemical_failure_events(drop_prior_columns=drop_prior_columns)
        self.standardize_biochemical_failure_years(drop_prior_columns=drop_prior_columns)


    def standardize_biochemical_failure_events(self, drop_prior_columns=True):
        """Merges variants of BCR, prioritizing phoenix, and naming everything biochemical_failure
        Args:
            drop_prior_columns(bool): If True, drops the original columns.
        """
        bcr_fields = [f for f in self.df.columns if 'biochemical' in f]
        e_bcr_fields = np.array([f for f in bcr_fields if 'year' not in f])
        idx_sort = []
        idx_sort.append(np.where(['phoenix' in e for e in e_bcr_fields])[0])
        idx_sort.append(np.where(['biochemical_failure' == e for e in e_bcr_fields])[0])
        idx_sort.append(np.where(['astro' in e for e in e_bcr_fields])[0])
        idx_sort = np.array([i[0] for i in idx_sort if len(i) > 0])
        e_bcr = self.df[e_bcr_fields[idx_sort]]
        new_values = e_bcr[e_bcr.columns[0]]
        for i in range(1,len(e_bcr.columns)):
            next_best = e_bcr[e_bcr.columns[i]][new_values.isnull()].values.copy()
            new_values = new_values.fillna(pd.Series(next_best))
        self.df = self.df.assign(biochemical_failure=new_values)


    def standardize_biochemical_failure_years(self, drop_prior_columns=True):
        """Merges variants of BCR, prioritizing phoenix, and naming everything biochemical_failure
        Args:
            drop_prior_columns(bool): If True, drops the original columns.
        """
        bcr_fields = [f for f in self.df.columns if 'biochemical' in f]
        e_bcr_fields = np.array([f for f in bcr_fields if 'years' in f])
        idx_sort = []
        idx_sort.append(np.where(['phoenix' in e for e in e_bcr_fields])[0])
        idx_sort.append(np.where(['biochemical_failure_years' == e for e in e_bcr_fields])[0])
        idx_sort.append(np.where(['astro' in e for e in e_bcr_fields])[0])
        idx_sort = np.array([i[0] for i in idx_sort if len(i) > 0])
        e_bcr = self.df[e_bcr_fields[idx_sort]]
        new_values = e_bcr[e_bcr.columns[0]]
        for i in range(1,len(e_bcr.columns)):
            next_best = e_bcr[e_bcr.columns[i]][new_values.isnull()].values.copy()
            new_values = new_values.fillna(pd.Series(next_best))
        self.df = self.df.assign(biochemical_failure_years=new_values)


    def standardize_baseline_psa(self, drop_prior_columns=True):
        """Merges variants of 'baseline_psa' together across studies.
        Args:
            drop_prior_columns(bool): If True, drops the original columns.
        """
        if self.study_number == '0126':
            self.df['baseline_psa'] = self.df['psa']
            if drop_prior_columns:
                self.df.drop(columns='psa')


    def standardize_baseline_serum(self, drop_prior_columns=True):
        """Merges baseline_serum* values into a single, column: baseline_serum_ng_dl, deleting the original columns.
        Args:
            drop_prior_columns(bool): If True, drops the original baseline_serum and baseline_serum_unit (or equivalent) columns.
        """
        baseline_serum_ngdl = []
        if self.study_number == "9202":
            # Has two columns: baseline_serum, and baseline_serum_nmol_l, which are all mixed up
            # Per Osama:
            #    if the value >100, it's in ng/dl, and belongs to baseline_serum
            #.   if the value <100, it's in nmol_l, and belongs to baseline_serum_nmol_l
            # After running the code below:
            #    import matplotlib.pyplot as plt
            #    v = list(r9202.df['baseline_serum_nmol_l'].values) + list(r9202.df['baseline_serum'])
            #    v = [val for val in v if not np.isnan(val)]
            #    plt.hist(v, bins='auto')
            # Is it evident that 75 is a better cutoff
            cutoff = 75

            for index, row in tqdm(self.df.iterrows()):

                # If there's a conflict between baseline_serum and baseline_serum_nmol_l, we set the value to NaN
                if not (np.isnan(row['baseline_serum']) or np.isnan(row['baseline_serum_nmol_l'])):
                    print("9202: serum conflict, setting to Nan: index={}, baseline_serum={}, baseline_serum_nmol_l={}".format(
                        index, row['baseline_serum'], row['baseline_serum_nmol_l']
                    ))
                    baseline_serum_ngdl.append(np.nan)
                    continue

                # Grab the row's serum value. One column has a nan, the other has a number.
                if np.isnan(row['baseline_serum']):
                    rowval = row['baseline_serum_nmol_l']
                else:
                    rowval = row['baseline_serum']

                if rowval < cutoff:
                    baseline_serum_ngdl.append(serum_values_to_ng_dl(rowval, 'nmol/l'))
                else:
                    baseline_serum_ngdl.append(rowval)
            if drop_prior_columns:
                self.df.drop(columns=['baseline_serum', 'baseline_serum_nmol_l'], inplace=True)

        elif self.study_number == "9408":
            # Conversion: 1= NMOL/L 2 = NG/DL 3 = NG/ML 4= Unit/NOS
            for index, row in tqdm(self.df.iterrows()):
                value = row['baseline_serum']
                unit = row['baseline_serum_unit']
                if np.isnan(value) or np.isnan(unit):
                    baseline_serum_ngdl.append(np.nan)
                    continue

                if unit == 1:
                    new_value = serum_values_to_ng_dl(value, "nmol/l")
                elif unit == 2:
                    new_value = serum_values_to_ng_dl(value, "ng/dl")
                elif unit == 3:
                    new_value = serum_values_to_ng_dl(value, "ng/ml")
                elif unit == 4:
                    #TODO: Adjust this, pending Osama/Felix clarifying how to convert unit/nos to ng/dl
                    print("9408: Action unknown for unit/nos, index={}. Setting baseline_serum value to nan".format(
                        index
                    ))
                    new_value = np.nan
                else:
                    raise ValueError("baseline_serum_unit type unknown: index={}, unit={}".format(index, unit))

                baseline_serum_ngdl.append(new_value)

            if drop_prior_columns:
                self.df.drop(columns=['baseline_serum', 'baseline_serum_unit'], inplace=True)

        elif self.study_number == "9413":
            # Conversion: 1 = ng/dl    2 = ng/ml     3 = nmol/l 4 = units/NOS
            for index, row in tqdm(self.df.iterrows()):
                value = row['baseline_serum']
                unit = row['baseline_serum_unit']
                if np.isnan(value) or type(unit) != str:
                    baseline_serum_ngdl.append(np.nan)
                    continue

                unit = unit.lower()
                if unit in ["ng/dl", "ng/ml", "nmol/l"]:
                    new_value = serum_values_to_ng_dl(value, unit)
                elif unit in ["unit/nos", "units/nos"]:
                    #TODO: Adjust this, pending Osama/Felix clarifying how to convert unit/nos to ng/dl
                    print("WARNING: Action unknown for unit/nos, index={}. Setting baseline_serum value to nan".format(
                        index
                    ))
                    new_value = np.nan
                elif unit in ['unk']:
                    new_value = np.nan
                else:
                    raise ValueError("baseline_serum_unit type unknown: index={}, unit={}".format(index, unit))

                baseline_serum_ngdl.append(new_value)

            if drop_prior_columns:
                self.df.drop(columns=['baseline_serum', 'baseline_serum_unit'], inplace=True)

        elif self.study_number == "9910":
            print("9910: no baseline_serum field. No action taken")
            self.table_sort()
            return
        elif self.study_number == '0126':
            # Conversion: 1=ng/dl   2=nmol/L   3=Other
            df_serum = self.df_serum.copy()
            df_serum['baseline_serum_value'] = df_serum['serum_value']
            df_serum.loc[df_serum['serum_years'] != 0.0, 'baseline_serum_value'] = np.nan
            df_serum = df_serum[1 ^ df_serum['baseline_serum_value'].isnull()]
            self.df = pd.merge(self.df, df_serum[['cn_deidentified', 'baseline_serum_value', 'serum_unit']], on=['cn_deidentified'], how='left')
            for index, row in tqdm(self.df.iterrows()):
                value = row['baseline_serum_value']
                unit = row['serum_unit']
                if np.isnan(value) or unit not in {1,2}:
                    baseline_serum_ngdl.append(np.nan)
                elif unit == 1:
                    baseline_serum_ngdl.append(value)
                elif unit == 2: # Unit is nmol/L
                    new_value = serum_values_to_ng_dl(value, 'nmol/l')
                    baseline_serum_ngdl.append(new_value)
                else:
                    raise ValueError("0126, index={}, action unknown for value={}, unit={}".format(index, value, unit))
            self.df.drop(columns=['baseline_serum_value', 'serum_unit'], inplace=True)

        else:
            raise ValueError("Study number not supported: {}".format(self.study_number))

        self.df['baseline_serum_ng_dl'] = baseline_serum_ngdl
        self.table_sort()


    def standardize_rx(self):
        """Standardizes the treatment arms according to the following convention.

        Notation:
            RT (Radiotherapy),
            STADT (Short-term Androgen Deprivation Therapy),
            LTADT (Long-term Androgen Deprivation Therapy)

        Classes:
            0: RT
            1: RT + STADT (Short-term Hormone)
            2: RT + LTADT (Long-term Hormone)
            3: RT + ITADT (Intermediate-term Hormone)
        """
        if self.study_number == "9202": # Already matches above key
            pass
        elif self.study_number == "9408":
            self.df.loc[self.df['rx'] == 2, 'rx'] = 0
        elif self.study_number == "9413":
            self.df['rx_orig'] = self.df['rx'] # We are required to store this for standardize_pelvic_rt
            self.df.loc[self.df['rx'] == 1, 'rx'] = 1   # Variable sheet: 1 = Pre + Boost
            self.df.loc[self.df['rx'] == 2, 'rx'] = 1   # Variable sheet: 2 = Pre + Prostate
            self.df.loc[self.df['rx'] == 3, 'rx'] = 1   # Variable sheet: 3 = Boost/Horm
            self.df.loc[self.df['rx'] == 4, 'rx'] = 1   # Variable sheet: 4 = Pros RT/Horm
        elif self.study_number == "9910": # Already matches above key. "Ask Osama what to do about this rx. It's "intermediate"
            self.df.loc[self.df['rx'] == 1, 'rx'] = 1   # Variable sheet: 1 = 8 Wks Pre-RT Hormone
            self.df.loc[self.df['rx'] == 2, 'rx'] = 3   # Variable sheet: 2 = 28 Wks Pre-Rt Hormone
        elif self.study_number == "0126": #TODO: ask Osama what to do about this rx. it's "intermediate"
            self.df.loc[self.df['rx'] == 1, 'rx'] = 0   # Variable sheet: 1 = 3D/IMRT 70.2 (3D Intensity-Modulated RadioTherapy)
            self.df.loc[self.df['rx'] == 2, 'rx'] = 0   # Variable sheet: 2 = 2 = 3D/IMRT 79.2


    def standardize_race(self):
        """Standardizes race according to the rules below.
        White=1,
        Hispanic=2,
        AA=3,
        Asian=4,
        Other=5
        Unknown=9

        9408:
            Expected Values: 1,2,3,4,5,6,98,99
            Actual Values: 1,  2,  3,  4,  5,  6, 98, 99
        9413:
            Expected Values: 1,2,3,4,6,7
            Actual Values: 1,  2,  3,  5,  6, 98, 99

        As a result of the above discrepancy (9413's actual values don't match the expected. They match the expected of 9408 instead)
        The code below has 9413's standardization following the rules of 9408.
        TODO: Osama to advise on what to do.
        """
        if self.study_number == "9202":
            self.df.loc[self.df['race'] == 6, 'race'] = 'Native American'
            self.df.loc[self.df['race'] == 7, 'race'] = 'Other'
            self.df.loc[self.df['race'] == 8, 'race'] = 'Unknown'
            self.df.loc[self.df['race'] == 9, 'race'] = 'Prefer not to answer'

            self.df.loc[self.df['race'] == 'Native American', 'race'] = 5
            self.df.loc[self.df['race'] == 'Other', 'race'] = 5
            self.df.loc[self.df['race'] == 'Unknown', 'race'] = 9
            self.df.loc[self.df['race'] == 'Prefer not to answer', 'race'] = 9

            # Changing the unknown class requires changing self.unknown_class_X
            self.unknown_class_X['race'] = [9]

        elif self.study_number == "9408":
            self.df.loc[self.df['race'] == 4, 'race'] = 'NativeHawaiian'  # Native Hawaiian -> Tmp
            self.df.loc[self.df['race'] == 5, 'race'] = 'Asian'      # Asian -> Asian
            self.df.loc[self.df['race'] == 6, 'race'] = 'NativeAmerican'      # Native American -> Other
            self.df.loc[self.df['race'] == 98, 'race'] = 'Other'      # Other -> Other
            self.df.loc[self.df['race'] == 99, 'race'] = 'Unknown'      # Unknown -> Unknown

            self.df.loc[self.df['race'] == 'NativeHawaiian', 'race'] = 5  # Tmp -> Other
            self.df.loc[self.df['race'] == 'Asian', 'race'] = 4  # Tmp -> Other
            self.df.loc[self.df['race'] == 'NativeAmerican', 'race'] = 5  # Tmp -> Other
            self.df.loc[self.df['race'] == 'Other', 'race'] = 5  # Tmp -> Other
            self.df.loc[self.df['race'] == 'Unknown', 'race'] = 9  # Tmp -> Other

            # Changing the unknown class requires changing self.unknown_class_X
            self.unknown_class_X['race'] = [9]

        elif self.study_number == "9413":
            # Copied rules from 9408. At some point I was told to do this.
            self.df.loc[self.df['race'] == 4, 'race'] = 'tmp'  # Native Hawaiian -> Tmp
            self.df.loc[self.df['race'] == 5, 'race'] = 4      # Asian -> Asian
            self.df.loc[self.df['race'] == 'tmp', 'race'] = 5  # Tmp -> Other
            self.df.loc[self.df['race'] == 6, 'race'] = 5      # Native American -> Other
            self.df.loc[self.df['race'] == 98, 'race'] = 5      # Other -> Other
            self.df.loc[self.df['race'] == 99, 'race'] = 9      # Unknown -> Unknown

            # Changing the unknown class requires changing self.unknown_class_X
            self.unknown_class_X['race'] = [9]

            # Original rules for 9413
#           self.df.loc[self.df['race'] == 6, 'race'] = 5      # Native American -> Other
#           self.df.loc[self.df['race'] == 7, 'race'] = 5      # Other -> Other

        elif self.study_number == "9910":
            self.df.loc[self.df['race'] == 4, 'race'] = 'Native Hawaiian'  # Native Hawaiian -> Tmp
            self.df.loc[self.df['race'] == 5, 'race'] = 'Asian'      # Asian -> Asian
            self.df.loc[self.df['race'] == 6, 'race'] = 'Native American'      # Native American -> Other
            self.df.loc[self.df['race'] == 98, 'race'] = 'Other'      # Other -> Other
            self.df.loc[self.df['race'] == 99, 'race'] = 'Unknown'      # Unknown -> Unknown

            self.df.loc[self.df['race'] == 'Native Hawaiian', 'race'] = 5
            self.df.loc[self.df['race'] == 'Asian', 'race'] = 4
            self.df.loc[self.df['race'] == 'Native American', 'race'] = 5
            self.df.loc[self.df['race'] == 'Other', 'race'] = 5
            self.df.loc[self.df['race'] == 'Unknown', 'race'] = 9

            # Changing the unknown class requires changing self.unknown_class_X
            self.unknown_class_X['race'] = [9]

        elif self.study_number == "0126":
            self.df.loc[self.df['race'] == 1, 'race'] = 'Native American'
            self.df.loc[self.df['race'] == 2, 'race'] = 'Asian'
            self.df.loc[self.df['race'] == 3, 'race'] = 'Black'
            self.df.loc[self.df['race'] == 4, 'race'] = 'Native Hawaiian'
            self.df.loc[self.df['race'] == 5, 'race'] = 'White'
            self.df.loc[self.df['race'] == 6, 'race'] = 'Multi-Race'
            self.df.loc[self.df['race'] == 9, 'race'] = 'Unknown'

            self.df.loc[self.df['race'] == 'Native American', 'race'] = 5
            self.df.loc[self.df['race'] == 'Asian', 'race'] = 4
            self.df.loc[self.df['race'] == 'Black', 'race'] = 3
            self.df.loc[self.df['race'] == 'Native Hawaiian', 'race'] = 5
            self.df.loc[self.df['race'] == 'White', 'race'] = 1
            self.df.loc[self.df['race'] == 'Multi-Race', 'race'] = 5
            self.df.loc[self.df['race'] == 'Unknown', 'race'] = 9
        else:
            raise ValueError("Study number not supported: {}".format(self.study_number))

        self.df['race'] = pd.to_numeric(self.df['race'], downcast="float")


    def standardize_unknown_values_in_predictor_variables(self, search_string=""):
        """Replaces all unknown values in predictor variables with nans.
        This is done to allow the model, or the programmer, to intelligently impute. E.g. xgboost benefits from this, if the data is very sparse.

        Args:
            search_string(str): standardizes the unknown values of any variable defined in self.unknown_class_X, iff that variable name contains 'search_string'
        """
        items = [(var, val) for var, val in self.unknown_class_X.items() if search_string in var]
        for var, unknown_vals in items:
            for uv in unknown_vals:
                self.df.loc[self.df[var] == uv, var] = np.nan


    def standardize_gleason_scores(self):
        """Fills in all three of (gleason_primary, secondary, combined), if possible.

        Primary: 1-5
        Secondary: 1-5
        Combined: 2-10
        Anything else (e.g. multiple studies have 9 or 99 for Unknown): set to nan
        """

        # This line handles the case of the values '9' and '99' referring to unknown gleason for
        # primary/secondary, and combined, respectively.
        self.standardize_unknown_values_in_predictor_variables(search_string="gleason")

        if self.study_number == "9202":
            self.df['gleason_combined'] = self.df['gleason_primary'] + self.df['gleason_secondary']
            self.df.drop(columns=['gleason'], inplace=True)

        elif self.study_number == "9408":
            pass

        elif self.study_number == "9413":
            self.df['gleason_secondary'] = self.df['gleason_combined'] - self.df['gleason_primary']

        elif self.study_number == "9910":
            self.df['gleason_combined'] = self.df['gleason']
            self.df.drop(columns=['gleason'], inplace=True)

        elif self.study_number == "0126":
            self.df['gleason_combined'] = self.df['gleason']
            self.df.drop(columns=['gleason'], inplace=True)

        else:
            raise ValueError("Study number not supported: {}".format(self.study_number))


    def standardize_tstage(self):
        """Consolidate T-stage: T1, T2, T3, T4
        0 - T0
        1 - T1
        2 - T2
        3 - T3
        4 - T4
        """
        if self.study_number == "9202":
            self.df.loc[self.df['tstage'] == 5, 'tstage'] = 1
            self.df.loc[self.df['tstage'] == 6, 'tstage'] = 1
            self.df.loc[self.df['tstage'] == 10, 'tstage'] = 1
            self.df.loc[self.df['tstage'] == 7, 'tstage'] = 2
            self.df.loc[self.df['tstage'] == 8, 'tstage'] = 2
            self.df.loc[self.df['tstage'] == 11, 'tstage'] = 2
            self.df.loc[self.df['tstage'] == 12, 'tstage'] = 3
            self.df.loc[self.df['tstage'] == 13, 'tstage'] = 3
            self.df.loc[self.df['tstage'] == 14, 'tstage'] = 3
            self.df.loc[self.df['tstage'] == 15, 'tstage'] = 4
            self.df.loc[self.df['tstage'] == 16, 'tstage'] = 4

        elif self.study_number == "9408":
            self.df.loc[self.df['tstage'] == 2, 'tstage'] = 1      # T1a -> T1
            self.df.loc[self.df['tstage'] == 3, 'tstage'] = 1      # T1b -> T1
            self.df.loc[self.df['tstage'] == 4, 'tstage'] = 1      # T1c -> T1
            self.df.loc[self.df['tstage'] == 5, 'tstage'] = 2      # T2 Nos -> T2
            self.df.loc[self.df['tstage'] == 6, 'tstage'] = 2      # T2a -> T2
            self.df.loc[self.df['tstage'] == 7, 'tstage'] = 2      # T2b -> T2
            self.df.loc[self.df['tstage'] == 8, 'tstage'] = 2      # T2c -> T2
            self.df.loc[self.df['tstage'] == 10, 'tstage'] = 3      # T3 NOS -> T3
            self.df.loc[self.df['tstage'] == 11, 'tstage'] = 3      # T3a -> T3
            self.df.loc[self.df['tstage'] == 12, 'tstage'] = 3      # T3b -> T3
            self.df.loc[self.df['tstage'] == 13, 'tstage'] = 3      # T3c -> T3
            self.df.loc[self.df['tstage'] == 14, 'tstage'] = 4      # T4 NOS -> T4

        elif self.study_number == "9413":
            self.df.loc[self.df['tstage'] == 2, 'tstage'] = 1      # T2 NOS -> T2
            self.df.loc[self.df['tstage'] == 3, 'tstage'] = 1      # T2 NOS -> T2
            self.df.loc[self.df['tstage'] == 4, 'tstage'] = 1      # T2 NOS -> T2
            self.df.loc[self.df['tstage'] == 5, 'tstage'] = 2      # T2 NOS -> T2
            self.df.loc[self.df['tstage'] == 6, 'tstage'] = 2      # T2a -> T2
            self.df.loc[self.df['tstage'] == 7, 'tstage'] = 2      # T2b -> T2
            self.df.loc[self.df['tstage'] == 8, 'tstage'] = 2      # T2c -> T2
            self.df.loc[self.df['tstage'] == 10, 'tstage'] = 3      # T3 NOS -> T3
            self.df.loc[self.df['tstage'] == 11, 'tstage'] = 3      # T3a -> T3
            self.df.loc[self.df['tstage'] == 12, 'tstage'] = 3      # T3b -> T3
            self.df.loc[self.df['tstage'] == 13, 'tstage'] = 3      # T3c -> T3
            self.df.loc[self.df['tstage'] == 14, 'tstage'] = 4      # T4 NOS-> T4
            self.df.loc[self.df['tstage'] == 15, 'tstage'] = 4      # Undefined in variable listing. 15 and 16 appear in data. Assume T4.
            self.df.loc[self.df['tstage'] == 16, 'tstage'] = 4      #

        elif self.study_number == "9910":
            self.df.loc[self.df['tstage'] == 2, 'tstage'] = 1      # T1a -> T1
            self.df.loc[self.df['tstage'] == 3, 'tstage'] = 1      # T1b -> T1
            self.df.loc[self.df['tstage'] == 4, 'tstage'] = 1      # T1c -> T1
            self.df.loc[self.df['tstage'] == 5, 'tstage'] = 2      # T2 Nos -> T2
            self.df.loc[self.df['tstage'] == 6, 'tstage'] = 2      # T2a -> T2
            self.df.loc[self.df['tstage'] == 7, 'tstage'] = 2      # T2b -> T2
            self.df.loc[self.df['tstage'] == 10, 'tstage'] = 3      # T3 NOS -> T3
            self.df.loc[self.df['tstage'] == 11, 'tstage'] = 3      # T3a -> T3
            self.df.loc[self.df['tstage'] == 12, 'tstage'] = 3      # T3b -> T3
            self.df.loc[self.df['tstage'] == 13, 'tstage'] = 4      # T4 -> T4

        elif self.study_number == "0126":
            self.df.loc[self.df['tstage'] == 2, 'tstage'] = 1      # T1c -> T1
            self.df.loc[self.df['tstage'] == 3, 'tstage'] = 2      # T2a -> T2
            self.df.loc[self.df['tstage'] == 4, 'tstage'] = 2      # T2b -> T2
            self.df.loc[self.df['tstage'] == 5, 'tstage'] = 3      # T3a -> T3
            self.df.loc[self.df['tstage'] == 6, 'tstage'] = 3      # T3b -> T3
            self.df.loc[self.df['tstage'] == 7, 'tstage'] = 4      # T4 -> T4

        else:
            raise ValueError("Study number not supported: {}".format(self.study_number))


    def standardize_pelvic_rt(self):
        """Creates variable pelvic_rt.
        0 - no
        1 - yes

        Instructions:
        Add new category, pelvic RT (yes=1, no=0, unknown=3).
        For 9202, look at pelvic_dose: if "0" then no. If integer, then "1". If blank, then "3".
        For 9408, keep the colum blank because its missing data (will clarify with NRG).
        For 9413: Rx 1 and 3 are "1". Rx 2 and 4 are "0".
        """
        if self.study_number == "9202":
            self.df.loc[self.df_rt['pelvis_dose'] == 0, 'pelvic_rt'] = 0
            self.df.loc[self.df_rt['pelvis_dose'] > 0, 'pelvic_rt'] = 1
            self.df.loc[self.df_rt['pelvis_dose'].isnull(), 'pelvic_rt'] = np.nan

        elif self.study_number == "9408":
            self.df['pelvic_rt'] = np.nan

        elif self.study_number == "9413":
            if 'rx_orig' in self.df.columns: # if we've run standardize_rx()
                column = 'rx_orig'
            else: # if we haven't yet
                column = 'rx'
            self.df.loc[self.df[column] == 1, 'pelvic_rt'] = 1
            self.df.loc[self.df[column] == 3, 'pelvic_rt'] = 1
            self.df.loc[self.df[column] == 2, 'pelvic_rt'] = 0
            self.df.loc[self.df[column] == 4, 'pelvic_rt'] = 0

        elif self.study_number == "9910":
            self.df = pd.merge(self.df, self.df_rt[['cn_deidentified', 'pelvic_rt']], on=['cn_deidentified'], how='left')

        elif self.study_number == "0126":
            print("WARNING: no pelvic_rt data for 0126")

        else:
            raise ValueError("Study number not supported: {}".format(self.study_number))


    def standardize_prostate_dose(self):
        """Creates variable prostate_dose

        Instructions:
        We have this for all but not 9408. Prostate dose=total dose.
        """
        if self.study_number == "9202":
            self.df = pd.merge(self.df, self.df_rt[['cn_deidentified', 'prostate_dose']], on=['cn_deidentified'], how='left')
            #TODO: this line above doesn't match based on cn_deidentified. fix it.

        elif self.study_number == "9408":
            self.df['prostate_dose'] = self.df['prostate_total_dose']

        elif self.study_number == "9413":
            self.df['prostate_dose'] = self.df['rt_total_dose']

        elif self.study_number == "9910":
            self.df['prostate_dose'] = self.df_rt['rt_total_dose']

        elif self.study_number == "0126":
            self.df['prostate_dose'] = self.df['rt_total_dose']

        else:
            raise ValueError("Study number not supported: {}".format(self.study_number))


    def standardize_rt_complete(self):
        """Creates variable rt_complete

        Instructions(Osama)
        Follow the code in 9202:
            1 - No
            2 - Yes
            9 - Unknown

        9408 and 9413:
            create a new category from "RT end reason prior to total dose":
            0=2; 1 and 2 and 3 and 4 and 5= 1; missing=9.
            Please, if data missing, please review rt_total_dose and if >60, then consider as RT complete or "2"
        """
        if self.study_number == "9202":
#           gcp_path = "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/NRG Statistics/RTOG 9202/All_RT_Data_9202.xlsx"
#           rt_df = pd.read_excel(gcp_path)
            self.df['rt_complete'] = self.df_rt['rt_complete']

            self.unknown_class_X['rt_complete'] = [9]

        elif self.study_number == "9408":
            self.df.loc[self.df['rt_end_reason'] == 0, 'rt_complete'] = 2
            self.df.loc[self.df['rt_end_reason'] == 1, 'rt_complete'] = 1
            self.df.loc[self.df['rt_end_reason'] == 2, 'rt_complete'] = 1
            self.df.loc[self.df['rt_end_reason'] == 3, 'rt_complete'] = 1
            self.df.loc[self.df['rt_end_reason'] == 4, 'rt_complete'] = 1
            self.df.loc[self.df['rt_end_reason'] == 5, 'rt_complete'] = 1

            for index, row in self.df.iterrows():
                if np.isnan(row['rt_complete']) and row['prostate_total_dose'] >= 60:
                    self.df.loc[index, 'rt_complete'] = 2

        elif self.study_number == "9413":
            self.df.loc[self.df['rt_end_reason'] == 0, 'rt_complete'] = 2
            self.df.loc[self.df['rt_end_reason'] == 1, 'rt_complete'] = 1
            self.df.loc[self.df['rt_end_reason'] == 2, 'rt_complete'] = 1
            self.df.loc[self.df['rt_end_reason'] == 3, 'rt_complete'] = 1
            self.df.loc[self.df['rt_end_reason'] == 4, 'rt_complete'] = 1
            self.df.loc[self.df['rt_end_reason'] == 5, 'rt_complete'] = 1

            for index, row in self.df.iterrows():
                if np.isnan(row['rt_complete']) and row['rt_total_dose'] >= 60:
                    self.df.loc[index, 'rt_complete'] = 2

        elif self.study_number == "9910":
            self.df = pd.merge(self.df,
                               self.df_rt[['cn_deidentified', 'rt_end_reason', 'rt_total_dose']],
                               on=['cn_deidentified'],
                               how='left')
            self.df.loc[self.df['rt_end_reason'] == 1, 'rt_complete'] = 2
            self.df.loc[self.df['rt_end_reason'] == 2, 'rt_complete'] = 1
            self.df.loc[self.df['rt_end_reason'] == 3, 'rt_complete'] = 1
            self.df.loc[self.df['rt_end_reason'] == 4, 'rt_complete'] = 1
            self.df.loc[self.df['rt_end_reason'] == 5, 'rt_complete'] = 1
            self.df.loc[self.df['rt_end_reason'] == 6, 'rt_complete'] = 1
            self.df.loc[self.df['rt_end_reason'] == 7, 'rt_complete'] = 1
            self.df.loc[self.df['rt_end_reason'] == 8, 'rt_complete'] = 1

            for index, row in self.df.iterrows():
                if np.isnan(row['rt_complete']) and row['rt_total_dose'] >= 60:
                    self.df.loc[index, 'rt_complete'] = 2

        elif self.study_number == "0126":
            self.df.loc[self.df['rt_end_reason'] == 1, 'rt_complete'] = 2
            self.df.loc[self.df['rt_end_reason'] == 2, 'rt_complete'] = 2
            self.df.loc[self.df['rt_end_reason'] == 3, 'rt_complete'] = 1
            self.df.loc[self.df['rt_end_reason'] == 4, 'rt_complete'] = 1
            self.df.loc[self.df['rt_end_reason'] == 5, 'rt_complete'] = 1
            self.df.loc[self.df['rt_end_reason'] == 6, 'rt_complete'] = 1
            self.df.loc[self.df['rt_end_reason'] == 7, 'rt_complete'] = 1
            self.df.loc[self.df['rt_end_reason'] == 8, 'rt_complete'] = 1
            self.df.loc[self.df['rt_end_reason'] == 9, 'rt_complete'] = 9

            for index, row in self.df.iterrows():
                if np.isnan(row['rt_complete']) and row['rt_total_dose'] >= 60:
                    self.df.loc[index, 'rt_complete'] = 2

        else:
            raise ValueError("Study number not supported: {}".format(self.study_number))


    def cause_of_death(self):
        """Standardizes cause of death to [0,1,2]

        0: Other / Unknown
        1: Prostate Cancer
        2: Complications, related to a treatment
        """
        if self.study_number == "9202":
            self.df.loc[self.df['cod'] == 1, 'cod'] = 'Cancer'
            self.df.loc[self.df['cod'] == 2, 'cod'] = 'Cancer'
            self.df.loc[self.df['cod'] == 3, 'cod'] = 'Complications'
            self.df.loc[self.df['cod'] == 4, 'cod'] = 'Complications'
            self.df.loc[self.df['cod'] == 5, 'cod'] = 'Other'
            self.df.loc[self.df['cod'] == 9, 'cod'] = 'Other'

            self.df.loc[self.df['cod'] == 'Cancer', 'cod'] = 1
            self.df.loc[self.df['cod'] == 'Complications', 'cod'] = 2
            self.df.loc[self.df['cod'] == 'Other', 'cod'] = 0

        elif self.study_number == "9408":
            self.df.loc[self.df['cod'] == 1, 'cod'] = 'Cancer'
            self.df.loc[self.df['cod'] == 2, 'cod'] = 'Cancer'
            self.df.loc[self.df['cod'] == 3, 'cod'] = 'Complications'
            self.df.loc[self.df['cod'] == 4, 'cod'] = 'Complications'
            self.df.loc[self.df['cod'] == 5, 'cod'] = 'Other'
            self.df.loc[self.df['cod'] == 9, 'cod'] = 'Other'

            self.df.loc[self.df['cod'] == 'Cancer', 'cod'] = 1
            self.df.loc[self.df['cod'] == 'Complications', 'cod'] = 2
            self.df.loc[self.df['cod'] == 'Other', 'cod'] = 0

        elif self.study_number == "9413":
            self.df.loc[self.df['cod'] == 1, 'cod'] = 'Cancer'
            self.df.loc[self.df['cod'] == 2, 'cod'] = 'Complications'
            self.df.loc[self.df['cod'] == 3, 'cod'] = 'Other'
            self.df.loc[self.df['cod'] == 4, 'cod'] = 'Other'

            self.df.loc[self.df['cod'] == 'Cancer', 'cod'] = 1
            self.df.loc[self.df['cod'] == 'Complications', 'cod'] = 2
            self.df.loc[self.df['cod'] == 'Other', 'cod'] = 0

        elif self.study_number == "9910":
            self.df.loc[self.df['cod'] == 1, 'cod'] = 'Cancer'
            self.df.loc[self.df['cod'] == 2, 'cod'] = 'Complications'
            self.df.loc[self.df['cod'] == 3, 'cod'] = 'Other'
            self.df.loc[self.df['cod'] == 4, 'cod'] = 'Other'

            self.df.loc[self.df['cod'] == 'Cancer', 'cod'] = 1
            self.df.loc[self.df['cod'] == 'Complications', 'cod'] = 2
            self.df.loc[self.df['cod'] == 'Other', 'cod'] = 0

        elif self.study_number == "0126":
            self.df.loc[self.df['cod'] == 'Treated Cancer', 'cod'] = 'Cancer'
            self.df.loc[self.df['cod'] == 'Complications of protocol trea', 'cod'] = 'Complications'
            self.df.loc[self.df['cod'] == 'Other', 'cod'] = 'Other'
            self.df.loc[self.df['cod'] == 'Unknown', 'cod'] = 'Other'

            self.df.loc[self.df['cod'] == 'Cancer', 'cod'] = 1
            self.df.loc[self.df['cod'] == 'Complications', 'cod'] = 2
            self.df.loc[self.df['cod'] == 'Other', 'cod'] = 0

        else:
            raise ValueError("Study number not supported: {}".format(self.study_number))


#   def XXXX(self):
#       if self.study_number == "9202":
#           pass
#       elif self.study_number == "9408":
#           pass
#       elif self.study_number == "9413":
#           pass

#       else:
#           raise ValueError("Study number not supported: {}".format(self.study_number))


