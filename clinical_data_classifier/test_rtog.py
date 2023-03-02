"""Unittests for NRG Data

test_variable_listing:
    defensive test to ensure that the variables in "Variable Listing" .docx
    actually match what's in baseline_and_efficacy + radiotherapy .xlsx
    As of the documents on 2/14/2021, they match and these tests pass.
"""

import unittest
from rtog_constants import rtog_variable_listings, rtog_field_mapping
from rtog_helper import rtog_from_study_number
from datetime import datetime
import numpy as np


def in_domain(var, domain):
    if domain == 'c':
        return np.isreal(var)
    if domain == 's':
        return type(var) == str
    if domain == 'i':
        return int(var) == var
    if domain == 'd':
        return type(var) == datetime or type(var) == np.datetime64
    if type(domain) == set:
        return var in domain
    raise ValueError("Domain type unrecognized: {}".format(domain))


class TestRTOG9202(unittest.TestCase):

    def test_variable_listing(self):
        sn = '9202'
        r = rtog_from_study_number(sn, create_endpoints=False)
        vl = rtog_variable_listings[sn]

        for col in r.df.columns:
            uv = r.df[col].unique()
            ev = vl[col]
            for u in uv:
                if (not type(u) == str) and (np.isnan(u)):
                    continue
                contained = [in_domain(u, e) for e in ev]
                assert np.any(contained), "Var {} in column {} is not in expected values {}".format(u, col, ev)


class TestRTOG9413(unittest.TestCase):

    def test_variable_listing(self):
        sn = '9413'
        r = rtog_from_study_number(sn, create_endpoints=False)
        vl = rtog_variable_listings[sn]

        for col in r.df.columns:
            uv = r.df[col].unique()
            ev = vl[col]
            for u in uv:
                if (not type(u) == str) and (np.isnan(u)):
                    continue
                contained = [in_domain(u, e) for e in ev]
                assert np.any(contained), "Var {} in column {} is not in expected values {}".format(u, col, ev)


class TestRTOG9408(unittest.TestCase):

    def test_variable_listing(self):
        sn = '9408'
        r = rtog_from_study_number(sn, create_endpoints=False)
        vl = rtog_variable_listings[sn]

        for col in r.df.columns:
            uv = r.df[col].unique()
            ev = vl[col]
            for u in uv:
                if (not type(u) == str) and (np.isnan(u)):
                    continue
                contained = [in_domain(u, e) for e in ev]
                assert np.any(contained), "Var {} in column {} is not in expected values {}".format(u, col, ev)


class TestRTOG9910(unittest.TestCase):

    def test_variable_listing(self):
        sn = '9910'
        r = rtog_from_study_number(sn, create_endpoints=False)
        vl = rtog_variable_listings[sn]

        for col in r.df.columns:
            uv = r.df[col].unique()
            ev = vl[col]
            for u in uv:
                if (not type(u) == str) and (np.isnan(u)):
                    continue
                contained = [in_domain(u, e) for e in ev]
                assert np.any(contained), "Var {} in column {} is not in expected values {}".format(u, col, ev)


class TestRTOG0126(unittest.TestCase):

    def test_variable_listing(self):
        sn = '0126'
        r = rtog_from_study_number(sn, create_endpoints=False)
        vl = rtog_variable_listings[sn]

        for col in r.df.columns:
            uv = r.df[col].unique()
            ev = vl[col]
            for u in uv:
                if (not type(u) == str) and (np.isnan(u)):
                    continue
                contained = [in_domain(u, e) for e in ev]
                assert np.any(contained), "Var {} in column {} is not in expected values {}".format(u, col, ev)

