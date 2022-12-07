#!/usr/bin/env python

"""Tests for `dataprocess` package."""

import pytest
import pandas as pd

from dataprocess import preprocessing


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')

@pytest.fixture # Variable to use as input of a function
def df():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    return pd.read_csv('./tests/data_test.csv', encoding='ISO-8859-1')

def test_one_hot_encoder(df):
    """Sample pytest test function with the pytest fixture as an argument."""
    out = preprocessing.one_hot_encoder(df,['Country'])
    
    assert out.shape[1] == 39
