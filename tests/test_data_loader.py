import os
import pandas as pd
from data.data_loader import load_and_clean_20newsgroups, load_and_clean_agnews


def test_load_and_clean_20newsgroups():
    path = "data/cleaned_20news.csv"
    load_and_clean_20newsgroups(path)
    assert os.path.exists(path), "20 Newsgroups file was not created"
    df = pd.read_csv(path)
    assert not df.empty, "Loaded DataFrame is empty"
    assert 'text' in df.columns and 'label' in df.columns, "Required columns missing"


def test_load_and_clean_agnews():
    path = "data/cleaned_agnews.csv"
    load_and_clean_agnews(path)
    assert os.path.exists(path), "AG News file was not created"
    df = pd.read_csv(path)
    assert not df.empty, "Loaded DataFrame is empty"
    assert 'text' in df.columns and 'label' in df.columns, "Required columns missing"
