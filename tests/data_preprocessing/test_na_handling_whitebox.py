import pandas as pd
import numpy as np

from src.data_preprocessing.na_handling import (
    fill_missing_height,
    fill_missing_hand,
    fill_missing_minutes,
    drop_unused_columns,
    fill_missing_serve_stats,
)

"""
For a single player missing heights should be filled
using the nearest known height. 190 should be sent to both sides
"""
def test_fill_missing_height_whitebox_ffill_bfill():
    df = pd.DataFrame({
        "p1_id": [1, 1, 1],
        "p1_ht": [None, 190, None]
    })

    out = fill_missing_height(df)

    assert out["p1_ht"].tolist() == [190, 190, 190]


"""
same forward backward fill check
and if no hand history set to "right"
"""
def test_fill_missing_hand_whitebox_default_right():
    df = pd.DataFrame({
        "p1_id": [1, 2],
        "p1_hand": [None, None]
    })

    out = fill_missing_hand(df)

    assert out["p1_hand"].tolist() == ["Right", "Right"]


"""
if match is a retirement minutes = 0 and else 
"""
def test_fill_missing_minutes_whitebox_branches():
    df = pd.DataFrame({
        "minutes": [None, None],
        "is_retirement": [1, 0]
    })

    out = fill_missing_minutes(df)

    assert out["minutes"].iloc[0] == 0      # retirement branch
    assert out["minutes"].iloc[1] == 0      # median of [0] = 0

#make sure only chose columns are removed
def test_drop_unused_columns_whitebox_missing_cols():
    df = pd.DataFrame({"keep": [1]})
    out = drop_unused_columns(df)
    assert list(out.columns) == ["keep"]

#only fill serve columns
def test_fill_missing_serve_stats_whitebox_partial_cols():
    df = pd.DataFrame({"p1_ace": [None], "other": [5]})
    out = fill_missing_serve_stats(df)
    assert out["p1_ace"].iloc[0] == 0
    assert out["other"].iloc[0] == 5
