import pandas as pd
import numpy as np

from src.data_preprocessing.na_handling import (
    fill_missing_serve_stats,
    fill_missing_score,
    fill_missing_height,
    fill_missing_hand,
    drop_unused_columns,
    fill_missing_rank,
    fill_missing_minutes,
    handle_na,
)


def test_fill_missing_serve_stats():
    df = pd.DataFrame({
        "p1_ace": [1, None],
        "p2_df": [None, 3],
        "other_col": [5, 6]
    })

    out = fill_missing_serve_stats(df)

    assert out["p1_ace"].tolist() == [1, 0]
    assert out["p2_df"].tolist() == [0, 3]
    assert out["other_col"].tolist() == [5, 6]

def test_fill_missing_score():
    df = pd.DataFrame({"score": ["6-4 6-4", None]})
    out = fill_missing_score(df)
    assert out["score"].tolist() == ["6-4 6-4", "0-0"]

def test_fill_missing_height():
    df = pd.DataFrame({
        "p1_id": [1, 1, 2],
        "p1_ht": [180, None, None]
    })

    out = fill_missing_height(df)

    #both fallback 180 previous and 185 median
    assert out["p1_ht"].tolist() == [180, 180, 185]

def test_fill_missing_hand():
    df = pd.DataFrame({
        "p1_id": [1, 1, 2],
        "p1_hand": ["Right", None, None]
    })

    out = fill_missing_hand(df)

    assert out["p1_hand"].tolist() == ["Right", "Right", "Right"]

def test_drop_unused_columns():
    df = pd.DataFrame({
        "p1_seed": [1],
        "p2_entry": ["WC"],
        "keep": [10]
    })

    out = drop_unused_columns(df)

    assert "p1_seed" not in out.columns
    assert "p2_entry" not in out.columns
    assert "keep" in out.columns

def test_fill_missing_rank():
    df = pd.DataFrame({
        "p1_rank": [10, None],
        "p2_rank_points": [None, 50]
    })

    out = fill_missing_rank(df)

    assert out["p1_rank"].tolist() == [10, 500]
    assert out["p2_rank_points"].tolist() == [0, 50]

def test_fill_missing_minutes():
    df = pd.DataFrame({
        "minutes": [50, None, None],
        "is_retirement": [0, 1, 0]
    })

    out = fill_missing_minutes(df)

    assert out["minutes"].tolist() == [50, 0, 25]

def test_handle_na():
    df = pd.DataFrame({
        "score": [None],
        "p1_ace": [None],
        "p1_rank": [None],
        "p1_ht": [None],
        "p1_id": [1],
        "p1_hand": [None],
        "minutes": [None],
        "is_retirement": [1]
    })

    out = handle_na(df)

    assert out["score"].iloc[0] == "0-0"
    assert out["p1_ace"].iloc[0] == 0
    assert out["p1_rank"].iloc[0] == 500
    assert out["p1_ht"].iloc[0] == 185
    assert out["p1_hand"].iloc[0] == "Right"
    assert out["minutes"].iloc[0] == 0

