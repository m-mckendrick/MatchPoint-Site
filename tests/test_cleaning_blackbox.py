import pandas as pd
import numpy as np
from src.data_preprocessing.clean_pipeline import (
    normalise_headers,
    parse_dates,
    convert_numeric,
    clean_ranks_and_seeds,
    map_hands,
    clean_categories,
    validate_physical_stats,
    filter_atp_events,
    clean_matches
)


#black-box tests
#test output of functions

#testing header to lowercase etc
def test_normalise_headers_basic():
    df = pd.DataFrame(columns=[" Winner Rank ", "loser-rank"])
    out = normalise_headers(df)
    assert list(out.columns) == ["winner_rank", "loser_rank"]

#testing date are turned to datatime correctly
def test_parse_dates_valid_and_invalid():
    df = pd.DataFrame({"tourney_date": ["20240101", "bad"]})
    out = parse_dates(df)
    assert pd.isna(out["tourney_date"].iloc[1])
    assert out["tourney_date"].iloc[0].year == 2024


#testing teh numeric convert for all the stats
def test_convert_numeric():
    df = pd.DataFrame({
        "w_aces": ["10", "bad"],
        "l_df": ["3", None]
    })
    out = convert_numeric(df)
    assert out["w_aces"].dtype == "float64"
    assert pd.isna(out["w_aces"].iloc[1])


#all rankings and seed to integers
def test_clean_ranks_and_seeds():
    df = pd.DataFrame({
        "winner_seed": ["1", None],
        "loser_seed": ["2", "3"],
        "winner_rank": ["10", "20"],
        "loser_rank": ["30", None],
        "winner_rank_points": ["1000", "2000"],
        "loser_rank_points": ["500", None]
    })
    out = clean_ranks_and_seeds(df)
    assert str(out["winner_seed"].dtype) == "Int64"
    assert pd.isna(out["winner_seed"].iloc[1])

#test r == right, u == unknown
def test_map_hands():
    df = pd.DataFrame({"winner_hand": ["R"], "loser_hand": ["U"]})
    out = map_hands(df)
    assert out["winner_hand"].iloc[0] == "Right"
    assert out["loser_hand"].iloc[0] == "Unknown"


def test_clean_categories():
    df = pd.DataFrame({"surface": ["hard"], "round": ["qf"]})
    out = clean_categories(df)
    assert out["surface"].iloc[0] == "Hard"
    assert out["round"].iloc[0] == "QF"

#testing laver cup is removed and aussie open kept
def test_filter_atp_events():
    df = pd.DataFrame({
        "tourney_level": ["A", "X"],
        "tourney_name": ["Australian Open", "Laver Cup"]
    })
    out = filter_atp_events(df)
    assert len(out) == 1
    assert out["tourney_level"].iloc[0] == "A"

#testing full pipeline
def test_clean_matches_full_pipeline():
    df = pd.DataFrame([{
        "winner_seed": 1,
        "loser_seed": 2,
        "winner_rank": 10,
        "loser_rank": 20,
        "winner_rank_points": 1000,
        "loser_rank_points": 500,
        "winner_hand": "R",
        "loser_hand": "L",
        "surface": "hard",
        "round": "qf",
        "winner_ht": 185,
        "loser_ht": 190,
        "winner_age": 25,
        "loser_age": 30,
        "best_of": 3,
        "minutes": 120,
        "tourney_level": "A",
        "tourney_name": "Australian Open",
        "tourney_date": "20240101"
    }])

    out = clean_matches(df)

    # Check that key transformations happened
    assert out["surface"].iloc[0] == "Hard"
    assert out["winner_hand"].iloc[0] == "Right"
    assert out["winner_rank"].dtype == "Int64"
    assert out["tourney_date"].iloc[0].year == 2024
