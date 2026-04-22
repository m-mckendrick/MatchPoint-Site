import pandas as pd
import numpy as np
from src.feature_engineering.match_context import add_pre_match_context_features, encode_round, assign_tour_finals_country, TOURNAMENT_COUNTRY , add_indoor_feature

#white-box tests
#these test every edge case and branch (every line of code)_row


#if round recorded as RR then encoded round cannot be encoded with algorithm
def test_encode_round_invalid_round():
    row = {
        "tourney_name": "Adelaide",
        "draw_size": 32,
        "round": "RR"
    }
    assert encode_round(row) is None

#test round encoder works if round number isn't a power of 2 (round is still 1st)
def test_encode_round_non_power_of_two_draw():
    row = {
        "tourney_name": "Adelaide",
        "draw_size": 28,
        "round": "R32"
    }
    assert encode_round(row) == 1

#if tourney level isn't known eg z (or could be one wasnt removed during cleaning like Olympics)
def test_prestige_unknown_level():
    df = pd.DataFrame([{
        "p1_rank": 10,
        "p2_rank": 20,
        "p1_rank_points": 3000,
        "p2_rank_points": 1500,
        "p1_seed": 1,
        "p2_seed": 2,
        "p1_hand": "R",
        "p2_hand": "R",
        "tourney_name": "Wimbledon",
        "surface": "Grass",
        "round": "F",
        "draw_size": 32,
        "tourney_level": "Z",
        "best_of": 3,
        "tourney_date": "2024-01-01",
        "p1_age": 30,
        "p2_age": 25,
        "p1_ht": 190,
        "p2_ht": 185,
        "p1_ioc" : "GBR",
        "p2_ioc" : "ESP"
    }])

    out = add_pre_match_context_features(df)

    #prestige should be NaN
    assert np.isnan(out["tourney_prestige"].iloc[0])

#testing all the combination of the lefty matchup indicators
def test_lefty_logic_all_combinations():
    df = pd.DataFrame([
        {"p1_hand": "Left", "p2_hand": "Left"},
        {"p1_hand": "Left", "p2_hand": "R"},
        {"p1_hand": "Right", "p2_hand": "Left"},
        {"p1_hand": "Right", "p2_hand": "Right"},
        {"p1_hand": "unknown", "p2_hand": "Right"},
        {"p1_hand": "unknown", "p2_hand": "Left"},
    ])

    df["p1_rank"] = 1
    df["p2_rank"] = 2
    df["p1_rank_points"] = 1000
    df["p2_rank_points"] = 900
    df["p1_seed"] = 1
    df["p2_seed"] = 2
    df["surface"] = "Hard"
    df["tourney_name"] = "Adelaide"
    df["round"] = "R32"
    df["draw_size"] = 32
    df["tourney_level"] = "A"
    df["best_of"] = 3
    df["tourney_date"] = "2024-01-01"
    df["p1_age"] = 30
    df["p2_age"] = 25
    df["p1_ht"] = 190
    df["p2_ht"] = 185
    df["p1_ioc"] = "AUS"
    df["p2_ioc"] = "ESP"

    out = add_pre_match_context_features(df)

    assert list(out["lefty_diff"]) == [0, 1, -1, 0, 0, -1]

def test_assign_tour_finals_country_branches():
    # houston era
    row = {"tourney_name": "Masters Cup", "tourney_date": pd.Timestamp("2003-11-10")}
    assert assign_tour_finals_country(row) == "USA"

    # shanghai era
    row = {"tourney_name": "Masters Cup", "tourney_date": pd.Timestamp("2007-11-10")}
    assert assign_tour_finals_country(row) == "CHN"

    # london era
    row = {"tourney_name": "Tour Finals", "tourney_date": pd.Timestamp("2015-11-10")}
    assert assign_tour_finals_country(row) == "GBR"

    # turin era
    row = {"tourney_name": "ATP Finals", "tourney_date": pd.Timestamp("2022-11-10")}
    assert assign_tour_finals_country(row) == "ITA"


def test_indoor_madrid_years():
    df = pd.DataFrame([
        {"tourney_name": "Madrid Masters", "tourney_date": "2005-05-10"},
        {"tourney_name": "Madrid Masters", "tourney_date": "2015-05-10"},
    ])
    df["tourney_date"] = pd.to_datetime(df["tourney_date"])

    out = add_indoor_feature(df)

    assert out["is_indoor"].iloc[0] == 1   # indoor era
    assert out["is_indoor"].iloc[1] == 0   # outdoor era


def test_indoor_tour_finals():
    df = pd.DataFrame([
        {"tourney_name": "Masters Cup", "tourney_date": "2002-11-10"},  # Houston outdoor
        {"tourney_name": "Masters Cup", "tourney_date": "2006-11-10"},  # Shanghai indoor
        {"tourney_name": "ATP Finals", "tourney_date": "2015-11-10"},   # London indoor
    ])
    df["tourney_date"] = pd.to_datetime(df["tourney_date"])

    out = add_indoor_feature(df)

    assert out["is_indoor"].tolist() == [0, 1, 1]
