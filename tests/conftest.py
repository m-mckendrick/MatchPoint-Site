import pandas as pd
import pytest
from datetime import datetime

from src.data_preprocessing.convert_to_p1p2 import convert_to_p1p2

#raw df example for testing
#inclueds 2 matches before cleaning
@pytest.fixture
def raw_match_df():
    return pd.DataFrame([
        {
            "tourney_id": "2024-580",
            "tourney_name": "Australian Open",
            "surface": "Hard",
            "draw_size": 128,
            "tourney_level": "G",
            "tourney_date": 20240115,
            "match_num": 131,

            "winner_id": 126094,
            "winner_seed": 5,
            "winner_entry": "",
            "winner_name": "Andrey Rublev",
            "winner_hand": "R",
            "winner_ht": 188,
            "winner_ioc": "RUS",
            "winner_age": 26.2,

            "loser_id": 205734,
            "loser_seed": "",
            "loser_entry": "",
            "loser_name": "Thiago Seyboth Wild",
            "loser_hand": "R",
            "loser_ht": 185,
            "loser_ioc": "BRA",
            "loser_age": 23.8,

            "score": "7-5 6-4 3-6 4-6 7-6(6)",
            "best_of": 5,
            "round": "R128",
            "minutes": 222,

            # winner serve-return stats
            "w_ace": 21,
            "w_df": 1,
            "w_svpt": 154,
            "w_1stIn": 100,
            "w_1stWon": 83,
            "w_2ndWon": 25,
            "w_SvGms": 27,
            "w_bpSaved": 4,
            "w_bpFaced": 8,

            #loser serve-return stats
            "l_ace": 15,
            "l_df": 6,
            "l_svpt": 202,
            "l_1stIn": 130,
            "l_1stWon": 91,
            "l_2ndWon": 34,
            "l_SvGms": 26,
            "l_bpSaved": 19,
            "l_bpFaced": 23,

            "winner_rank": 5,
            "winner_rank_points": 5010,
            "loser_rank": 77,
            "loser_rank_points": 718,
        },
        {
            "tourney_id": "2024-580",
            "tourney_name": "Australian Open",
            "surface": "Hard",
            "draw_size": 128,
            "tourney_level": "G",
            "tourney_date": 20240115,
            "match_num": 211,

            "winner_id": 207989,
            "winner_seed": 2,
            "winner_entry": "",
            "winner_name": "Carlos Alcaraz",
            "winner_hand": "R",
            "winner_ht": 183,
            "winner_ioc": "ESP",
            "winner_age": 20.6,

            "loser_id": 209992,
            "loser_seed": "",
            "loser_entry": "WC",
            "loser_name": "Juncheng Shang",
            "loser_hand": "L",
            "loser_ht": 180,
            "loser_ioc": "CHN",
            "loser_age": 18.9,

            "score": "6-1 6-1 1-0 RET",
            "best_of": 5,
            "round": "R32",
            "minutes": 66,

            "w_ace": 3,
            "w_df": 1,
            "w_svpt": 34,
            "w_1stIn": 19,
            "w_1stWon": 16,
            "w_2ndWon": 13,
            "w_SvGms": 8,
            "w_bpSaved": 0,
            "w_bpFaced": 0,

            "l_ace": 1,
            "l_df": 0,
            "l_svpt": 50,
            "l_1stIn": 26,
            "l_1stWon": 13,
            "l_2ndWon": 7,
            "l_SvGms": 7,
            "l_bpSaved": 6,
            "l_bpFaced": 12,

            "winner_rank": 2,
            "winner_rank_points": 8855,
            "loser_rank": 140,
            "loser_rank_points": 435,
        }
    ])


#cleaned df with matches that have been cleaned
@pytest.fixture
def cleaned_match_df(raw_match_df):
    from src.data_preprocessing.clean_pipeline import clean_matches
    return clean_matches(raw_match_df.copy())

#new fixture that converts cleaned df to model ready df with 2 perspectives
@pytest.fixture
def p1p2_match_df(cleaned_match_df):
    from src.data_preprocessing.clean_pipeline import convert_for_model
    return convert_for_model(cleaned_match_df.copy())

#pytest fixture for explainability tests
from src.prediction.loader import load_model, load_matches, load_rankings
from src.model_training.logistical_regression import prepare_dataset

@pytest.fixture(scope="session")
def model_bundle():
    model, scaler = load_model()
    return {"model": model, "scaler": scaler}

@pytest.fixture(scope="session")
def full_data():
    df = load_matches()
    rankings = load_rankings()
    return {"df": df, "rankings": rankings}

@pytest.fixture(scope="session")
def model_features(full_data):
    df = full_data["df"]
    x, y = prepare_dataset(df)
    return x.columns.tolist()

@pytest.fixture
def synthetic_row(model_features):
    # Create a minimal synthetic row with correct columns
    data = {col: 0 for col in model_features}

    # Add some realistic values
    if "p1_rank" in data: data["p1_rank"] = 1
    if "p2_rank" in data: data["p2_rank"] = 5
    if "p1_age" in data: data["p1_age"] = 35
    if "p2_age" in data: data["p2_age"] = 27
    if "p1_ht" in data: data["p1_ht"] = 188
    if "p2_ht" in data: data["p2_ht"] = 192
    if "surface_Hard" in data: data["surface_Hard"] = 1
    if "round_encoded" in data: data["round_encoded"] = 5
    if "is_best_of_five" in data: data["is_best_of_five"] = 1

    return pd.DataFrame([data])

