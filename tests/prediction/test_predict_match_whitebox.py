import pandas as pd
from datetime import datetime

from src.prediction.match_builder_modular import (
    get_player_ranking,
    build_player_profile,
    get_tournament_metadata,
    build_match
)

from data.loader import load_model, load_features, load_rankings

# ============================
# RANKING TESTS
# ============================

def test_get_player_ranking_whitebox():
    rankings = pd.DataFrame({
        "player_id": [1, 1, 1],
        "ranking_date": [
            datetime(2018, 1, 1),
            datetime(2018, 6, 1),
            datetime(2018, 8, 20)
        ],
        "rank": [10, 5, 3],
        "points": [3000, 4500, 6000]
    })

    result = get_player_ranking(rankings, 1, datetime(2018, 8, 25))

    assert result["rank"] == 3
    assert result["rank_points"] == 6000


def test_ranking_freshness():
    rankings = pd.DataFrame({
        "player_id": [1, 1, 1, 1],
        "ranking_date": [
            datetime(2018, 7, 1),
            datetime(2018, 8, 1),
            datetime(2018, 8, 20),
            datetime(2018, 8, 27),
        ],
        "rank": [20, 12, 8, 6],
        "points": [2000, 3000, 4000, 5000]
    })

    tournament_date = datetime(2018, 8, 27)

    result = get_player_ranking(rankings, 1, tournament_date)
    assert result["rank"] == 6
    assert result["rank_points"] == 5000


# ============================
# PLAYER PROFILE TEST
# ============================

def test_build_player_profile_whitebox():
    df = pd.DataFrame({
        "tourney_date": [datetime(2018, 1, 1), datetime(2018, 6, 1)],
        "p1_name": ["Novak Djokovic", "Other"],
        "p2_name": ["Other", "Novak Djokovic"],
        "p1_id": [1, 2],
        "p2_id": [2, 1],
        "p1_age": [30, 28],
        "p2_age": [28, 30],
        "p1_ht": [188, 185],
        "p2_ht": [185, 188],
        "p1_hand": ["R", "L"],
        "p2_hand": ["L", "R"],
        "p1_ioc": ["SRB", "GBR"],
        "p2_ioc": ["GBR", "SRB"],
        "p1_elo_global": [1500, 1400],
        "p2_elo_global": [1400, 1500],
        "p1_elo_surface": [1500, 1400],
        "p2_elo_surface": [1400, 1500],
    })

    rankings = pd.DataFrame({
        "player_id": [1],
        "ranking_date": [datetime(2018, 5, 1)],
        "rank": [2],
        "points": [5000]
    })

    profile = build_player_profile(
        df=df,
        rankings_df=rankings,
        player_name="Novak Djokovic",
        cutoff_date=datetime(2018, 7, 1)
    )

    assert profile["rank"] == 2
    assert profile["rank_points"] == 5000
    assert profile["age"] == 30
    assert profile["height"] == 188
    assert profile["hand"] == "R"
    assert profile["ioc"] == "SRB"


# ============================
# TOURNAMENT METADATA TEST
# ============================

def test_get_tournament_metadata_whitebox():
    df = pd.DataFrame({
        "tourney_id": ["123-US"],
        "tourney_name": ["US Open"],
        "match_year": [2018],
        "tourney_date": [datetime(2018, 8, 27)],
        "surface_raw": ["Hard"],
        "best_of": [5],
        "tourney_level": ["G"],
        "draw_size": [128]
    })

    meta = get_tournament_metadata(df, "US Open", 2018)

    assert meta["surface"] == "Hard"
    assert meta["best_of"] == 5
    assert meta["tourney_level"] == "G"
    assert meta["draw_size"] == 128


# ============================
# BUILD MATCH TEST
# ============================

def test_build_match_whitebox():
    df = load_features()

    rankings = load_rankings()

    model, scaler, model_features = load_model("models/xgboost_3.pkl")

    row_model, row_full = build_match(
        df=df,
        rankings_df=rankings,
        p1_name="Novak Djokovic",
        p2_name="Kevin Anderson",
        tournament="US Open",
        year=2018,
        round_label="F",
        model_features=model_features,
        debug=False,
    )

    assert set(row_model.columns) == set(model_features)
    assert row_model.iloc[0]["p1_rank"] == 6
    assert row_model.iloc[0]["p2_rank"] == 5
    assert row_model.iloc[0]["surface_Hard"] == 1


# ============================
# MODEL ALIGNMENT TEST
# ============================

def test_build_match_model_alignment_whitebox():
    df = load_features()
    rankings = load_rankings()
    model, scaler, model_features = load_model("models/xgboost_3.pkl")

    row_model, _ = build_match(
        df=df,
        rankings_df=rankings,
        p1_name="Novak Djokovic",
        p2_name="Daniil Medvedev",
        tournament="US Open",
        year=2018,
        round_label="F",
        model_features=model_features,
    )

    assert set(row_model.columns) == set(model_features)
    assert row_model.isna().sum().sum() == 0


# ============================
# SYNTHETIC FORM TEST
# ============================

def test_synthetic_form_correctness_whitebox():
    df = load_features()
    rankings = load_rankings()
    model, scaler, model_features = load_model("models/xgboost_3.pkl")

    from src.feature_engineering.feature_builder import FEATURE_CONFIG
    FEATURE_CONFIG["fatigue"] = False

    row_model, row_full = build_match(
        df=df,
        rankings_df=rankings,
        p1_name="Novak Djokovic",
        p2_name="Daniil Medvedev",
        tournament="US Open",
        year=2018,
        round_label="F",
        model_features=model_features,
    )

    syn = row_full.iloc[-1]

    form_cols = [c for c in syn.index if "form_" in c]
    assert len(form_cols) > 0
    assert any(syn[c] != 0 for c in form_cols)
