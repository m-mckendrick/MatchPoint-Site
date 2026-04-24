import pandas as pd
from datetime import datetime

from src.prediction.match_builder import (
    get_player_ranking,
    build_player_profile,
    get_tournament_metadata,
    build_match
)

from data.loader import load_model, load_features, load_rankings

#rnaking tests
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
            datetime(2018, 7, 1),   # 8 weeks old
            datetime(2018, 8, 1),   # 4 weeks old
            datetime(2018, 8, 20),  # 1 week old
            datetime(2018, 8, 27),  # tournament week
        ],
        "rank": [20, 12, 8, 6],
        "points": [2000, 3000, 4000, 5000]
    })

    tournament_date = datetime(2018, 8, 27)

    result = get_player_ranking(rankings, 1, tournament_date)
    #must pick most recent rank
    assert result["rank"] == 6
    assert result["rank_points"] == 5000


# player profile tests
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
        "p1_ioc" : ["SRB", "GBR"],
        "p2_ioc" : ["GBR", "SRB"],
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

#tournament metadata test
def test_get_tournament_metadata_whitebox():
    df = pd.DataFrame({
        "tourney_id":["123-US"],
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


def test_build_match_whitebox():
    df = pd.DataFrame({
        "tourney_id":["123-US"],
        "tourney_name": ["US Open"],
        "match_year": [2018],
        "tourney_date": [datetime(2018, 8, 27)],
        "surface_raw": ["Hard"],
        "best_of": [5],
        "tourney_level": ["G"],
        "draw_size": [128],
        "minutes": [120],
        "is_retirement": [0],
        "tiebreak_scores": [None],


        "p1_name": ["Novak Djokovic"],
        "p2_name": ["Daniil Medvedev"],
        "p1_id": [1],
        "p2_id": [2],
        "p1_age": [31],
        "p2_age": [22],
        "p1_ht": [188],
        "p2_ht": [198],
        "p1_hand": ["R"],
        "p2_hand": ["R"],
        "p1_ioc" : ["SRB"],
        "p2_ioc" : ["RUS"],
        "p1_elo_global": [1500],
        "p2_elo_global": [1450],
        "p1_elo_surface": [1500],
        "p2_elo_surface": [1450],

    })

    rankings = pd.DataFrame({
        "player_id": [1, 2],
        "ranking_date": [datetime(2018, 8, 20), datetime(2018, 8, 20)],
        "rank": [6, 5],
        "points": [5000, 5200]
    })

    model, scaler, model_features = load_model("models/xgboost_3.pkl")

    row_model, row_full = build_match(
        df=df,
        rankings_df=rankings,
        p1_name="Novak Djokovic",
        p2_name="Daniil Medvedev",
        tournament="US Open",
        year=2018,
        round_label="F",
        model_features=model_features,
        debug=False
    )

    assert set(row_model.columns) == set(model_features)
    assert row_model.iloc[0]["p1_rank"] == 6
    assert row_model.iloc[0]["p2_rank"] == 5
    assert row_model.iloc[0]["surface_Hard"] == 1
    assert row_model.iloc[0]["surface_Clay"] == 0
    assert row_model.iloc[0]["surface_Grass"] == 0


#strip history test
def test_strip_prematch_features_whitebox():
    df = pd.DataFrame({
        "rank_diff": [1],
        "p1_fatigue_pre": [0.2],
        "p1_form_all_5_winrate": [0.8],
        "p1_name": ["Novak Djokovic"],  # raw column that must remain
        "p1_age": [31]
    })

    from src.prediction.match_builder import strip_prematch_features
    cleaned = strip_prematch_features(df)

    assert "rank_diff" not in cleaned.columns
    assert "p1_fatigue_pre" not in cleaned.columns
    assert "p1_form_all_5_winrate" not in cleaned.columns

    # raw columns must remain
    assert "p1_name" in cleaned.columns
    assert "p1_age" in cleaned.columns


def test_build_match_history_clean_whitebox():
    df = load_features()
    rankings = load_rankings()

    _, row_full, history = build_match(
        df=df,
        rankings_df=rankings,
        p1_name="Novak Djokovic",
        p2_name="Daniil Medvedev",
        tournament="US Open",
        year=2018,
        round_label="F",
        model_features=[],
        return_history=True
    )

    # history rows are all except last
    history_rows = history.iloc[:-1]

    forbidden = [
        "rank_diff", "p1_fatigue_pre", "p1_form_all_5_winrate",
    ]

    for col in forbidden:
        assert col not in history_rows.columns

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
        model_features=model_features
    )

    assert set(row_model.columns) == set(model_features)
    assert row_model.isna().sum().sum() == 0

def test_synthetic_form_correctness_whitebox():
    """
    Ensures that form features for the synthetic row are computed correctly
    from the cleaned history returned by build_match.
    """

    df = load_features()
    rankings = load_rankings()
    model, scaler, model_features = load_model("models/xgboost_3.pkl")

    # Disable fatigue for simplicity
    from src.feature_engineering.feature_builder import FEATURE_CONFIG
    FEATURE_CONFIG["fatigue"] = False

    # Build match and capture history BEFORE feature building
    row_model, row_full, history = build_match(
        df=df,
        rankings_df=rankings,
        p1_name="Novak Djokovic",
        p2_name="Daniil Medvedev",
        tournament="US Open",
        year=2018,
        round_label="F",
        model_features=model_features,
        return_history=True
    )

    # Synthetic row is last
    syn = row_full.iloc[-1]

    # Check that form features exist
    form_cols = [c for c in syn.index if "form_" in c]
    assert len(form_cols) > 0

    # Check that form is NOT all zeros (unless truly no history)
    assert any(syn[c] != 0 for c in form_cols)