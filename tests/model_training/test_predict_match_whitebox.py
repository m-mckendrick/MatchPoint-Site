import pandas as pd
from datetime import datetime

from src.model_training.predict_match import (
    get_player_ranking,
    build_player_profile,
    get_tournament_metadata,
    build_match
)


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

import pandas as pd
from datetime import datetime
from src.model_training.predict_match import get_player_ranking


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


def test_get_tournament_metadata_whitebox():
    df = pd.DataFrame({
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
        "tourney_name": ["US Open"],
        "match_year": [2018],
        "tourney_date": [datetime(2018, 8, 27)],
        "surface_raw": ["Hard"],
        "best_of": [5],
        "tourney_level": ["G"],
        "draw_size": [128],

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
    })

    rankings = pd.DataFrame({
        "player_id": [1, 2],
        "ranking_date": [datetime(2018, 8, 20), datetime(2018, 8, 20)],
        "rank": [6, 5],
        "points": [5000, 5200]
    })

    model_features = [
        "p1_rank", "p1_rank_points", "p1_age", "p1_ht",
        "p2_rank", "p2_rank_points", "p2_age", "p2_ht",
        "surface_Hard", "surface_Clay", "surface_Grass",
        "best_of", "round"
    ]

    row = build_match(
        df=df,
        rankings_df=rankings,
        p1_name="Novak Djokovic",
        p2_name="Daniil Medvedev",
        tournament="US Open",
        year=2018,
        round_label="F",
        model_features=model_features
    )

    assert set(row.columns) == set(model_features)
    assert row.iloc[0]["p1_rank"] == 6
    assert row.iloc[0]["p2_rank"] == 5
    assert row.iloc[0]["surface_Hard"] == 1
    assert row.iloc[0]["surface_Clay"] == 0
    assert row.iloc[0]["surface_Grass"] == 0

