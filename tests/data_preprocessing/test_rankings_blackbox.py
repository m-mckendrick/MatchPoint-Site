import pandas as pd
from pathlib import Path
from src.data_preprocessing.rankings import combine_rankings
from data.loader import RAW_MATCHES_PATH, PROCESSED_PATH


"""
test for the combine rankings function
uses monkeypatch so the function thinks it’s reading from data/raw, 
but actually reads from the temporary test directory.
"""

def test_combine_rankings_basic(tmp_path, monkeypatch):
    f00 = tmp_path / "atp_rankings_00s.csv"
    f10 = tmp_path / "atp_rankings_10s.csv"
    f20 = tmp_path / "atp_rankings_20s.csv"

    df00 = pd.DataFrame({
        "player": [1, 1],
        "ranking_date": ["20000101", "20010101"],
        "rank": [100, 90]
    })
    df10 = pd.DataFrame({
        "player": [1],
        "ranking_date": ["20100101"],
        "rank": [20]
    })
    df20 = pd.DataFrame({
        "player": [1],
        "ranking_date": ["20200101"],
        "rank": [5]
    })

    df00.to_csv(f00, index=False)
    df10.to_csv(f10, index=False)
    df20.to_csv(f20, index=False)

    #makes raw matches path point to the test directory
    monkeypatch.setattr("src.data_preprocessing.rankings.RAW_MATCHES_PATH", tmp_path)
    monkeypatch.setattr("src.data_preprocessing.rankings.PROCESSED_PATH", tmp_path)

    out = combine_rankings()

    assert len(out) == 4

    assert "player_id" in out.columns
    assert "player" not in out.columns

    assert pd.api.types.is_datetime64_any_dtype(out["ranking_date"])

    assert out["ranking_date"].is_monotonic_increasing

    assert (tmp_path / "atp_rankings_2000-2020.csv").exists()
