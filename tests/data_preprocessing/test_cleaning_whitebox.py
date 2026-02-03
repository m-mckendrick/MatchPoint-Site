import pandas as pd
import numpy as np
from src.data_preprocessing.clean_pipeline import (
    validate_physical_stats,
    validate_surface,
    validate_round,
    validate_best_of,
    validate_minutes
)

#white-box tests
#these test every edge case and branch (every line of code)

def test_validate_physical_stats_filters_invalid_values():
    df = pd.DataFrame([{
        "winner_ht": 300,
        "loser_ht": 150, #only valid entry
        "winner_age": 10,
        "loser_age": 50
    }])
    out = validate_physical_stats(df)
    assert len(out) == 0  # all rows removed

#if surface isn't in know list
#capsys captures printed output of the data health test
def test_validate_surface_warns_on_bad_values(capsys):
    df = pd.DataFrame({"surface": ["Hard", "Moon"]})
    validate_surface(df)
    captured = capsys.readouterr()
    assert "unexpected surface" in captured.out.lower()


def test_validate_round_warns_on_bad_values(capsys):
    df = pd.DataFrame({"round": ["QF", "XYZ"]})
    validate_round(df)
    captured = capsys.readouterr()
    assert "unexpected round" in captured.out.lower()


def test_validate_best_of_warns_on_bad_values(capsys):
    df = pd.DataFrame({"best_of": [3, 7]})
    validate_best_of(df)
    captured = capsys.readouterr()
    assert "unexpected best_of" in captured.out.lower()


def test_validate_minutes_short_and_long(capsys):
    df = pd.DataFrame({"minutes": [10, 500]})
    validate_minutes(df)
    captured = capsys.readouterr()
    assert "short" in captured.out.lower()
    assert "long" in captured.out.lower()
