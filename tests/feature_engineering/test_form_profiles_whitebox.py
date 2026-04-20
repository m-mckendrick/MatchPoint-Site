import pandas as pd

from src.feature_engineering.form_profiles import (
    select_career,
    select_h2h,
    select_last_52_weeks,
    select_tournament_career,
)

from tests.feature_engineering.test_form_profiles_blackbox import make_df


def test_selector_career_whitebox():
    df = make_df()
    out = select_career(df)
    assert len(out) == len(df)


def test_selector_h2h_whitebox():
    df = make_df()
    current_row = df.iloc[0]  # p1=1, p2=2
    out = select_h2h(df, current_row, prefix="p1_")

    # should include only matches involving player 2
    assert all((out["p1_id"] == 2) | (out["p2_id"] == 2))


def test_selector_52weeks_whitebox():
    df = make_df()
    current_row = df[df["match_year"] == 2024].iloc[0]
    out = select_last_52_weeks(df, current_row)

    # should exclude 2023 matches
    assert all(out["tourney_date"] >= current_row["tourney_date"] - pd.Timedelta(days=365))



def test_selector_tournament_career_whitebox():
    df = make_df()
    current_row = df[df["match_year"] == 2024].iloc[0]
    out = select_tournament_career(df, current_row)

    # should include only 2023 matches from same tournament
    assert all(out["match_year"] < 2024)
    assert all(out["tourney_id"] == current_row["tourney_id"])
