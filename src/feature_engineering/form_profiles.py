import pandas as pd
from .form import add_form_features

"""
contains match selectors that can be passed into the add form feature function
**kwargs allows the engine to pass any extra arguments it wants
and the selectors ignore what they don’t need
"""

def select_career(past_matches, **kwargs):
    return past_matches


def select_h2h(past_matches, current_row, prefix, **kwargs):
    if prefix == "p1_":
        opponent_id = current_row["p2_id"]
    else:
        opponent_id = current_row["p1_id"]

    return past_matches[
        (past_matches["p1_id"] == opponent_id) |
        (past_matches["p2_id"] == opponent_id)
        ]


def select_last_52_weeks(past_matches, current_row, **kwargs):
    cutoff = current_row["tourney_date"] - pd.Timedelta(days=365)
    return past_matches[past_matches["tourney_date"] >= cutoff]


def select_tournament_career(past_matches, current_row, **kwargs):
    # excludes matches from this current tournament
    tid = current_row["tourney_id"]
    year = current_row["match_year"]

    return past_matches[
        (past_matches["tourney_id"] == tid) &
        (past_matches["match_year"] < year)
        ]


"""
these are all 
just calling add form feature using the appropriate match selector
"""

def add_career_form(df, windows=[5], surface_mode="all"):

    return add_form_features(
        df,
        windows=windows,
        surface_mode=surface_mode,
        match_selector=select_career
    )


def add_h2h_form(df, windows=[5], surface_mode="all"):
    """
    Add H2H form features using only matches vs the current opponent.
    """
    return add_form_features(
        df,
        windows=windows,
        surface_mode=surface_mode,
        match_selector=select_h2h
    )


def add_52week_form(df, windows=[5], surface_mode="all"):
    """
    Add last-52-weeks form features.
    """
    return add_form_features(
        df,
        windows=windows,
        surface_mode=surface_mode,
        match_selector=select_last_52_weeks
    )


def add_tournament_form(df, windows=[5], surface_mode="all"):
    """
    Add tournament-career form features.
    """
    return add_form_features(
        df,
        windows=windows,
        surface_mode=surface_mode,
        match_selector=select_tournament_career
    )
