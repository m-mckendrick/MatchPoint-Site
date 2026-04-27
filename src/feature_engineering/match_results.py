#basically score parser

import pandas as pd
import numpy as np
import re

from src.data_preprocessing.clean_pipeline import extract_tiebreak_from_set, extract_tiebreak_score_string

"""
get score and tiebreak score and transform into userful stats for historical context features

"""

#get number of game won in each set
def parse_games_from_set(set_str):

    try:
        g1, g2 = map(int, set_str.split("-"))
        return g1, g2
    except:
        return None, None

#p1 win is 1 when player 1 wins and 0 when p 2 wins
def compute_match_result(score_str, tiebreak_str, p1_win):
    """
    Given:
        score_str      -> '7-6 6-7 6-2'
        tiebreak_str   -> '7-4 10-8 none'
        p1_win         -> 1 or 0

    Returns:
        p1_sets, p2_sets,
        p1_games, p2_games,
        p1_tb, p2_tb,
        p1_ss, p2_ss
    """

    if pd.isna(score_str):
        return (0, 0, 0, 0, 0, 0, 0, 0)

    sets = score_str.split()
    tiebreaks = tiebreak_str.split() if isinstance(tiebreak_str, str) else []

    # fill with none id needed
    while len(tiebreaks) < len(sets):
        tiebreaks.append("none")

    p1_sets = p2_sets = 0
    p1_games = p2_games = 0
    p1_tb = p2_tb = 0

    for set_score, tb_score in zip(sets, tiebreaks):

        # remove tiebreak part from set score
        base_set = set_score.split("(")[0]

    # look through sets add number of games together
        g1, g2 = parse_games_from_set(base_set)
        if g1 is None:
            continue

        # accumulate games
        if g1 and g2 >= 6:
            p1_games += 6
            p2_games += 6
        else:
            p1_games += g1
            p2_games += g2

        # set winner
        if g1 > g2:
            p1_sets += 1
        else:
            p2_sets += 1

        # tiebreak winner
        if tb_score != "none":
            tb_winner, tb_loser = map(int, tb_score.split("-"))
            if g1 > g2:
                p1_tb += 1
            else:
                p2_tb += 1

    # straight sets
    p1_ss = int(p1_win == 1 and p2_sets == 0)
    p2_ss = int(p1_win == 0 and p1_sets == 0)

    return (
        p1_sets, p2_sets,
        p1_games, p2_games,
        p1_tb, p2_tb,
        p1_ss, p2_ss
    )


def add_match_result_features(df):

    results = df.apply(
        lambda row: compute_match_result(
            row["score"],
            row["tiebreak_scores"],
            row["label"]
        ),
        axis=1
    )

    results_df = pd.DataFrame(results.tolist(), columns=[
        "p1_sets_won", "p2_sets_won",
        "p1_games_won", "p2_games_won",
        "p1_tiebreaks_won", "p2_tiebreaks_won",
        "p1_straight_sets_win", "p2_straight_sets_win"
    ])

    df = pd.concat([df, results_df], axis=1)

    # derived fields
    df["p1_sets_lost"] = df["p2_sets_won"]
    df["p2_sets_lost"] = df["p1_sets_won"]

    df["p1_games_lost"] = df["p2_games_won"]
    df["p2_games_lost"] = df["p1_games_won"]

    return df

