import pandas as pd
import numpy as np
from collections import defaultdict

"""
base config:
BASE_ELO: starting rating for all players
K_FACTOR: controls how fast Elo changes e.g higher = more reactive
SURFACE_K_FACTOR: slightly lower because surface-specific elo should be more stable
"""
BASE_ELO = 1500.0
K_FACTOR = 32.0
SURFACE_K_FACTOR = 24.0
H2H_K_FACTOR = 16.0


def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _update_elo(rating: float, score: float, expected: float, k: float) -> float:
    return rating + k * (score - expected)


# core computation
def add_elo_features(
        df: pd.DataFrame,
        player1_col: str = "p1_id",
        player2_col: str = "p2_id",
        winner_label_col: str = "label",
        surface_col: str = "surface_raw",
        date_col: str = "tourney_date",
        round_col="round_encoded",
        compute_global: bool = True,
        compute_surface: bool = True,
        compute_h2h: bool = False,
) -> pd.DataFrame:
    """
    Elo ratings estimate player strength by comparing expected performance
    to actual match outcomes; when a player performs better than expected,
    their rating increases, and when they underperform,
    it decreases, with larger upsets producing larger rating adjustments
    this function calculate elo ratings surface elo rating and h2h elo ratings

    this function store pre match elo as a feature
    so it can be used for prediction later in the project
    """

    df = df.copy()

    df = df.sort_values([date_col, round_col]).reset_index(drop=True)


    # Elo stores
    global_elo = defaultdict(lambda: BASE_ELO)
    surface_elo = defaultdict(lambda: BASE_ELO)
    h2h_elo = defaultdict(lambda: BASE_ELO)

    # output columns
    p1_elo_global_list = []
    p2_elo_global_list = []
    elo_global_diff_list = []

    p1_elo_surface_list = []
    p2_elo_surface_list = []
    elo_surface_diff_list = []

    p1_elo_h2h_list = []
    p2_elo_h2h_list = []
    elo_h2h_diff_list = []

    # iterate matches
    for _, row in df.iterrows():
        p1 = row[player1_col]
        p2 = row[player2_col]
        surface = row[surface_col] if surface_col in df.columns else "Unknown"
        label = row[winner_label_col]  # 1 if p1 wins, 0 if p2 wins

        # global
        if compute_global:
            r1_g = global_elo[p1]
            r2_g = global_elo[p2]

            exp1_g = _expected_score(r1_g, r2_g)
            exp2_g = 1.0 - exp1_g

            # store pre-match
            p1_elo_global_list.append(r1_g)
            p2_elo_global_list.append(r2_g)
            elo_global_diff_list.append(r1_g - r2_g)

            # actual scores
            s1 = 1.0 if label == 1 else 0.0
            s2 = 1.0 - s1

            # update
            global_elo[p1] = _update_elo(r1_g, s1, exp1_g, K_FACTOR)
            global_elo[p2] = _update_elo(r2_g, s2, exp2_g, K_FACTOR)
        else:
            p1_elo_global_list.append(np.nan)
            p2_elo_global_list.append(np.nan)
            elo_global_diff_list.append(np.nan)

        # surface
        if compute_surface:
            key1 = (p1, surface)
            key2 = (p2, surface)

            r1_s = surface_elo[key1]
            r2_s = surface_elo[key2]

            exp1_s = _expected_score(r1_s, r2_s)
            exp2_s = 1.0 - exp1_s

            # store pre-match
            p1_elo_surface_list.append(r1_s)
            p2_elo_surface_list.append(r2_s)
            elo_surface_diff_list.append(r1_s - r2_s)

            # actual scores
            s1 = 1.0 if label == 1 else 0.0
            s2 = 1.0 - s1

            # update
            surface_elo[key1] = _update_elo(r1_s, s1, exp1_s, SURFACE_K_FACTOR)
            surface_elo[key2] = _update_elo(r2_s, s2, exp2_s, SURFACE_K_FACTOR)
        else:
            p1_elo_surface_list.append(np.nan)
            p2_elo_surface_list.append(np.nan)
            elo_surface_diff_list.append(np.nan)

        # h2h
        if compute_h2h:
            # order pair so (p1, p2) and (p2, p1) share the same key
            pair_key = tuple(sorted([p1, p2]))
            r1_h = h2h_elo[(pair_key, p1)]
            r2_h = h2h_elo[(pair_key, p2)]

            exp1_h = _expected_score(r1_h, r2_h)
            exp2_h = 1.0 - exp1_h

            # store pre-match
            p1_elo_h2h_list.append(r1_h)
            p2_elo_h2h_list.append(r2_h)
            elo_h2h_diff_list.append(r1_h - r2_h)

            s1 = 1.0 if label == 1 else 0.0
            s2 = 1.0 - s1

            h2h_elo[(pair_key, p1)] = _update_elo(r1_h, s1, exp1_h, H2H_K_FACTOR)
            h2h_elo[(pair_key, p2)] = _update_elo(r2_h, s2, exp2_h, H2H_K_FACTOR)
        else:
            p1_elo_h2h_list.append(np.nan)
            p2_elo_h2h_list.append(np.nan)
            elo_h2h_diff_list.append(np.nan)

    # attach columns
    if compute_global:
        df["p1_elo_global"] = p1_elo_global_list
        df["p2_elo_global"] = p2_elo_global_list
        df["elo_global_diff"] = elo_global_diff_list

    if compute_surface:
        df["p1_elo_surface"] = p1_elo_surface_list
        df["p2_elo_surface"] = p2_elo_surface_list
        df["elo_surface_diff"] = elo_surface_diff_list

    if compute_h2h:
        df["p1_elo_h2h"] = p1_elo_h2h_list
        df["p2_elo_h2h"] = p2_elo_h2h_list
        df["elo_h2h_diff"] = elo_h2h_diff_list

    return df
