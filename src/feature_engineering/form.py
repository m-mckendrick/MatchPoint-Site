import numpy as np
import pandas as pd

# main function
def add_form_features(
        df,
        windows=[5],
        surface_mode="all",
        match_selector=None,
        prefix_tag = None
):
    """
    Adds form features for p1 and p2 using rolling windows e.g 5 games, 10 games
    Form is computed from past matches only and defaults to 0 when no history exists.

    windows is the number of rolling windows e.g ([3],[5]) =average of 3 and 5 games
    surface_mode is "all"  use all past matches / or "same" only the same as the match.

    match_selector: optional function that filters past matches (career, h2h, 52w, etc.)
    """

    df = df.sort_values(["tourney_date", "match_num"]).copy()

    # compute form for each player
    df = _compute_player_form(df, "p1_id", "p1_", windows, surface_mode, match_selector, prefix_tag)
    df = _compute_player_form(df, "p2_id", "p2_", windows, surface_mode, match_selector, prefix_tag)

    return df



"""
helper functions
"""

def _form_keys():
    """List of all form metric suffixes."""
    return [
        # match results
        "winrate", "sets_pct", "games_pct", "tiebreak_win_pct",
        "deciding_set_win", "straight_sets_win",

        # serve
        "first_serve_pct", "first_serve_won_pct", "second_serve_won_pct",
        "double_fault_rate", "aces_per_game", "serve_pts_won_pct",
        "estimated_hold_pct", "serve_efficiency",

        # return
        "return_pts_won_pct", "estimated_break_pct",
        
        # pressure form
        "pressure_index", "close_sets_won",
        "bp_save_pct", "bp_conversion_pct"

    ]

def _extract_stats(matches):
    """extract all stats needed for form calculations."""
    return {
        "win": (matches["label"] == 1).astype(int),
        "sets_won": matches["p1_sets_won"],
        "sets_lost": matches["p1_sets_lost"],
        "games_won": matches["p1_games_won"],
        "games_lost": matches["p1_games_lost"],
        "tiebreaks_won": matches["p1_tiebreaks_won"],
        "tiebreaks_lost": matches["p2_tiebreaks_won"],
        "deciding_set_win": matches["p1_deciding_set_win"],
        "straight_sets_win": matches["p1_straight_sets_win"],

        "pressure_index": matches["p1_pressure_index"],
        "close_sets_won": matches["p1_close_sets_won"],
        "bp_save_pct": matches["p1_bp_save_pct"],
        "bp_conversion_pct": matches["p1_bp_conversion_pct"],


        # serve
        "first_serve_pct": matches["p1_first_serve_pct"],
        "first_serve_won_pct": matches["p1_first_serve_won_pct"],
        "second_serve_won_pct": matches["p1_second_serve_won_pct"],
        "double_fault_rate": matches["p1_double_fault_rate"],
        "aces_per_game": matches["p1_aces_per_game"],
        "serve_pts_won_pct": matches["p1_serve_points_won_pct"],
        "estimated_hold_pct": matches["p1_estimated_hold_pct"],
        "serve_efficiency": matches["p1_serve_efficiency"],

        # return
        "return_pts_won_pct": matches["p1_return_points_won_pct"],
        "estimated_break_pct": matches["p1_estimated_break_pct"],
    }

"""
internal functions
"""

def _filter_past_matches(
        player_matches,
        match_pos,
        surface_mode,
        match_selector,
        prefix
):

    # past matches = before current
    past_matches = player_matches.iloc[:match_pos]

    if len(past_matches) == 0:
        return past_matches

    current_row = player_matches.iloc[match_pos]
    current_surface = current_row["surface_raw"]

    # surface filter
    if surface_mode == "same":
        past_matches = past_matches[past_matches["surface_raw"] == current_surface]

    # match selector
    if match_selector is not None:
        past_matches = match_selector(
            past_matches=past_matches,
            player_matches=player_matches,
            current_row=current_row,
            prefix=prefix
        )

    return past_matches



def _compute_player_form(df, player_id_col, prefix, windows, surface_mode, match_selector, prefix_tag):

    form_records = []

    for player_id, player_matches in df.groupby(player_id_col):

        # remove mirrored rows that belong to the opponent
        player_matches = player_matches[player_matches[player_id_col] == player_id]

        player_matches = player_matches.sort_values(["tourney_date", "match_num"]).copy()

        # extract stats for speed
        stats = _extract_stats(player_matches)

        # compute form for each match
        for match_pos in range(len(player_matches)):

            df_index = player_matches.index[match_pos]

            # filter past matches
            past_matches = _filter_past_matches(
                player_matches=player_matches,
                match_pos=match_pos,
                surface_mode=surface_mode,
                match_selector=match_selector,
                prefix=prefix
            )

            # compute rolling windows
            form_values = {}
            for window_size in windows:
                form_values.update(
                    _rolling_form_block(past_matches, stats, window_size, prefix, prefix_tag)
                )

            form_records.append((df_index, form_values))

    # merge back
    """
    instead of writing each form feature into the DataFrame one cell at a time
    which caused fragmentation and slow performance
    the updated version collects all computed form values into a temporary table
    and merges them back into the main DataFrame in a single concat operation
    making the process dramatically faster without changing any outputs.”
    """

    if form_records:
        form_df = pd.DataFrame(
            {idx: vals for idx, vals in form_records}).T
        form_df.index = form_df.index.astype(df.index.dtype)

        df = pd.concat([df, form_df], axis=1)

    # fill missing
    form_cols = [c for c in df.columns if c.startswith(prefix + "form_")]
    df[form_cols] = df[form_cols].fillna(0)

    return df



def _rolling_form_block(past_matches, stats, window_size, prefix, prefix_tag):
    """
    Compute rolling form for a given window size.
    """

    tag = f"{prefix_tag}_" if prefix_tag else ""

    # if no history return zeros for all form metrics
    if len(past_matches) == 0:
        return {f"{prefix}form_{tag}{window_size}_{metric}": 0 for metric in _form_keys()}

    # select last N matches
    recent_matches = past_matches.iloc[-window_size:]
    recent_indices = recent_matches.index.tolist()

    out = {}

    # results form
    out[f"{prefix}form_{tag}{window_size}_winrate"] = stats["win"].loc[recent_indices].mean()

    out[f"{prefix}form_{tag}{window_size}_sets_pct"] = (
            stats["sets_won"].loc[recent_indices] /
            (stats["sets_won"].loc[recent_indices] + stats["sets_lost"].loc[recent_indices] + 1e-6)
    ).mean()

    out[f"{prefix}form_{tag}{window_size}_games_pct"] = (
            stats["games_won"].loc[recent_indices] /
            (stats["games_won"].loc[recent_indices] + stats["games_lost"].loc[recent_indices] + 1e-6)
    ).mean()

    out[f"{prefix}form_{tag}{window_size}_tiebreak_win_pct"] = (
            stats["tiebreaks_won"].loc[recent_indices] /
            (stats["tiebreaks_won"].loc[recent_indices] + stats["tiebreaks_lost"].loc[recent_indices] + 1e-6)
    ).mean()

    out[f"{prefix}form_{tag}{window_size}_deciding_set_win"] = stats["deciding_set_win"].loc[recent_indices].mean()
    out[f"{prefix}form_{tag}{window_size}_straight_sets_win"] = stats["straight_sets_win"].loc[recent_indices].mean()

    # serve form
    serve_metrics = [
        "first_serve_pct", "first_serve_won_pct", "second_serve_won_pct",
        "double_fault_rate", "aces_per_game", "serve_pts_won_pct",
        "estimated_hold_pct", "serve_efficiency"
    ]

    for metric in serve_metrics:
        out[f"{prefix}form_{tag}{window_size}_{metric}"] = stats[metric].loc[recent_indices].mean()

    # return form
    return_metrics = [
        "return_pts_won_pct", "bp_conversion_pct", "estimated_break_pct"
    ]

    for metric in return_metrics:
        out[f"{prefix}form_{tag}{window_size}_{metric}"] = stats[metric].loc[recent_indices].mean()

    # pressure form
    out[f"{prefix}form_{tag}{window_size}_pressure_index"] = stats["pressure_index"].loc[recent_indices].mean()
    out[f"{prefix}form_{tag}{window_size}_close_sets_won"] = stats["close_sets_won"].loc[recent_indices].mean()
    out[f"{prefix}form_{tag}{window_size}_bp_save_pct"] = stats["bp_save_pct"].loc[recent_indices].mean()
    out[f"{prefix}form_{tag}{window_size}_bp_conversion_pct"] = stats["bp_conversion_pct"].loc[recent_indices].mean()

    return out




