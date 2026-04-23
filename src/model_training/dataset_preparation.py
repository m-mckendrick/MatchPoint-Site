import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib

#get rid of columns that aren't predictive
#intentionally excludes match level data
def prepare_dataset(df: pd.DataFrame):

    y = df["label"]

    NON_PREDICTIVE = [
    "label",
    "tourney_id",
    "tourney_name",
    "tourney_date",
    "tourney_country",
    "match_num",
    "match_year",
    "draw_size",
    "minutes",
    "tiebreak_scores",
    "score",
    "p1_name",
    "p2_name",
    "p1_id",
    "p2_id",
    "perspective",
    "round",            # encoded version is used instead
    "surface_raw",      # encoded versions used instead
    "tourney_signature", # used only for grouping, not prediction

    "minutes",
    "is_retirement",

    ]

    POST_MATCH = [
        # base df stats
        "p1_ace","p1_df","p1_svpt","p1_1stin","p1_1stwon","p1_2ndwon",
        "p1_svgms","p1_bpsaved","p1_bpfaced",
        "p2_ace","p2_df","p2_svpt","p2_1stin","p2_1stwon","p2_2ndwon",
        "p2_svgms","p2_bpsaved","p2_bpfaced",

        # match result features
        "p1_sets_won","p2_sets_won",
        "p1_sets_lost","p2_sets_lost",
        "p1_games_won","p2_games_won",
        "p1_games_lost","p2_games_lost",
        "p1_tiebreaks_won","p2_tiebreaks_won",
        "p1_straight_sets_win","p2_straight_sets_win",
        "p1_deciding_set_win","p2_deciding_set_win",
        "p1_close_sets_won","p2_close_sets_won",
        "p1_tiebreak_win_pct","p2_tiebreak_win_pct",


        # performance metrics
        "p1_bp_save_pct","p2_bp_save_pct",
        "p1_bp_conv_pct","p2_bp_conv_pct",
        "p1_first_serve_pct","p2_first_serve_pct",
        "p1_first_serve_won_pct","p2_first_serve_won_pct",
        "p1_second_serve_won_pct","p2_second_serve_won_pct",
        "p1_return_points_won_pct","p2_return_points_won_pct",
        "p1_serve_points_won_pct","p2_serve_points_won_pct",
        "p1_double_fault_rate","p2_double_fault_rate",
        "p1_aces_per_game","p2_aces_per_game",
        "p1_aces_per_first_serve_in","p2_aces_per_first_serve_in",
        "p1_first_serve_unreturned_pct","p2_first_serve_unreturned_pct",
        "p1_estimated_hold_pct","p2_estimated_hold_pct",
        "p1_estimated_break_pct","p2_estimated_break_pct",
        "p1_first_serve_in_play_won_pct","p2_first_serve_in_play_won_pct",
        "p1_second_serve_in_play_won_pct","p2_second_serve_in_play_won_pct",
        "p1_bp_conversion_pct","p2_bp_conversion_pct",



        "p1_serve_efficiency","p2_serve_efficiency",
        "p1_sdi","p2_sdi",
        "p1_pressure_index","p2_pressure_index",
    ]

    OUTCOME_TERMS = [
        "sets_pct",
        "games_pct",
        "deciding_set_win",
        "straight_sets_win",
        "close_sets_won",
        "tiebreak_win_pct",
    ]
    
    RAW_PRESSURE = [
        "stage_pressure",
        "match_importance",
    ]
    
    cols_to_drop = set(NON_PREDICTIVE + POST_MATCH + RAW_PRESSURE)

    for col in df.columns:
        # drop outcome‑based rolling stats
        if any(term in col for term in OUTCOME_TERMS):
            cols_to_drop.add(col)

        # drop H2H except winrate
        if "h2h" in col and "winrate" not in col:
            cols_to_drop.add(col)

        # drop tournament_3 except winrate
        if "tournament_3" in col and "winrate" not in col:
            cols_to_drop.add(col)


    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # only numeric columns
    numeric_df = df.select_dtypes(include=["int64", "float64", "bool"]).copy()

    # convert bool to int
    for col in numeric_df.select_dtypes(include=["bool"]).columns:
        numeric_df[col] = numeric_df[col].astype(int)

    numeric_df = numeric_df.fillna(0)

    # x == numeric df
    print("columns used:",numeric_df.columns.tolist())
    return numeric_df, y
