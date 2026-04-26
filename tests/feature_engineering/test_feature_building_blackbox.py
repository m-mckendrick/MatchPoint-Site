import pandas as pd
import pytest

from src.feature_engineering.feature_builder import build_features, FEATURE_CONFIG
from data.loader import load_matches
from tests.conftest import small_df


# returns df
def test_build_features_returns_dataframe(small_df):
    df_out = build_features(small_df)
    assert isinstance(df_out, pd.DataFrame)


#no rows removed or added
def test_row_count_preserved(small_df):
    df_out = build_features(small_df)
    assert df_out.shape[0] == small_df.shape[0]


# none form columns must excist
EXPECTED_NON_FORM_COLS = [
    # original match metadata
    "tourney_id", "tourney_name", "draw_size", "tourney_level",
    "tourney_date", "match_num", "score", "best_of", "round",
    "minutes", "is_retirement", "tiebreak_scores", "perspective", "label",

    # player metadata
    "p1_id", "p1_rank", "p1_age", "p1_ht", "p1_hand", "p1_ioc",
    "p1_rank_points", "p1_ace", "p1_df", "p1_svpt", "p1_1stin",
    "p1_1stwon", "p1_2ndwon", "p1_svgms", "p1_bpsaved", "p1_bpfaced", "p1_name",

    "p2_id", "p2_rank", "p2_age", "p2_ht", "p2_hand", "p2_ioc",
    "p2_rank_points", "p2_ace", "p2_df", "p2_svpt", "p2_1stin",
    "p2_1stwon", "p2_2ndwon", "p2_svgms", "p2_bpsaved", "p2_bpfaced", "p2_name",

    # match context features
    "rank_diff", "rank_points_diff", "is_grand_slam", "is_masters",
    "is_best_of_five", "p1_lefty", "p2_lefty", "lefty_diff",
    "surface_raw", "surface_Carpet", "surface_Clay", "surface_Grass",
    "surface_Hard", "round_encoded", "match_year", "age_diff",
    "height_diff", "tourney_prestige", "round_points", "tourney_country",
    "p1_home", "p2_home", "home_advantage_diff", "is_indoor",

    # serve/return features
    "p1_first_serve_pct", "p2_first_serve_pct",
    "p1_first_serve_won_pct", "p2_first_serve_won_pct",
    "p1_first_serve_in_play_won_pct", "p2_first_serve_in_play_won_pct",
    "p1_second_serve_won_pct", "p2_second_serve_won_pct",
    "p1_second_serve_in_play_won_pct", "p2_second_serve_in_play_won_pct",
    "p1_serve_points_won_pct", "p2_serve_points_won_pct",
    "p1_double_fault_rate", "p2_double_fault_rate",
    "p1_aces_per_game", "p2_aces_per_game",
    "p1_aces_per_first_serve_in", "p2_aces_per_first_serve_in",
    "p1_first_serve_unreturned_pct", "p2_first_serve_unreturned_pct",
    "p1_bp_save_pct", "p2_bp_save_pct",
    "p1_estimated_hold_pct", "p2_estimated_hold_pct",
    "p1_serve_efficiency", "p2_serve_efficiency",
    "p1_return_points_won_pct", "p2_return_points_won_pct",
    "p1_bp_conversion_pct", "p2_bp_conversion_pct",
    "p1_sdi", "p2_sdi",
    "p1_estimated_break_pct", "p2_estimated_break_pct",

    # match result features
    "p1_sets_won", "p2_sets_won",
    "p1_games_won", "p2_games_won",
    "p1_tiebreaks_won", "p2_tiebreaks_won",
    "p1_straight_sets_win", "p2_straight_sets_win",
    "p1_sets_lost", "p2_sets_lost",
    "p1_games_lost", "p2_games_lost",

    # fatigue
    "p1_fatigue_in", "p2_fatigue_in","p1_fatigue_pre", "p2_fatigue_pre",

    # pressure performance
    "p1_tiebreak_win_pct", "p2_tiebreak_win_pct",
    "p1_deciding_set_win", "p2_deciding_set_win",
    "p1_close_sets_won", "p2_close_sets_won",
    "p1_pressure_index", "p2_pressure_index",
    "match_importance", "stage_pressure",
]


def test_expected_non_form_columns_present(small_df):
    df_out = build_features(small_df)
    missing = [c for c in EXPECTED_NON_FORM_COLS if c not in df_out.columns]
    assert not missing, f"Missing expected columns: {missing}"


#test no na in non- form columns
def test_no_nas_in_non_form_features(small_df):
    df_out = build_features(small_df)

    non_form_cols = [
        c for c in df_out.columns
        if "form" not in c.lower()
           and "h2h" not in c.lower()
           and "52week" not in c.lower()
           and "career" not in c.lower()
    ]

    assert not df_out[non_form_cols].isna().any().any(), \
        "NaNs found in non-form features"


#test that feature toggle disable the feature from being computed
def test_feature_toggle_disables_feature(small_df):

    FEATURE_CONFIG["fatigue"] = False

    df_out = build_features(small_df)

    assert "p1_fatigue_in" not in df_out.columns
    assert "p2_fatigue_in" not in df_out.columns
    assert "p1_fatigue_pre" not in df_out.columns
    assert "p2_fatigue_pre" not in df_out.columns

    # reset toggle
    FEATURE_CONFIG["fatigue"] = True


# must not mutate input df
def test_input_dataframe_not_modified(small_df):
    df_copy = small_df.copy(deep=True)
    _ = build_features(small_df)

    # compare shapes and columns
    assert list(small_df.columns) == list(df_copy.columns)
    assert small_df.shape == df_copy.shape


# builder runs with all feature disabled
def test_build_features_runs_without_form(small_df):
    # Disable all form-related features
    FEATURE_CONFIG["form"] = False
    FEATURE_CONFIG["surface_form"] = False
    FEATURE_CONFIG["h2h"] = False
    FEATURE_CONFIG["career"] = False
    FEATURE_CONFIG["recency"] = False
    FEATURE_CONFIG["tournament_form"] = False

    df_out = build_features(small_df)

    assert isinstance(df_out, pd.DataFrame)
    assert df_out.shape[0] == small_df.shape[0]

    # reset toggles
    FEATURE_CONFIG["form"] = True
    FEATURE_CONFIG["surface_form"] = True
    FEATURE_CONFIG["h2h"] = True
    FEATURE_CONFIG["career"] = True
    FEATURE_CONFIG["recency"] = True
    FEATURE_CONFIG["tournament_form"] = True
