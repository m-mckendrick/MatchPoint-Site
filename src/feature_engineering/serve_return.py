import pandas as pd
import numpy as np

#serve features
#.replace protects from garbage values
def compute_first_serve_pct(df):
    return df["1stin"] / df["svpt"].replace(0, np.nan)

def compute_first_serve_won_pct(df):
    return df["1stwon"] / df["1stin"].replace(0, np.nan)

def compute_first_serve_in_play_won_pct(df):
    # first serves that were returned (basically rally points won on first serve returned)
    first_serve_in_play = df["1stin"] - df["ace"] - df["df"]
    return (df["1stwon"] - df["ace"]) / first_serve_in_play.replace(0, np.nan)


def compute_second_serve_won_pct(df):
    second_serve_pts = df["svpt"] - df["1stin"]
    return df["2ndwon"] / second_serve_pts.replace(0, np.nan)

def compute_second_serve_in_play_won_pct(df):
    second_serve_pts = df["svpt"] - df["1stin"]
    second_serve_in_play = second_serve_pts - df["df"]
    return df["2ndwon"] / second_serve_in_play.replace(0, np.nan)


def compute_serve_points_won_pct(df):
    return (df["1stwon"] + df["2ndwon"]) / df["svpt"]

def compute_double_fault_rate(df):
    return df["df"] / df["svpt"]

def compute_aces_per_game(df):
    return df["ace"] / df["svgms"].replace(0, np.nan)

def compute_aces_per_first_serve_in(df):
    return df["ace"] / df["1stin"].replace(0, np.nan)

def compute_first_serve_unreturned_pct(df):
    return df["ace"] / df["1stin"].replace(0, np.nan)


def compute_bp_save_pct(df):
    return df["bpsaved"] / df["bpfaced"].replace(0, np.nan)

#this is estimated so not 100% reliable
def compute_estimated_hold_pct(df):
    spw = (df["1stwon"] + df["2ndwon"]) / df["svpt"]
    return spw ** 4

def compute_serve_efficiency(df):
    spw = (df["1stwon"] + df["2ndwon"]) / df["svpt"]
    return spw - (df["df"] / df["svpt"])


#return features
def compute_return_points_won_pct(df, opp_df):
    opp_spw = (opp_df["1stwon"] + opp_df["2ndwon"]) / opp_df["svpt"]
    return 1 - opp_spw

def compute_bp_conversion_pct(df, opp_df):
    opp_bp_faced = opp_df["bpfaced"]
    opp_bp_saved = opp_df["bpsaved"]
    return (opp_bp_faced - opp_bp_saved) / opp_bp_faced.replace(0, np.nan)

#maybe change this
def compute_serve_dominance_index(spw_pct, rpw_pct):
    return spw_pct - rpw_pct

#like hold percentage estimated
def compute_estimated_break_pct(df, opp_df):
    opp_spw = (opp_df["1stwon"] + opp_df["2ndwon"]) / opp_df["svpt"]
    opp_hold = opp_spw ** 4
    return 1 - opp_hold

