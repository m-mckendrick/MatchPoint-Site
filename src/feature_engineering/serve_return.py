import pandas as pd
import numpy as np

#serve features
#.replace protects from garbage values
def compute_first_serve_pct(df):
    return df["1stin"] / df["svpt"].replace(0, np.nan)

def compute_first_serve_won_pct(df):
    return df["1stwon"] / df["1stin"].replace(0, np.nan)

def compute_second_serve_won_pct(df):
    second_serve_pts = df["svpt"] - df["1stin"]
    return df["2ndwon"] / second_serve_pts.replace(0, np.nan)

def compute_serve_points_won_pct(df):
    return (df["1stwon"] + df["2ndwon"]) / df["svpt"]

def compute_double_fault_rate(df):
    return df["df"] / df["svpt"]

def compute_aces_per_game(df):
    return df["ace"] / df["svgms"].replace(0, np.nan)

def compute_bp_save_pct(df):
    return df["bpsaved"] / df["bpfaced"].replace(0, np.nan)

#return features
def compute_return_points_won_pct(df, opp_df):
    opp_spw = (opp_df["1stwon"] + opp_df["2ndwon"]) / opp_df["svpt"]
    return 1 - opp_spw

def compute_bp_conversion_pct(df, opp_df):
    opp_bp_faced = opp_df["bpfaced"]
    opp_bp_saved = opp_df["bpsaved"]
    return (opp_bp_faced - opp_bp_saved) / opp_bp_faced.replace(0, np.nan)

def compute_serve_dominance_index(spw_pct, rpw_pct):
    return spw_pct - rpw_pct

def add_serve_return_features(df):

    # Winner stats
    #these line both get the w or l stats then renames them without the w or l in front
    #therefore the can both be plugged into the functions above
    w = df.filter(regex="^w_").rename(columns=lambda c: c[2:])
    l = df.filter(regex="^l_").rename(columns=lambda c: c[2:])

    df["w_first_serve_pct"] = compute_first_serve_pct(w)
    df["w_first_serve_won_pct"] = compute_first_serve_won_pct(w)
    df["w_second_serve_won_pct"] = compute_second_serve_won_pct(w)
    df["w_df_rate"] = compute_double_fault_rate(w)
    df["w_aces_per_game"] = compute_aces_per_game(w)
    df["w_bp_save_pct"] = compute_bp_save_pct(w)
    df["w_spw_pct"] = compute_serve_points_won_pct(w)
    df["w_rpw_pct"] = compute_return_points_won_pct(w, l)
    df["w_bp_conv_pct"] = compute_bp_conversion_pct(w, l)
    df["w_sdi"] = compute_serve_dominance_index(df["w_spw_pct"], df["w_rpw_pct"])

    # Loser stats
    df["l_first_serve_pct"] = compute_first_serve_pct(l)
    df["l_first_serve_won_pct"] = compute_first_serve_won_pct(l)
    df["l_second_serve_won_pct"] = compute_second_serve_won_pct(l)
    df["l_df_rate"] = compute_double_fault_rate(l)
    df["l_aces_per_game"] = compute_aces_per_game(l)
    df["l_bp_save_pct"] = compute_bp_save_pct(l)
    df["l_spw_pct"] = compute_serve_points_won_pct(l)
    df["l_rpw_pct"] = compute_return_points_won_pct(l, w)
    df["l_bp_conv_pct"] = compute_bp_conversion_pct(l, w)
    df["l_sdi"] = compute_serve_dominance_index(df["l_spw_pct"], df["l_rpw_pct"])

    return df
