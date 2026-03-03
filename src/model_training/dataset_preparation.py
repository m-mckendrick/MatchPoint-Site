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

    ALLOWED_FEATURES = [ "p1_rank", "p1_age", "p1_ht", "p1_rank_points", "p2_rank", "p2_age",
                         "p2_ht", "p2_rank_points", "rank_diff", "rank_points_diff",
                         "is_grand_slam", "is_masters", "is_best_of_five",
                         "p1_lefty", "p2_lefty", "lefty_diff", "surface_Clay", "surface_Grass",
                         "surface_Hard", "round_encoded", "match_year", "age_diff", "height_diff",
                         "tourney_prestige"
                         ]

    x = df[ALLOWED_FEATURES].copy()
    x = handle_na(x)

    print("columns used:",x.columns.tolist())
    return x, y

def handle_na(df: pd.DataFrame):
    # Numeric columns: fill with 0
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[num_cols] = df[num_cols].fillna(0)
    # Categorical columns: fill with "Unknown"
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    return df