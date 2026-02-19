from src.feature_engineering.feature_builder import build_features

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

'''
logical regression
nutshell: calculate probability to 0 or 1 (no or yes)
rule is set for 0 or 1 eg. if over 0.4 predict 1
graph created is a s shaped line
'''

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

#build our df with match level features
#out df is stats + match level features
def load_feature_data(processed_dir):
    df = build_features(processed_dir)
    return df

#get rid of columns that arn't predictive
def prepare_dataset(df: pd.DataFrame):

    y = df["label"]

    # columns that leak match outcome or depend on the opponent
    leaky_cols = [
        "score", "tiebreak_scores", "minutes", "perspective",

        # match-level stats
        "p1_ace", "p1_df", "p1_svpt", "p1_1stin", "p1_1stwon", "p1_2ndwon",
        "p1_svgms", "p1_bpsaved", "p1_bpfaced",
        "p2_ace", "p2_df", "p2_svpt", "p2_1stin", "p2_1stwon", "p2_2ndwon",
        "p2_svgms", "p2_bpsaved", "p2_bpfaced",

        # engineered match-level stats
        "p1_first_serve_pct", "p1_first_serve_won_pct", "p1_second_serve_won_pct",
        "p1_df_rate", "p1_aces_per_game", "p1_bp_save_pct", "p1_spw_pct",
        "p1_rpw_pct", "p1_bp_conv_pct", "p1_sdi",
        "p2_first_serve_pct", "p2_first_serve_won_pct", "p2_second_serve_won_pct",
        "p2_df_rate", "p2_aces_per_game", "p2_bp_save_pct", "p2_spw_pct",
        "p2_rpw_pct", "p2_bp_conv_pct", "p2_sdi",
    ]

    # id and text columns
    id_text_cols = [
        "tourney_id", "match_num", "p1_id", "p2_id",
        "tourney_name", "p1_name", "p2_name",
        "p1_ioc", "p2_ioc", "p1_entry", "p2_entry",
        "tourney_date", "round", "tourney_level", "p1_hand", "p2_hand"
    ]

    cols_to_drop = [c for c in leaky_cols + id_text_cols if c in df.columns]

    x = df.drop(columns=cols_to_drop)

    # Fill NA
    x = x.fillna(0)

    print("Non-numeric columns:", x.select_dtypes(include="object").columns.tolist())
    print("columns used:",x.columns.tolist())
    return x, y


#use standard scaler
def train_logistical_regression(x_train, y_train):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    model = LogisticRegression(max_iter=2000)
    model.fit(x_train, y_train)
    return model

#
def evaluate_logistical_regression(model, x_test, y_test):
    y_preds = model.predict(x_test)
    y_probs = model.predict_proba(x_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_preds),
        "roc_auc": roc_auc_score(y_test, y_probs),
    }

def save_logistical_regression(model, path="models/logreg_baseline.pkl"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

def main():
    print("Loading and building features...")
    print(Path("../../data/processed").resolve())

    df = load_feature_data(PROCESSED_DIR)

    print("Preparing dataset...")
    x, y = prepare_dataset(df)

    print("Splitting train/test...")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training logistic regression...")
    model = train_logistical_regression(x_train, y_train)

    print("Evaluating model...")
    metrics = evaluate_logistical_regression(model, x_test, y_test)

    print("Accuracy:", metrics["accuracy"])

    print("ROC AUC:", metrics["roc_auc"])

    print("Saving model...")
    save_logistical_regression(model)

    print("Done!")

if __name__ == "__main__":
    main()
