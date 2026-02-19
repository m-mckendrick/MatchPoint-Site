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

#build our df with features
def load_feature_data(processed_dir):
    df = build_features(processed_dir)
    return df

def prepare_dataset(df: pd.DataFrame):
    #step 1 drop columns that are not useful
    #keep only numeric, meaningful, predictive features.
    y=df["label"]

    cols_to_drop = [
        #ids
        "tourney_id","match_num", "p1_id", "p2_id",
        # Raw text fields
        "tourney_name", "score", "tiebreak_scores", "p1_name", "p2_name", "p1_ioc", "p2_ioc", "p1_entry", "p2_entry",
        # Raw categories (replaced by encoded versions)
        "tourney_level", "round", "surface", "p1_hand", "p2_hand",
        # Raw date
        "tourney_date",
        # Perspective column (string)
        "perspective",
    ]


    #make sure columns arnt na e.g(tiebreak scores)
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]

    x=df.drop(columns=cols_to_drop)

    #model cant handle na so replace na with 0 probably figure out individual
    x = x.fillna(0)

    print("Non-numeric columns:", x.select_dtypes(include="object").columns.tolist())
    print("columns used:",x.columns.tolist())

    return x,y

#
def train_logistical_regression(x_train, y_train):
    model = LogisticRegression()
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
