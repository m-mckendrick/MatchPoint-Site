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

#build our df with features
def load_feature_data(processed_dir="../../data/processed"):
    df = build_features(processed_dir)
    return df

def prepare_dataset(df: pd.DataFrame):
    #step 1 drop columns that are not useful
    #keep only numeric, meaningful, predictive features.
    y=df["label"]

    cols_to_drop = [
        #ids
        "tourney_id", "match_num", "winner_id", "loser_id",

        #raw text
        "score", "tiebreak_scores", "tourney_name", "winner_entry", "loser_entry", "winner_name", "winner_ioc", "loser_name", "loser_ioc",

        #raw categories that are replaced
        "surface", "round", "tourney_level","winner_hand", "loser_hand",

        #raw date
        "tourney_date",
    ]


    #make sure columns arnt na e.g(tiebreak scores)
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]

    #so also drop id columns
    id_cols = [c for c in df.columns if "id" in c.lower()]
    cols_to_drop.extend(id_cols)

    x=df.drop(columns=cols_to_drop)

    #model cant handle na so replace na with 0
    x = x.fillna(0)

    print("Non-numeric columns:", x.select_dtypes(include="object").columns.tolist())


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

    df = load_feature_data()

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
    '''
    print("Saving model...")
    save_logistical_regression(model)
    '''
    print("Done!")

if __name__ == "__main__":
    main()
