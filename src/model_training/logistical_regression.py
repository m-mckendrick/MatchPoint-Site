from src.feature_engineering.feature_builder import build_features
from src.model_training.dataset_preparation import prepare_dataset

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib

from data.loader import (
    load_matches,
    load_rankings,
    load_matches,
)

from src.feature_engineering.feature_builder import build_features


'''
logical regression
nutshell: calculate probability to 0 or 1 (no or yes)
rule is set for 0 or 1 eg. if over 0.4 predict 1
graph created is a s shaped line
'''

#use standard scaler
#it standardises every numeric feature so that they all live on the same scale
def train_logistical_regression(x_train, y_train):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    #because my features vary in scale alot a the optimiser need more steps than normal(100) to find the best solution
    model = LogisticRegression(max_iter=2000)
    model.fit(x_train_scaled, y_train)
    return model, scaler

#ROC AUC: 69% of the time, the model ranks a winning match higher than a losing match
def evaluate_logistical_regression(model, scaler, x_test, y_test):
    x_test_scaled = scaler.transform(x_test)
    y_preds = model.predict(x_test_scaled)
    y_probs = model.predict_proba(x_test_scaled)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_preds),
        "roc_auc": roc_auc_score(y_test, y_probs),
    }

def save_logistical_regression(model, scaler, path="models/logreg_baseline.pkl"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, path)

def main():
    print("Loading and building features...")
    df = load_matches()
    df = build_features(df)

    print ("Training Columns...")
    print(df.columns)

    print("Preparing dataset...")
    x, y = prepare_dataset(df)

    print("Splitting train/test...")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training logistic regression...")
    model, scaler = train_logistical_regression(x_train, y_train)

    print("Evaluating model...")
    metrics = evaluate_logistical_regression(model, scaler, x_test, y_test)

    print("Accuracy:", metrics["accuracy"])

    print("ROC AUC:", metrics["roc_auc"])

    print("Saving model...")
    save_logistical_regression(model, scaler)

    print("Done!")

if __name__ == "__main__":
    main()
