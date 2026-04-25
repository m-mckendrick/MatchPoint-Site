from pathlib import Path
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from data.loader import load_features
from src.model_training.dataset_preparation import prepare_dataset

import json


def train_xgboost(x_train, y_train):
    model = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        tree_method="hist",   # FAST on CPU
        random_state=42
    )
    model.fit(x_train, y_train)
    return model


def evaluate_xgboost(model, x_test, y_test):
    y_preds = model.predict(x_test)
    y_probs = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_preds),
        "roc_auc": roc_auc_score(y_test, y_probs),
    }


def save_xgboost(model, feature_names, path="models/xgboost_3.pkl"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": feature_names}, path)


def main():
    print("Loading features...")
    df = load_features()

    print("Preparing dataset...")
    x, y = prepare_dataset(df)
    feature_names = list(x.columns)


    print("Splitting train/test...")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training XGBoost...")
    model = train_xgboost(x_train, y_train)

    print(f" Train size: {len(x_train)}")
    print(f" Test size:  {len(x_test)}")

    print("Evaluating model...")
    metrics = evaluate_xgboost(model, x_test, y_test)
    print("Accuracy:", metrics["accuracy"])
    print("ROC AUC:", metrics["roc_auc"])

    print("Saving model...")
    save_xgboost(model, feature_names)

    print("Done!")


if __name__ == "__main__":
    main()
