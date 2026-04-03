from src.feature_engineering.feature_builder import build_features
from src.model_training.dataset_preparation import prepare_dataset

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

"""
random forest 
Each tree is trained on a slightly different sample of the data and 
only sees a random subset of features at each split, 
which forces the trees to learn different patterns rather than all memorising the same structure
"""

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def load_feature_data(processed_dir):
    df = build_features(processed_dir)
    return df

def train_random_forest(x_train, y_train):
    model = RandomForestClassifier(
        n_estimators=300,       #number of trees
        max_depth=None,         # allow full depth
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",    # good default for classification
        random_state=42,
        n_jobs=-1               # use all CPU cores
    )
    model.fit(x_train, y_train)
    return model

def evaluate_random_forest(model, x_test, y_test):
    y_preds = model.predict(x_test)
    y_probs = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_preds),
        "roc_auc": roc_auc_score(y_test, y_probs),
    }

def save_random_forest(model, path="models/random_forest.pkl"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model}, path)


def main():
    print("Loading and building features...")
    df = load_feature_data(PROCESSED_DIR / "matches")

    print("Preparing dataset...")
    x, y = prepare_dataset(df)

    print("Splitting train/test...")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Random Forest...")
    model = train_random_forest(x_train, y_train)

    print("Evaluating model...")
    metrics = evaluate_random_forest(model, x_test, y_test)

    print("Accuracy:", metrics["accuracy"])
    print("ROC AUC:", metrics["roc_auc"])

    print("Saving model...")
    save_random_forest(model)

    print("Done!")

if __name__ == "__main__":
    main()
