import joblib
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from data.loader import load_features
from src.model_training.dataset_preparation import prepare_dataset


def main(model_path="models/xgboost_3.pkl"):
    print("Loading model...")
    bundle = joblib.load(model_path)
    model = bundle["model"]

    print("Loading features...")
    df = load_features()
    X, _ = prepare_dataset(df)

    print("Computing fast feature importance (gain)...")
    booster = model.get_booster()
    importance = booster.get_score(importance_type="gain")

    # convert to DataFrame
    imp_df = pd.DataFrame({
        "feature": list(importance.keys()),
        "importance": list(importance.values())
    })

    imp_df = imp_df.sort_values("importance", ascending=False)

    # Print full list
    print("\n=== FULL FEATURE IMPORTANCE LIST (sorted) ===")
    print(imp_df.to_string(index=False))


if __name__ == "__main__":
    main()
