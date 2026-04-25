import json
import pandas as pd

from src.explainability.explainer import explain_prediction
from data.loader import load_model


def test_explainability_print():
    """
    Debug test: runs the full explainability pipeline and prints the final API response.
    No assertions — this is purely for inspection.
    """

    # Load model + data
    model, scaler, model_features = load_model("models/xgboost_3.pkl")

    row_model = pd.read_csv("tests/prediction/synthetic_dfs/synthetic_row_model.csv")
    row_full = pd.read_csv("tests/prediction/synthetic_dfs/synthetic_row_full.csv")

    # Predict winner
    prob = model.predict_proba(row_model)[0][1]
    winner = "Player A" if prob >= 0.5 else "Player B"

    # Run full explainability pipeline
    explanation = explain_prediction(
        model=model,
        row_model=row_model,
        row_full=row_full,
        probability=prob,
        winner=winner
    )

    # Pretty‑print the final API response
    print("\n\n================ FINAL EXPLAINABILITY API RESPONSE ================\n")
    print(json.dumps(explanation, indent=4,default=str))
    print("\n===================================================================\n")
