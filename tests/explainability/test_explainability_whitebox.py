import json
import pandas as pd

from src.explainability.explainer import explain_prediction
from data.loader import load_features, load_model, load_rankings
from src.prediction.match_builder_modular import build_match


def test_explainability_print():
    """
    Debug test: runs the full explainability pipeline and prints the final API response.
    No assertions — this is purely for inspection.
    """
    model, scaler, model_features = load_model("models/xgboost_3.pkl")

    df = load_features()
    rankings = load_rankings()

    row_model, row_full = build_match(
        df=df,
        rankings_df=rankings,
        p1_name="Novak Djokovic",
        p2_name="Daniil Medvedev",
        tournament="US Open",
        year=2018,
        round_label="F",
        model_features=model_features,
    )


    # Predict winner
    prob = model.predict_proba(row_model)[0][1]
    winner = "Player A" if prob >= 0.5 else "Player B"

    # Run full explainability pipeline
    explanation = explain_prediction(
        row_model=row_model,
        row_full=row_full,
        probability=prob,
        winner=winner
    )

    # Pretty‑print the final API response
    print("\n\n================ FINAL EXPLAINABILITY API RESPONSE ================\n")
    print(json.dumps(explanation, indent=4,default=str))
    print("\n===================================================================\n")
