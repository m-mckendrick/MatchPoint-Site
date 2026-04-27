from src.explainability.explainer import explain_prediction, WIDGET_REGISTRY
from data.loader import load_features, load_model, load_rankings
import pandas as pd
from src.prediction.match_builder_modular import build_match

def test_explainability_basic():
    model, scaler, model_features = load_model("models/xgboost_3.pkl")

    df = load_features()
    rankings = load_rankings()
    model, scaler, model_features = load_model("models/xgboost_3.pkl")

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


    prob = model.predict_proba(row_model)[0][1]
    winner = "Player A" if prob >= 0.5 else "Player B"

    explanation = explain_prediction(
        row_model=row_model,
        row_full=row_full,
        probability=prob,
        winner=winner
    )

    #structure check
    assert isinstance(explanation, dict)

    #narritive excists
    assert "narrative" in explanation
    assert isinstance(explanation["narrative"], str)
    assert len(explanation["narrative"]) > 0

    # Player profiles exist
    assert "players" in explanation
    assert "p1" in explanation["players"]
    assert "p2" in explanation["players"]

    # Tournament metadata exists
    assert "tournament" in explanation
    assert isinstance(explanation["tournament"], dict)

    # Widget zones exist
    assert "widgets" in explanation
    widgets = explanation["widgets"]
    assert isinstance(widgets, dict)
    assert len(widgets) > 0

    # Each widget zone contains a list of widgets
    for group_name, widget_list in widgets.items():
        assert isinstance(widget_list, list)
        assert len(widget_list) > 0

        # Check widget structure
        w = widget_list[0]
        assert "type" in w
        assert "title" in w
        assert "data" or "text" in w