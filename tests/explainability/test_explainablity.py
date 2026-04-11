from src.explainability.explainer import explain_prediction

def test_explainability_basic(model_bundle, synthetic_row):
    model = model_bundle["model"]
    scaler = model_bundle["scaler"]

    #predict using real model
    if scaler:
        row_scaled = scaler.transform(synthetic_row)
    else:
        row_scaled = synthetic_row

    prob = model.predict_proba(row_scaled)[0][1]
    winner = "Player A" if prob >= 0.5 else "Player B"

    explanation = explain_prediction(
        model=model,
        row_df=synthetic_row,
        probability=prob,
        winner=winner
    )

    #structure check
    assert isinstance(explanation, dict)

    #narritive excists
    assert "narrative" in explanation
    assert isinstance(explanation["narrative"], str)
    assert len(explanation["narrative"]) > 0

    #so does feature importance
    assert "top_features" in explanation
    assert isinstance(explanation["top_features"], list)
    assert len(explanation["top_features"]) > 0

    #each feature importance entry has the right structure
    first = explanation["top_features"][0]
    assert "feature" in first
    assert "value" in first
    assert "impact" in first
