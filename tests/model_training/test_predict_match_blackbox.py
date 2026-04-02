import pandas as pd
from src.model_training.predict_match import load_model, load_data, build_match, load_rankings
from src.model_training.logistical_regression import prepare_dataset

#testing inputs to output
def test_predict_match_blackbox():
    # load model and data
    model, scaler = load_model()
    df = load_data()
    ranking_df = load_rankings()

    # get model features
    x, y = prepare_dataset(df)
    model_features = x.columns.tolist()

    # build a synthetic match
    row = build_match(
        df=df,
        rankings_df = ranking_df,
        p1_name="Novak Djokovic",
        p2_name="Daniil Medvedev",
        tournament="US Open",
        year=2018,
        round_label="F",
        model_features=model_features
    )

    # ensure row is a DataFrame with correct columns
    assert isinstance(row, pd.DataFrame)
    assert list(row.columns) == model_features

    # Scale + predict
    row_scaled = scaler.transform(row)
    prob = model.predict_proba(row_scaled)[0][1]

    # Probability must be between 0 and 1
    assert 0 <= prob <= 1