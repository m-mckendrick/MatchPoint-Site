def predict(model, scaler, row_df):
    if scaler:
        row_scaled = scaler.transform(row_df)
    else:
        row_scaled = row_df

    prob = model.predict_proba(row_scaled)[0][1]
    return float(prob)
