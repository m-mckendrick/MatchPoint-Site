import numpy as np

def extract_feature_importance(model, row_df, top_n=5):
    row = row_df.iloc[0]
    feature_names = [str(f) for f in row_df.columns]    #ensure json compatibility

    #random forest / XGBoost
    if hasattr(model, "feature_importances_"):
        impacts = model.feature_importances_

    #logistic regression
    elif hasattr(model, "coef_"):
        coefs = model.coef_[0]
        impacts = coefs * row.values

    else:
        impacts = np.zeros(len(feature_names))

    contributions = []
    for feature, value, impact in zip(feature_names, row.values, impacts):
        contributions.append({
            "feature": str(feature),
            "value": float(value),
            "impact": float(impact)
        })

    contributions = sorted(contributions, key=lambda x: abs(x["impact"]), reverse=True)

    return contributions[:top_n]
