import numpy as np
from .narrative import build_narrative
from .stats import extract_stats

def explain_prediction(model, row_df):
    feature_names = row_df.columns
    row = row_df.iloc[0]

    # SHAP will upgrade later
    if hasattr(model, "predict_proba") and hasattr(model, "feature_importances_"):
        #random Forest / XGBoost style
        importances = model.feature_importances_
        impacts = importances
    elif hasattr(model, "coef_"):
        # Logistic Regression style
        coefs = model.coef_[0]
        impacts = coefs * row.values
    else:
        #if neither
        impacts = np.zeros(len(feature_names))

    contributions = []
    for feature, value, impact in zip(feature_names, row.values, impacts):
        contributions.append({
            "feature": feature,
            "value": float(value),
            "impact": float(impact)
        })

    #sort by impact on prediction
    contributions = sorted(contributions, key=lambda x: abs(x["impact"]), reverse=True)

    return {
        "top_features": contributions[:5],
        "narrative": build_narrative(contributions[:5]),
        "stats": extract_stats(row_df)
    }
