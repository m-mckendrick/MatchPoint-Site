import numpy as np
import shap
import pandas as pd

from data.loader import load_model_shap
from src.model_training.shap_modular import get_feature_groups

#built for xgboost_3
def compute_shap_explanation(row_df):

    # load shap explainer
    explainer = load_model_shap()


    # SHAP values for the single prediction row
    shap_values = explainer.shap_values(row_df)

    # convert to array
    shap_row = shap_values[0]
    feature_names = list(row_df.columns)

    # per feature stats
    per_feature = []
    for feat, val, shap_val in zip(feature_names, row_df.iloc[0].values, shap_row):
        per_feature.append({
            "feature": feat,
            "value": float(val),
            "shap_value": float(shap_val),
            "abs_shap": float(abs(shap_val)),
        })

    per_feature = sorted(per_feature, key=lambda x: x["abs_shap"], reverse=True)

    # grouped stats
    grouped = {}


    groups = get_feature_groups(row_df)

    for group_name, cols in groups.items():

        if callable(cols):
            cols = cols(feature_names)

        cols = [c for c in cols if c in feature_names]

        if not cols:
            continue

        idx = [feature_names.index(c) for c in cols]
        group_shap = shap_row[idx]

        grouped[group_name] = {
            "features": cols,
            "sum_abs_shap": float(np.sum(np.abs(group_shap))),
            "mean_abs_shap": float(np.mean(np.abs(group_shap))),
            "raw_shap_values": {c: float(shap_row[feature_names.index(c)]) for c in cols},
        }

    # sort groups by importance
    grouped_sorted = dict(
        sorted(grouped.items(), key=lambda x: x[1]["sum_abs_shap"], reverse=True)
    )


    return {
        "per_feature": per_feature,
        "grouped": grouped_sorted,
        "total_abs_shap": float(np.sum(np.abs(shap_row))),
        "prediction_logit": float(explainer.expected_value + shap_row.sum()),
    }
