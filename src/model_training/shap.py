import shap
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from .dataset_preparation import prepare_dataset
from data.loader import load_model

MODEL_PATH = "models/xgboost_3.pkl"
DATA_PATH = "data/processed/2000-2023_features.csv"
GRAPH_DIR = Path("src/model_training/graphs")
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

model, scaler = load_model(MODEL_PATH)

df = pd.read_csv(DATA_PATH)
X, y = prepare_dataset(df)

# compute shap values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# feature groups
groups = {
    "Ranking & Profile": [
        "rank_diff","rank_points_diff",
        "p1_rank","p2_rank","p1_age","p2_age","p1_ht","p2_ht",
        "age_diff","height_diff"
    ],
    "Elo Ratings": [
        "elo_global_diff","p1_elo_global","p2_elo_global",
        "elo_surface_diff","p1_elo_surface","p2_elo_surface"
    ],
    "Form – All Surfaces (5)": [c for c in X.columns if "form_all_5" in c],
    "Form – Same Surface (5)": [c for c in X.columns if "form_same_5" in c],
    "Fatigue": [
        "p1_fatigue_pre","p2_fatigue_pre",
        "p1_fatigue_in","p2_fatigue_in"
    ],
    "Context": [
        c for c in X.columns
        if c.startswith("surface_")
           or c in ["is_grand_slam","is_masters","is_best_of_five",
                    "is_indoor","home_advantage_diff",
                    "round_encoded","tourney_prestige","round_points"]
    ],
}

# grouped shap importance
group_importance = {}

for group_name, cols in groups.items():
    idx = [X.columns.get_loc(c) for c in cols if c in X.columns]
    if not idx:
        continue
    group_shap = shap_values[:, idx]
    group_importance[group_name] = abs(group_shap).mean()

# sort groups by importance
group_importance = dict(sorted(group_importance.items(),
                               key=lambda x: x[1],
                               reverse=True))

#print group importance
print("GROUPED SHAP IMPORTANCE\n")
for g, v in group_importance.items():
    print(f"{g:30s}  {v:.6f}")

# top overall features
abs_shap = np.abs(shap_values).mean(axis=0)
feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": abs_shap
}).sort_values("importance", ascending=False)


print("   TOP 30 FEATURES (GLOBAL SHAP)\n")
print(feature_importance.head(30).to_string(index=False))

# top features per group
for group_name, cols in groups.items():
    cols = [c for c in cols if c in X.columns]
    if not cols:
        continue


    print(f"   TOP FEATURES IN GROUP: {group_name}\n")

    idx = [X.columns.get_loc(c) for c in cols]
    group_shap_vals = np.abs(shap_values[:, idx]).mean(axis=0)

    df_group = pd.DataFrame({
        "feature": cols,
        "importance": group_shap_vals
    }).sort_values("importance", ascending=False)

    print(df_group.head(10).to_string(index=False))

# importance group plot
plt.figure(figsize=(10,6))
plt.barh(list(group_importance.keys()), list(group_importance.values()))
plt.title("Grouped SHAP Importance")
plt.xlabel("Mean SHAP value")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(GRAPH_DIR / "grouped_shap_importance.png", dpi=300)
plt.close()

#feature level-plot per group
for group_name, cols in groups.items():
    cols = [c for c in cols if c in X.columns]
    if not cols:
        continue

    print(f"\n=== Saving SHAP plot for group: {group_name} ===")

    shap.summary_plot(
        shap_values[:, [X.columns.get_loc(c) for c in cols]],
        X[cols],
        plot_type="bar",
        show=False
    )

    safe_name = (
        group_name.replace(" ", "_")
        .replace("&", "and")
        .replace("–", "-")
        .replace("(", "")
        .replace(")", "")
    )

    plt.tight_layout()
    plt.xlabel("Mean SHAP value")
    plt.savefig(GRAPH_DIR / f"shap_{safe_name}.png", dpi=300)
    plt.close()