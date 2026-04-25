import shap
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt

from data.loader import load_model
from src.model_training.dataset_preparation import prepare_dataset


MODEL_PATH = "models/xgboost_3.pkl"
DATA_PATH = "data/processed/2000-2023_features.csv"

EXPLAINER_PATH = Path("models/shap_explainer_xgboost_3.pkl")
FEATURE_NAMES_PATH = Path("models/model_feature_names.json")
GRAPH_DIR = Path("src/model_training/shap_graphs")
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


def build_shap_explainer(model):
    print("Building SHAP explainer (slow, done once)...")
    explainer = shap.TreeExplainer(model)
    return explainer


def save_explainer(explainer, feature_names):
    joblib.dump(explainer, EXPLAINER_PATH)
    print(f"Saved SHAP explainer → {EXPLAINER_PATH}")

    with open(FEATURE_NAMES_PATH, "w") as f:
        json.dump(feature_names, f)
    print(f"Saved feature names → {FEATURE_NAMES_PATH}")


def compute_global_shap(explainer, X):
    print("Computing SHAP values for full dataset...")
    shap_values = explainer.shap_values(X)
    return shap_values


def get_feature_groups(X):
    return {
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


def compute_group_importance(groups, X, shap_values):
    group_importance = {}

    for group_name, cols in groups.items():
        idx = [X.columns.get_loc(c) for c in cols if c in X.columns]
        if not idx:
            continue
        group_shap = shap_values[:, idx]
        group_importance[group_name] = abs(group_shap).mean()

    return dict(sorted(group_importance.items(),
                       key=lambda x: x[1],
                       reverse=True))


def plot_group_importance(group_importance):
    plt.figure(figsize=(10,6))
    plt.barh(list(group_importance.keys()), list(group_importance.values()))
    plt.title("Grouped SHAP Importance")
    plt.xlabel("Mean SHAP value")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "grouped_shap_importance.png", dpi=300)
    plt.close()
    print("Saved grouped SHAP importance plot.")


def plot_group_shap(groups, X, shap_values):
    for group_name, cols in groups.items():
        cols = [c for c in cols if c in X.columns]
        if not cols:
            continue

        print(f"Saving SHAP plot for group: {group_name}")

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


def main(save_graphs=False, compute_global=False):
    print("Loading model + dataset...")
    model, scaler, features = load_model(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)
    X, y = prepare_dataset(df)
    feature_names = list(X.columns)

    #added save
    explainer = build_shap_explainer(model)
    save_explainer(explainer, feature_names)

    if not compute_global:
        print("Skipping global SHAP analysis.")
        return

    shap_values = compute_global_shap(explainer, X)
    groups = get_feature_groups(X)
    group_importance = compute_group_importance(groups, X, shap_values)

    print("\nGROUPED SHAP IMPORTANCE\n")
    for g, v in group_importance.items():
        print(f"{g:30s}  {v:.6f}")

    if save_graphs:
        plot_group_importance(group_importance)
        plot_group_shap(groups, X, shap_values)


if __name__ == "__main__":
    main()
