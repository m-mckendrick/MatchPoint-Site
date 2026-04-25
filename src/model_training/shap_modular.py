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

    serve_return = [
        c for c in X
        if (
                "serve" in c
                or "return" in c
                or "hold" in c
                or "aces_per_game" in c
                or "double_fault" in c
        )
    ]

    pressure = [
        c for c in X
        if (
                "pressure" in c
                or "bp_save" in c
                or "bp_conversion" in c
        )
    ]

    return {

        "Ranking": [
            "p1_rank", "p2_rank",
            "rank_diff",
            "p1_rank_points", "p2_rank_points",
            "rank_points_diff",
        ],

        "Profile": [
            "p1_age", "p2_age", "age_diff",
            "p1_ht", "p2_ht", "height_diff",
            "p1_lefty", "p2_lefty", "lefty_diff",
        ],

        "Elo Ratings": [
            "p1_elo_global", "p2_elo_global", "elo_global_diff",
            "p1_elo_surface", "p2_elo_surface", "elo_surface_diff",
        ],

        "Serve & Return": serve_return,

        "Pressure Performance": pressure,

        "Form – All Surfaces": [
            c for c in X
            if (
                    "form_all_5" in c
                    and c not in serve_return
                    and c not in pressure
            )
        ],

        "Form – Same Surface": [
            c for c in X
            if (
                    "form_same_5" in c
                    and c not in serve_return
                    and c not in pressure
            )
        ],

        "Fatigue": [
            "p1_fatigue_pre", "p2_fatigue_pre",
            "p1_fatigue_in", "p2_fatigue_in",
        ],

        "Context": [
            c for c in X
            if c in [
                "is_grand_slam", "is_masters", "is_best_of_five",
                "is_indoor", "round_encoded",
                "tourney_prestige", "round_points",
                "p1_home", "p2_home", "home_advantage_diff",
            ] or c.startswith("surface_")
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
    main(save_graphs=True, compute_global=True)
