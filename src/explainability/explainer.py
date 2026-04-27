from data.loader import load_model_shap
from .player_profiles import build_player_profiles
from .shap_values import compute_shap_explanation
from .tournament_meta import extract_tournament_meta
from .narrative import build_narrative_groups

# import widget builders
from .widgets.ranking_profile import build_widgets as build_profile_ranking_widgets
from .widgets.elo_rating import build_widgets as build_elo_widgets
from .widgets.serve_return import build_widgets as build_serve_return_widgets
from .widgets.pressure import build_widgets as build_pressure_widgets
from .widgets.fatigue import build_widgets as build_fatigue_widgets
from .widgets.context import build_widgets as build_context_widgets


# mapping SHAP group to widget builder
WIDGET_REGISTRY = {
    "Player Profile & Ranking": build_profile_ranking_widgets,
    "Elo Ratings": build_elo_widgets,
    "Serve & Return Form": build_serve_return_widgets,
    "Pressure & Clutch Form": build_pressure_widgets,
    "Fatigue": build_fatigue_widgets,
    "Match Context": build_context_widgets
}

# maps teh actual group names to readable ones for the site display
GROUP_NAME_MAP = {
    "Elo Ratings": "Elo Ratings",
    "Player Profile & Ranking": "Player Profile & Ranking",

    "Serve & Return": "Serve & Return Form",
    "Pressure Performance": "Pressure & Clutch Form",
    "Context": "Match Context",

    "Fatigue": "Fatigue"
}



def explain_prediction(row_model, row_full, probability, winner):


    shap_info = compute_shap_explanation(row_df=row_model)

    narrative = build_narrative_groups(
        grouped_shap=shap_info["grouped"],
        winner=winner
    )

    widget_zones = {}

    for raw_group_name, group_data in shap_info["grouped"].items():

        #convert internal group name to readable
        group_name = GROUP_NAME_MAP.get(raw_group_name, raw_group_name)

        print("SHAP GROUP:", raw_group_name)

        if group_name in WIDGET_REGISTRY:
            widget_builder = WIDGET_REGISTRY[group_name]

            widgets = widget_builder(
                row_model.iloc[0],
                row_full.iloc[0],
                group_data
            )
        else:
            widgets = []

        widget_zones[group_name] = widgets

    # final api response
    return {
        "winner": winner,
        "probability": float(probability),
        "players": build_player_profiles(row_full),
        "tournament": extract_tournament_meta(row_full),
        "narrative": narrative,
        "widgets": widget_zones
    }
