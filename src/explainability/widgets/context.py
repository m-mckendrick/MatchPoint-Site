from src.explainability.narrative import narrative_context

def build_widgets(row_model, row_full, shap_group):

    p1_name = row_full["p1_name"]
    p2_name = row_full["p2_name"]
    surface = row_full["surface_raw"]
    round_label = row_full.get("round", "Unknown")

    # Narrative text (SHAP-driven)
    narrative_text = narrative_context(
        shap_group=shap_group,
        p1_name=p1_name,
        p2_name=p2_name,
        surface=surface,
        round_label=round_label
    )

    # Extract context values
    is_gs = row_model["is_grand_slam"]
    is_masters = row_model["is_masters"]
    is_best_of_five = row_model["is_best_of_five"]
    is_indoor = row_model["is_indoor"]

    round_encoded = row_model["round_encoded"]
    prestige = row_model["tourney_prestige"]
    round_points = row_model["round_points"]

    p1_home = row_model["p1_home"]
    p2_home = row_model["p2_home"]
    home_diff = row_model["home_advantage_diff"]

    # Surface one-hot flags
    surface_flags = {
        "Carpet": row_model["surface_Carpet"],
        "Clay": row_model["surface_Clay"],
        "Grass": row_model["surface_Grass"],
        "Hard": row_model["surface_Hard"],
    }

    # Determine surface profile
    surface_profile = {
        "Clay": {"speed": "Slow", "bounce": "High", "style": "Topspin, long rallies"},
        "Grass": {"speed": "Fast", "bounce": "Low", "style": "Serve‑and‑volley, big servers and big hitters"},
        "Hard": {"speed": "Medium", "bounce": "Medium", "style": "All‑court, balanced"},
        "Carpet": {"speed": "Very fast", "bounce": "Low", "style": "Aggressive, first‑strike tennis"},
    }.get(surface, {"speed": "Unknown", "bounce": "Unknown", "style": "Unknown"})

    return [

        {
            "type": "group_narrative",
            "title": "Match Context Overview",
            "group": "Match Context",
            "text": narrative_text
        },

        {
            "type": "stat_card",
            "title": "Tournament Importance",
            "icon": "🏆",
            "group": "Match Context",
            "data": {
                "surface": surface,
                "round": round_label,
                "round_encoded": round_encoded,
                "round_points": round_points,
                "prestige": prestige,
                "is_grand_slam": bool(is_gs),
                "is_masters": bool(is_masters),
                "best_of_five": bool(is_best_of_five),
                "indoor": bool(is_indoor)
            }
        },

        {
            "type": "stat_card",
            "title": f"{surface} Surface Profile",
            "icon": "🌍",
            "group": "Match Context",
            "data": {
                "speed": surface_profile["speed"],
                "bounce": surface_profile["bounce"],
                "favoured_style": surface_profile["style"],
                "flags": surface_flags
            }
        },

        {
            "type": "comparison",
            "title": f"Home Advantage ({p1_name} vs {p2_name})",
            "icon": "🏠",
            "group": "Match Context",
            "data": {
                "label_p1": p1_name,
                "label_p2": p2_name,
                "p1": p1_home,
                "p2": p2_home,
                "diff": home_diff
            }
        },

        {
            "type": "shap_bar",
            "title": "SHAP Impact – Match Context",
            "icon": "📊",
            "group": "Match Context",
            "data": shap_group
        }
    ]
