from src.explainability.narrative import narrative_pressure

def build_widgets(row_model, row_full, shap_group):


    p1_name = row_full["p1_name"]
    p2_name = row_full["p2_name"]

    # Narrative text (SHAP-driven)
    narrative_text = narrative_pressure(
        shap_group=shap_group,
        p1_name=p1_name,
        p2_name=p2_name
    )

    # Extract pressure values
    p1_index = row_model["p1_form_all_5_pressure_index"]
    p2_index = row_model["p2_form_all_5_pressure_index"]

    # Break point stats
    p1_bp_save = row_model["p1_form_all_5_bp_save_pct"]
    p2_bp_save = row_model["p2_form_all_5_bp_save_pct"]

    p1_bp_conv = row_model["p1_form_all_5_bp_conversion_pct"]
    p2_bp_conv = row_model["p2_form_all_5_bp_conversion_pct"]

    return [

        {
            "type": "group_narrative",
            "title": "Pressure & Clutch Overview",
            "group": "Pressure & Clutch Form",
            "text": narrative_text
        },

        {
            "type": "comparison",
            "title": f"Pressure Index ({p1_name} vs {p2_name})",
            "icon": "🔥",
            "group": "Pressure & Clutch Form",
            "data": {
                "label_p1": p1_name,
                "label_p2": p2_name,
                "p1": p1_index,
                "p2": p2_index,
                "diff": p1_index - p2_index
            }
        },

        {
            "type": "comparison",
            "title": f"Break‑Point Performance ({p1_name} vs {p2_name})",
            "icon": "🎯",
            "group": "Pressure & Clutch Form",
            "data": {
                "label_p1": p1_name,
                "label_p2": p2_name,
                "p1_bp_save": p1_bp_save,
                "p2_bp_save": p2_bp_save,
                "p1_bp_conv": p1_bp_conv,
                "p2_bp_conv": p2_bp_conv
            }
        },

        {
            "type": "shap_bar",
            "title": "SHAP Impact – Pressure",
            "icon": "📊",
            "group": "Pressure & Clutch Form",
            "data": shap_group
        }
    ]
