from src.explainability.narrative import narrative_fatigue

def build_widgets(row_model, row_full, shap_group):

    # Narrative text (SHAP-driven)
    narrative_text = narrative_fatigue(
        shap_group=shap_group,
        p1_name=row_full["p1_name"],
        p2_name=row_full["p2_name"]
    )

    p1_name = row_full["p1_name"]
    p2_name = row_full["p2_name"]

    # Extract fatigue values
    p1_pre = row_model["p1_fatigue_pre"]
    p2_pre = row_model["p2_fatigue_pre"]
    p1_in = row_model["p1_fatigue_in"]
    p2_in = row_model["p2_fatigue_in"]


    widgets = [
        {
            "type": "group_narrative",
            "title": "Fatigue Overview",
            "group": "Fatigue",
            "text": narrative_text
        },

        {
            "type": "comparison",
            "title": f"Fatigue Comparison ({p1_name} vs {p2_name})",
            "icon": "😮‍💨",
            "group": "Fatigue",
            "data": {
                "label_p1": p1_name,
                "label_p2": p2_name,
                "p1_pre-tourny": p1_pre,
                "p2_pre-tourny": p2_pre,
                "p1_in-tourny": p1_in,
                "p2_in-tourny": p2_in
            }
        },

        {
            "type": "shap_bar",
            "title": "SHAP Impact – Fatigue",
            "icon": "📊",
            "group": "Fatigue",
            "data": shap_group
        }
    ]


    return widgets
