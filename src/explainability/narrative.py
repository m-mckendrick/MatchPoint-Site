#very simple narrative for prototype
def build_narrative(top_features):
    parts = []

    for f in top_features:
        name = f["feature"]

        if "rank" in name:
            parts.append("ranking difference was a major factor")
        elif "surface" in name:
            parts.append("surface suitability influenced the prediction")
        elif "age" in name:
            parts.append("age difference contributed to the outcome")
        elif "height" in name:
            parts.append("player physique had some impact")

    if not parts:
        return "The model used several factors to generate this prediction."

    return "The prediction was mainly influenced by: " + ", ".join(parts) + "."
