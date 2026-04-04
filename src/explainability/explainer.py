from .player_profiles import build_player_profiles
from .feature_importance import extract_feature_importance
from .tournament_meta import extract_tournament_meta
from .narrative import build_narrative

#conductor gather all relevant info for display in web page 
def explain_prediction(model, row_df, probability, winner):
    top_features = extract_feature_importance(model, row_df)

    return {
        "winner": winner,
        "probability": float(probability),
        "players": build_player_profiles(row_df),
        "tournament": extract_tournament_meta(row_df),
        "top_features": top_features,
        "narrative": build_narrative(top_features, winner)
    }
