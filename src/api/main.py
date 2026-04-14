from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path

from data.loader import (
    load_model,
    load_rankings,
    load_matches_features,
)
from src.feature_engineering.feature_builder import build_features
from src.prediction.match_builder import build_match
from src.prediction.predictor import predict
from src.model_training.logistical_regression import prepare_dataset

from src.explainability.explainer import explain_prediction


#load model + data
model, scaler = load_model()
matches_df = load_matches_features()
rankings_df = load_rankings()

#get model features
x, y = prepare_dataset(matches_df)
model_features = x.columns.tolist()


app = FastAPI()
app.mount("/static", StaticFiles(directory="web/static"), name="static")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = PROJECT_ROOT / "web"

#front end routes
@app.get("/")
def home():
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/results")
def results_page():
    return FileResponse(WEB_ROOT / "results.html")

#api structure
class PredictRequest(BaseModel):
    p1_name: str
    p2_name: str
    tournament: str
    year: int
    round_label: str

#api endpoint
@app.post("/api/predict")
def api_predict(payload: PredictRequest):

    #build match row
    #has been changed to give the row containing all column and the row the model predicts with
    row_model, row_meta = build_match(
        df=matches_df,
        rankings_df=rankings_df,
        p1_name=payload.p1_name,
        p2_name=payload.p2_name,
        tournament=payload.tournament,
        year=payload.year,
        round_label=payload.round_label,
        model_features=model_features
    )

    #predict probability Player1 wins
    prob_p1 = float(predict(model, scaler, row_model))
    prob_p2 = 1 - prob_p1

    #get winner
    if prob_p1 >= 0.5:
        winner = payload.p1_name
        loser = payload.p2_name
        winner_prob = prob_p1
    else:
        winner = payload.p2_name
        loser = payload.p1_name
        winner_prob = prob_p2

    winner_prob = round(winner_prob * 100, 2)

    explanation = explain_prediction(
        model=model,
        row_model=row_model,
        row_meta=row_meta,
        probability=winner_prob / 100,  # convert back to 0–1
        winner=winner
    )

    #get player cards and tourny metadata form explanation
    players = explanation["players"]
    tournament = explanation["tournament"]
    narrative = explanation["narrative"]

    #widget zone to be added for explanation
    widgets = explanation["widgets"]

    #return JSON
    return {
        "prediction": {
            "winner": winner,
            "probability": winner_prob
        },
        "explainability": {
            "players": players,
            "tournament": tournament,
            "narrative": narrative,
            "widgets": widgets
        }
    }

@app.get("/results")
def results_page():
    return FileResponse(WEB_ROOT / "results.html")
