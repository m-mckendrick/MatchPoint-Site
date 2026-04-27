from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import time

from data.loader import (
    load_model,
    load_rankings,
    load_features,
)

from src.prediction.match_builder_modular import build_match
from src.prediction.predictor import predict
from src.model_training.logistical_regression import prepare_dataset
from src.explainability.explainer import explain_prediction
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_3.pkl"


#load model + data
model, scaler, features = load_model(str(MODEL_PATH))
features_df = load_features()
rankings_df = load_rankings()

#get model features
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = PROJECT_ROOT / "web"

app = FastAPI()
app.mount("/static", StaticFiles(directory="web/static"), name="static")


#helper function that converts numpty types in api response to python types
def to_py(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.bool_)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_py(v) for v in obj]
    return obj


#front end routes
@app.get("/")
def home():
    return FileResponse(WEB_ROOT / "home.html")


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

    #validate values passed through
    valid_players = set(features_df["p1_name"].unique()) | set(features_df["p2_name"].unique())

    if payload.p1_name not in valid_players:
        return JSONResponse(
            status_code=400,
            content={"error": f"Player '{payload.p1_name}' not found in dataset."}
        )

    if payload.p2_name not in valid_players:
        return JSONResponse(
            status_code=400,
            content={"error": f"Player '{payload.p2_name}' not found in dataset."}
        )

    if "tournament" in features_df.columns:
        valid_tournaments = set(features_df["tournament"].unique())
        if payload.tournament not in valid_tournaments:
            return JSONResponse(
                status_code=400,
                content={"error": f"Tournament '{payload.tournament}' not found in dataset."}
            )

    if "year" in features_df.columns:
        valid_years = set(features_df["year"].unique())
        if payload.year not in valid_years:
            return JSONResponse(
                status_code=400,
                content={"error": f"Year '{payload.year}' not available in dataset."}
            )

    if "round_label" in features_df.columns:
        valid_rounds = set(features_df["round_label"].unique())
        if payload.round_label not in valid_rounds:
            return JSONResponse(
                status_code=400,
                content={"error": f"Round '{payload.round_label}' is not valid for this dataset."}
            )



    #build match row
    #has been changed to give the row containing all column and the row the model predicts with
    row_model, row_full = build_match(
        df=features_df,
        rankings_df=rankings_df,
        p1_name=payload.p1_name,
        p2_name=payload.p2_name,
        tournament=payload.tournament,
        year=payload.year,
        round_label=payload.round_label,
        model_features=features
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
        row_model=row_model,
        row_full=row_full,
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

    response= {
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
    python_response = to_py(response)

    return JSONResponse(content=python_response)

@app.get("/results")
def results_page():
    return FileResponse(WEB_ROOT / "results.html")
