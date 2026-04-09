from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path

from src.prediction.loader import load_model, load_matches, load_rankings
from src.prediction.match_builder import build_match
from src.prediction.predictor import predict
from src.model_training.logistical_regression import prepare_dataset

from src.explainability.explainer import explain_prediction

#loade model + data
model, scaler = load_model()
matches_df = load_matches()
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
    row_df = build_match(
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
    prob_p1 = float(predict(model, scaler, row_df))
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

#player cards
    players = {
        "p1": {
            "name": payload.p1_name,
            "ranking": int(row_df["p1_rank"].iloc[0]),
            "age": int(row_df["p1_age"].iloc[0])
        },
        "p2": {
            "name": payload.p2_name,
            "ranking": int(row_df["p2_rank"].iloc[0]),
            "age": int(row_df["p2_age"].iloc[0])
        }
    }

    #torny metadata
    #get surface string
    surface = "Unknown"
    surface_map = {
        "surface_grass": "Grass",
        "surface_clay": "Clay",
        "surface_hard": "Hard",
        "surface_carpet": "Carpet"
    }

    for col, name in surface_map.items():
        if col in row_df.columns and int(row_df[col].iloc[0]) == 1:
            surface = name
            break

    tournament = {
        "name": payload.tournament,
        "year": payload.year,
        "round": payload.round_label,
        "surface": surface
    }


    explanation = explain_prediction(
        model=model,
        row_df=row_df,
        probability=winner_prob / 100,  # convert back to 0–1
        winner=winner
    )

    narrative = explanation["narrative"]

    #widget zone to be added for explanation
    widgets = explanation["widgets"]

    #return JSON
    return {
        "winner": winner,
        "probability": winner_prob,
        "players": players,
        "tournament": tournament,
        "narrative": narrative,
        "widgets": widgets
    }

@app.get("/results")
def results_page():
    return FileResponse(WEB_ROOT / "results.html")
