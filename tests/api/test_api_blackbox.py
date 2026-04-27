def test_api_predict_valid(client):
    payload = {
        "p1_name": "Novak Djokovic",
        "p2_name": "Rafael Nadal",
        "tournament": "Wimbledon",
        "year": 2023,
        "round_label": "QF"
    }

    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "prediction" in data
    assert "explainability" in data

def test_api_predict_structure(client):
    payload = {
        "p1_name": "Novak Djokovic",
        "p2_name": "Rafael Nadal",
        "tournament": "Wimbledon",
        "year": 2023,
        "round_label": "QF"
    }

    data = client.post("/api/predict", json=payload).json()

    assert "winner" in data["prediction"]
    assert "probability" in data["prediction"]

    explain = data["explainability"]
    assert "players" in explain
    assert "tournament" in explain
    assert "narrative" in explain
    assert "widgets" in explain


def test_api_predict_invalid_input(client):
    payload = {
        "p1_name": "Djokovic",
        "p2_name": "Nadal",
        "tournament": "Wimbledon",
        "year": "not_a_year",
        "round_label": "Quarterfinal"
    }

    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422
