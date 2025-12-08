import sys
import os
import pytest
from fastapi.testclient import TestClient

# Ajout du chemin pour trouver l'API
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

# --- FIXTURE (La solution au problème 503) ---
@pytest.fixture(scope="module")
def client():
    # Le 'with' déclenche le lifespan (chargement du modèle)
    with TestClient(app) as c:
        yield c
    # À la fin du 'with', l'API s'éteint proprement

# --- TESTS ---

def test_read_root(client):
    """Vérifie que l'API est en vie"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"

def test_predict_nominal_accord(client):
    """Cas nominal : Données valides"""
    payload = {
        "AMT_INCOME_TOTAL": 200000.0,
        "AMT_CREDIT": 500000.0,
        # CORRECTION ICI : Minuscules comme dans schemas.py
        "AGE_YEARS": 45,       
        "YEARS_EMPLOYED": 15,
        "EXT_SOURCE_2": 0.75,
        "EXT_SOURCE_3": 0.70
    }
    response = client.post("/predict", json=payload)
    
    # Debug si ça échoue encore
    if response.status_code != 200:
        print(f"Erreur API: {response.json()}")
        
    assert response.status_code == 200
    assert "decision" in response.json()

def test_predict_missing_data(client):
    """Cas critique : Champ obligatoire manquant"""
    payload = {
        "AMT_INCOME_TOTAL": 200000.0,
        # AMT_CREDIT manque
        "AGE_YEARS": 45
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_invalid_range_age(client):
    """Cas critique : Âge impossible"""
    payload = {
        "AMT_INCOME_TOTAL": 200000.0,
        "AMT_CREDIT": 500000.0,
        "AGE_YEARS": 5 
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422