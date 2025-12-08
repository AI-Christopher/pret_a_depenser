import mlflow
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from schemas import CreditApplication
import json
import os

# --- CONFIGURATION ---
# URI du trackeur MLflow 
MLFLOW_TRACKING_URI = "http://localhost:5001" 
# Nom du modèle enregistré dans le Registry (Notebook 04)
MODEL_NAME = "LightGBM_CreditScoring_Optimized"
# Version du modèle
MODEL_VERSION = "1" 

# Variable globale pour stocker le modèle
model = None

# --- LIFESPAN (CHARGEMENT UNIQUE) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gère le cycle de vie de l'application.
    Le modèle est chargé UNE SEULE FOIS au démarrage.
    """
    global model, MODEL_COLUMNS

    # 1. Chargement de la liste des features
    if os.path.exists("features_list.json"):
        with open("features_list.json", "r") as f:
            MODEL_COLUMNS = json.load(f)
            print(f"✅ Liste des {len(MODEL_COLUMNS)} features chargée.")
    else:
        print("⚠️ features_list.json non trouvé ! L'API risque de ne pas fonctionner correctement.")

    print("🔄 Chargement du modèle depuis MLflow...")
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        # Chargement via le Model Registry
        model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"✅ Modèle {MODEL_NAME} v{MODEL_VERSION} chargé avec succès !")
    except Exception as e:
        print(f"❌ Erreur critique lors du chargement du modèle : {e}")
        # En production, on pourrait vouloir arrêter l'API si le modèle ne charge pas
    
    yield # L'application tourne ici
    
    print("🛑 Arrêt de l'API, nettoyage des ressources.")
    model = None

# --- INITIALISATION API ---
app = FastAPI(
    title="API Prêt à Dépenser",
    description="API de scoring crédit utilisant un modèle LightGBM optimisé.",
    version="1.0.0",
    lifespan=lifespan
)

# --- ROUTES ---

@app.get("/")
def read_root():
    return {"status": "alive", "model": MODEL_NAME, "version": MODEL_VERSION}

@app.post("/predict", tags=["Prediction"])
def predict_credit_score(application: CreditApplication):
    """
    Reçoit les données d'un client et retourne la probabilité de défaut.
    """
    global model
    if not model:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas chargé.")

    try:
        # 1. Conversion Pydantic -> Dict
        data_dict = application.model_dump()
        
        # 2. Création DataFrame
        df_input = pd.DataFrame([data_dict])
        
        # 3. 🚨 ÉTAPE CRUCIALE : Réalignement des colonnes
        # On force le DataFrame à avoir exactement les colonnes du modèle, dans le bon ordre.
        # Si des colonnes supplémentaires ont été envoyées (non prévues), elles sont ignorées.
        # Si des colonnes manquent (et que le validateur a laissé passer), cela créera des NaN (ou plantera selon le modèle).
        if MODEL_COLUMNS:
             df_input = df_input.reindex(columns=MODEL_COLUMNS)
        else:
             # Fallback si le fichier JSON manquait (évite le crash, mais risque d'erreur modèle)
             print("⚠️ Attention: Réalignement des colonnes impossible (liste manquante)")
        
        # Sécurité supplémentaire : Remplacer les NaN éventuels par 0 si le modèle ne gère pas les NaN natifs
        # df_input = df_input.fillna(0) 

        # 4. Prédiction
        probability = model.predict_proba(df_input)[:, 1][0]
        
        THRESHOLD = 0.50 
        prediction = 1 if probability >= THRESHOLD else 0
        decision = "REFUS" if prediction == 1 else "ACCORD"

        return {
            "decision": decision,
            "probability_default": float(round(probability, 4)),
            "risk_class": prediction
        }

    except Exception as e:
        print(f"Erreur : {e}")
        raise HTTPException(status_code=400, detail=f"Erreur : {str(e)}")

# Pour lancer directement si on exécute le fichier
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)