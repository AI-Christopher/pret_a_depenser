import mlflow
import pandas as pd
import uvicorn
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi import FastAPI, HTTPException, Depends, status
from contextlib import asynccontextmanager
from pythonjsonlogger import jsonlogger
import time
import logging
import json
import os
import sys

# --- 1. CONFIGURATION DES CHEMINS (ROBUSTE) ---
# On récupère le dossier où se trouve ce fichier main.py (c'est-à-dire le dossier 'api')
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# On ajoute ce dossier au path pour que python trouve 'schemas.py' qui est à côté
sys.path.append(CURRENT_DIR)
from schemas import CreditApplication

# Définition des chemins absolus (Tout est dans le dossier 'api')
FEATURES_PATH = os.path.join(CURRENT_DIR, "features_list.json")
MODEL_DIR = os.path.join(CURRENT_DIR, "model_files")
LOG_DIR = os.path.join(CURRENT_DIR, "production_logs")
LOG_FILE = os.path.join(LOG_DIR, "api_request_log.jsonl")

# --- 2. CONFIGURATION LOGGING ---
# Création du dossier s'il n'existe pas
os.makedirs(LOG_DIR, exist_ok=True)

# Création du logger
logger = logging.getLogger("api_logger")
logger.setLevel(logging.INFO)

# Configuration du Handler (Écriture fichier)
if not logger.handlers:
    file_handler = logging.FileHandler(LOG_FILE)
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# --- 3. CONFIGURATION MLFLOW ---
MLFLOW_TRACKING_URI = "http://localhost:5001" 
MODEL_NAME = "LightGBM_CreditScoring_Optimized"
MODEL_VERSION = "1" 

model = None
MODEL_COLUMNS = []

# ... (La suite avec def lifespan reste inchangée) ...

# --- LIFESPAN (CHARGEMENT UNIQUE) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gère le cycle de vie de l'application.
    Le modèle est chargé UNE SEULE FOIS au démarrage.
    """
    global model, MODEL_COLUMNS

    # 1. Chargement de la liste des features
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH, "r") as f:
            MODEL_COLUMNS = json.load(f)
            print(f"✅ Liste des {len(MODEL_COLUMNS)} features chargée.")
    else:
        print(f"⚠️ features_list.json non trouvé ici : {FEATURES_PATH}")
    
    
    print("🔄 Chargement du modèle...")
    try:
        # Priorité au modèle local (Production / Docker / Tests)
        if os.path.exists(MODEL_DIR):
            # On passe le chemin absolu du dossier
            model = mlflow.sklearn.load_model(MODEL_DIR)
            print("✅ Modèle chargé depuis le fichier local.")
        else:
            # Fallback (Dev local avec serveur MLflow)
            print(f"⚠️ Dossier local non trouvé ({MODEL_DIR}), tentative MLflow serveur...")
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
            model = mlflow.sklearn.load_model(model_uri)
            print("✅ Modèle chargé depuis le serveur MLflow.")
            
    except Exception as e:
        print(f"❌ Erreur critique lors du chargement : {e}")
    
    yield
    model = None

# --- INITIALISATION API ---
app = FastAPI(
    title="API Prêt à Dépenser",
    description="API de scoring crédit utilisant un modèle LightGBM optimisé.",
    version="1.0.0",
    lifespan=lifespan
)

# --- SÉCURITÉ (Basic Auth) ---
security = HTTPBasic()

# Identifiants pour télécharger les logs (Changez-les si vous voulez)
ADMIN_USER = "admin"
ADMIN_PASSWORD = "password123"

# --- ROUTES ---

@app.get("/")
def read_root():
    return {"status": "alive", "model": MODEL_NAME, "version": MODEL_VERSION}

@app.post("/predict", tags=["Prediction"])
def predict_credit_score(application: CreditApplication):
    """
    Reçoit les données d'un client et retourne la probabilité de défaut.
    """
    start_time = time.time() # Début du chronométrage

    global model
    if not model:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas chargé.")

    try:
        # --- PRÉPARATION DES DONNÉES ---
        data_dict = application.model_dump()

        # On récupère l'âge et le nombre d'années employé
        age = data_dict.pop("AGE_YEARS") 
        employed = data_dict.pop("YEARS_EMPLOYED")
        
        # On calcule les jours négatifs (Années * 365.25 pour les années bissextiles)
        # On utilise -abs() pour être sûr que ce soit négatif
        
        # On injecte la nouvelle clé que le modèle attend
        data_dict["DAYS_BIRTH"] = -int(abs(age) * 365.25)

        if employed > 0:
            data_dict["DAYS_EMPLOYED"] = -int(abs(employed) * 365.25)
        else:
            data_dict["DAYS_EMPLOYED"] = 365243 # Valeur souvent utilisée pour "Non applicable/Chômeur" dans ce dataset
            # Si le modèle ne connait pas 365243, essayez simplement 0.
        
        # 2. Création DataFrame
        df_input = pd.DataFrame([data_dict])
        
        # 3. 🚨 ÉTAPE CRUCIALE : Réalignement des colonnes
        # On force le DataFrame à avoir exactement les colonnes du modèle, dans le bon ordre.
        # Si des colonnes supplémentaires ont été envoyées (non prévues), elles sont ignorées.
        # Si des colonnes manquent (et que le validateur a laissé passer), cela créera des NaN (ou plantera selon le modèle).
        if MODEL_COLUMNS:
             df_input = df_input.reindex(columns=MODEL_COLUMNS)
             df_input = df_input.fillna(0)  # On remplit les NaN avec 0 (ou une autre valeur par défaut si besoin)
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

        # 4. 📝 LOGGING (SUCCÈS)
        # On enregistre les données telles qu'elles sont entrées dans le modèle (df_input)
        # C'est crucial pour détecter le Data Drift plus tard.
        duration = (time.time() - start_time) * 1000
        
        log_data = {
            "timestamp": time.time(),
            "input_features": df_input.iloc[0].to_dict(), # Les features finales
            "prediction_proba": float(probability),
            "prediction_class": int(prediction),
            "decision": decision,
            "latency_ms": round(duration, 2),
            "status": "SUCCESS"
        }
        logger.info("Prediction processed", extra=log_data)

        return {
            "decision": decision,
            "probability_default": float(round(probability, 4)),
            "risk_class": prediction
        }

    except Exception as e:
        # 5. 📝 LOGGING (ERREUR)
        duration = (time.time() - start_time) * 1000
        error_log = {
            "timestamp": time.time(),
            "error_message": str(e),
            "latency_ms": round(duration, 2),
            "status": "FAILURE",
            # On loggue l'input brut pour pouvoir debugger
            "raw_input": application.model_dump()
        }
        logger.error("Prediction failed", extra=error_log)
        
        print(f"Erreur : {e}")
        raise HTTPException(status_code=400, detail=f"Erreur : {str(e)}")

@app.get("/download_logs", tags=["Monitoring"])
def download_logs(credentials: HTTPBasicCredentials = Depends(security)):
    """
    Permet de télécharger le fichier de logs (Protégé par mot de passe).
    """
    # 1. Vérification du mot de passe
    # On compare ce que l'utilisateur tape avec nos constantes
    is_user_ok = credentials.username == ADMIN_USER
    is_pass_ok = credentials.password == ADMIN_PASSWORD
    
    if not (is_user_ok and is_pass_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Identifiants incorrects",
            headers={"WWW-Authenticate": "Basic"},
        )

    # 2. Vérification que le fichier existe
    if not os.path.exists(LOG_FILE):
        raise HTTPException(status_code=404, detail="Aucun log n'a encore été généré.")

    # 3. Envoi du fichier
    return FileResponse(
        path=LOG_FILE, 
        filename="production_logs_backup.jsonl", 
        media_type='application/json'
    )

# Pour lancer directement si on exécute le fichier
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)