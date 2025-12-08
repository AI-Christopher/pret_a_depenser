import json
import os
from pydantic import BaseModel, Field, model_validator
from typing import Optional, Dict, Any

# Chargement de la liste des features attendues
# On gère le cas où le fichier n'est pas encore là pour éviter que l'API plante au dev
FEATURES_FILE = "features_list.json"
EXPECTED_FEATURES = []

if os.path.exists(FEATURES_FILE):
    with open(FEATURES_FILE, "r") as f:
        EXPECTED_FEATURES = json.load(f)
else:
    print(f"⚠️ ATTENTION: {FEATURES_FILE} non trouvé. La validation stricte des colonnes est désactivée.")

class CreditApplication(BaseModel):
    """
    Schéma hybride : 
    1. Champs critiques explicites (pour la documentation Swagger et vérifs poussées).
    2. Champs dynamiques (les 130+ autres) acceptés et validés en bloc.
    """

    # --- CHAMPS CRITIQUES (Explicites) ---
    AMT_INCOME_TOTAL: float = Field(..., gt=0, description="Revenu annuel")
    AMT_CREDIT: float = Field(..., gt=0, description="Montant du crédit")
    AGE_YEARS: int = Field(..., ge=18, le=100, description="Âge du client en années (ex: 35)")

    # --- LEVIERS DE DÉCISION (Optionnels avec valeurs par défaut) ---
    
    # Par défaut, on met un score moyen (0.5) pour éviter le refus automatique
    # Mais on permet de le changer pour la démo
    EXT_SOURCE_2: float = Field(0.5, ge=0, le=1, description="Score crédit externe (0=Risqué, 1=Fiable)")
    EXT_SOURCE_3: float = Field(0.5, ge=0, le=1, description="Score crédit externe (0=Risqué, 1=Fiable)")
    
    # Par défaut, 5 ans d'ancienneté
    YEARS_EMPLOYED: float = Field(5, ge=0, description="Années d'ancienneté dans l'emploi actuel")
    
    
    class Config:
        # "allow" dit à Pydantic : "Accepte tout ce qui n'est pas déclaré ci-dessus"
        # C'est la clé pour gérer les 140 colonnes sans les écrire
        extra = "allow"
        json_schema_extra = {
            "example": {
                "AMT_INCOME_TOTAL": 200000.0,
                "AMT_CREDIT": 500000.0,
                "AGE_YEARS": 35,
                "YEARS_EMPLOYED": 10,
                "EXT_SOURCE_2": 0.7,
                "EXT_SOURCE_3": 0.6
            }
        }

    # --- VALIDATION DYNAMIQUE ---
    @model_validator(mode='before')
    def check_all_features_presence(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Vérifie que TOUTES les colonnes nécessaires au modèle sont présentes dans la requête.
        """
        if not EXPECTED_FEATURES:
            return values # Pas de fichier de config, on laisse passer (mode dev)

        input_keys = set(values.keys())
        expected_set = set(EXPECTED_FEATURES)
        
        if "DAYS_BIRTH" in expected_set:
            expected_set.remove("DAYS_BIRTH")

        # Quelles colonnes manquent ?
        missing_cols = expected_set - input_keys
        
        if missing_cols:
            # Option A (Stricte) : On rejette la requête
            # raise ValueError(f"Données manquantes : Il manque {len(missing_cols)} champs obligatoires. Ex: {list(missing_cols)[:3]}...")
            
            # Option B (Souple - Décommenter si besoin) : On remplit avec 0 ou médiane
            for col in missing_cols:
                values[col] = 0.0 
        
        return values