import pandas as pd
import json
import os
import sys
import warnings # <--- Pour faire taire les warnings Numpy
import logging  # <--- Pour faire taire les warnings Evidently

# --- 1. SILENCE RADIO ---
# On ignore les warnings de division par zéro (dus au faible volume de données)
warnings.filterwarnings("ignore")
# On monte le niveau de log pour ignorer les avertissements de type "root"
logging.getLogger().setLevel(logging.ERROR)

# --- IMPORTS COMPATIBLES EVIDENTLY 0.6.0 ---
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# --- CONFIGURATION DES CHEMINS ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# Fichiers d'entrée
REF_PATH = os.path.join(CURRENT_DIR, "reference_sample.csv")
LOGS_PATH = os.path.join(PROJECT_ROOT, "api", "production_logs", "api_request_log.jsonl")

# Fichiers de sortie
REPORTS_DIR = os.path.join(CURRENT_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

HTML_REPORT_PATH = os.path.join(REPORTS_DIR, "drift_report.html")
JSON_REPORT_PATH = os.path.join(REPORTS_DIR, "drift_metrics.json")

def load_production_logs():
    """Charge et transforme les logs JSONL en DataFrame utilisable."""
    print(f"📂 Lecture des logs depuis : {LOGS_PATH}")
    
    if not os.path.exists(LOGS_PATH):
        print("⚠️  Aucun fichier de log trouvé.")
        return None

    data = []
    # Lecture ligne par ligne pour gérer les erreurs JSON potentielles
    with open(LOGS_PATH, 'r') as f:
        for line in f:
            try:
                log = json.loads(line)
                # On ne garde que les succès
                if log.get("status") == "SUCCESS":
                    # On récupère les features 
                    row = log.get("input_features", {}).copy()
                    data.append(row)
            except json.JSONDecodeError:
                continue
    
    if not data:
        print("⚠️  Logs vides ou illisibles.")
        return None

    return pd.DataFrame(data)

def run_analysis():
    print("🚀 Démarrage de l'analyse de Drift (Evidently 0.6.0)...")

    # 1. Chargement Référence
    if not os.path.exists(REF_PATH):
        print(f"❌ Erreur : Référence introuvable ({REF_PATH}).")
        return

    df_ref = pd.read_csv(REF_PATH)
    print(f"✅ Référence chargée : {df_ref.shape}")

    # 2. Chargement Production
    df_prod = load_production_logs()
    
    if df_prod is None or len(df_prod) < 2:
        print("⚠️  Trop peu de données de prod. Fais quelques prédictions sur l'API d'abord !")
        return

    # 3. Alignement des colonnes
    common_cols = [c for c in df_ref.columns if c in df_prod.columns and c != "TARGET"]
    
    if not common_cols:
        print("❌ Erreur : Aucune colonne commune trouvée.")
        return

    # Nettoyage des types (Conversion numérique forcée pour éviter les bugs)
    df_ref_clean = df_ref[common_cols].apply(pd.to_numeric, errors='coerce')
    df_prod_clean = df_prod[common_cols].apply(pd.to_numeric, errors='coerce')
    
    print(f"✅ Alignement terminé sur {len(common_cols)} colonnes.")

    # 4. Configuration & Calcul
    print("⏳ Calcul du drift en cours...")
    
    # Utilisation du Preset (Disponible en 0.6.0)
    report = Report(metrics=[
        DataDriftPreset(), 
    ])

    report.run(reference_data=df_ref_clean, current_data=df_prod_clean)

    # 5. Sauvegarde
    report.save_html(HTML_REPORT_PATH)
    print(f"📄 Rapport HTML généré : {HTML_REPORT_PATH}")
    
    report.save_json(JSON_REPORT_PATH)
    print(f"💾 JSON généré : {JSON_REPORT_PATH}")

    # Résumé console (Structure JSON de la 0.6.0)
    try:
        results = report.as_dict()
        # En 0.6.0, le chemin est standard
        metrics_result = results['metrics'][0]['result']
        drift_share = metrics_result['share_of_drifted_columns']
        n_drifted = metrics_result['number_of_drifted_columns']
        is_drift = metrics_result['dataset_drift']
        
        status = "🔴 ALERTE" if is_drift else "🟢 OK"
        print(f"\n📊 RÉSULTAT : {status} - {n_drifted} colonnes en drift ({drift_share*100:.1f}%).")
    except Exception as e:
        print(f"\n✅ Analyse terminée (détails dans le HTML).")

if __name__ == "__main__":
    run_analysis()