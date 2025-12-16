import pandas as pd
import os
import json

# --- CONFIGURATION ---
# 1. Fichier complet d'entrainement nettoyé
ORIGINAL_DATA_PATH = "../datas/02_preprocess/datas.csv" 

# 2. La liste des features utilisées par le modèle
FEATURES_JSON_PATH = "../api/features_list.json"

# 3. Sauvegarder de l'échantillon léger dans ce dossier
OUTPUT_PATH = "reference_sample.csv"

# Taille de l'échantillon (10 000 à 50 000 est suffisant pour le drift)
SAMPLE_SIZE = 10000 

def prepare_reference():
    print("🚀 Démarrage de la préparation de la référence...")

    # 1. Vérifications des fichiers
    if not os.path.exists(ORIGINAL_DATA_PATH):
        print(f"❌ ERREUR : Le fichier original est introuvable ici : {ORIGINAL_DATA_PATH}")
        print("-> Vérifiez le chemin dans le script.")
        return

    if not os.path.exists(FEATURES_JSON_PATH):
        print(f"❌ ERREUR : La liste des features est introuvable ici : {FEATURES_JSON_PATH}")
        return

    # 2. Chargement des noms de colonnes (Pour ne garder que l'utile)
    with open(FEATURES_JSON_PATH, "r") as f:
        model_features = json.load(f)
    print(f"✅ Liste des features chargée ({len(model_features)} colonnes).")

    # 3. Chargement des données
    print("⏳ Chargement du dataset original...")
    df = pd.read_csv(ORIGINAL_DATA_PATH, index_col=0)
    
    # Si le fichier contient la Cible (TARGET), on la garde, c'est utile pour l'analyse !
    # Mais on s'assure d'avoir toutes les features du modèle.
    cols_to_keep = model_features.copy()
    if 'TARGET' in df.columns:
        cols_to_keep.append('TARGET')
    
    # Filtrage des colonnes (On vire ce qui ne sert pas au modèle)
    # On utilise intersection pour ne pas planter si une colonne manque
    final_cols = [c for c in cols_to_keep if c in df.columns]
    df = df[final_cols]
    
    print(f"✅ Dataset chargé : {df.shape}")

    # 4. Échantillonnage (Sampling)
    if len(df) > SAMPLE_SIZE:
        print(f"✂️  Réduction du dataset à {SAMPLE_SIZE} lignes aléatoires...")
        # random_state=42 assure qu'on aura toujours le même échantillon (reproductible)
        df_sample = df.sample(n=SAMPLE_SIZE, random_state=42)
    else:
        print("⚠️ Le dataset est petit, on garde tout.")
        df_sample = df

    # 5. Sauvegarde
    df_sample.to_csv(OUTPUT_PATH, index=False)
    print(f"🎉 Succès ! Fichier de référence créé : {os.path.abspath(OUTPUT_PATH)}")
    print(f"   Taille : {len(df_sample)} lignes x {len(df_sample.columns)} colonnes")

if __name__ == "__main__":
    prepare_reference()