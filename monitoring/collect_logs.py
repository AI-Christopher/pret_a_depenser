import requests
import os
import shutil
from requests.auth import HTTPBasicAuth

# --- CONFIGURATION ---
# Mettez l'URL de production ici quand vous déploierez
API_URL = "https://ai-christopher-pret-a-depenser-api.hf.space" 
DOWNLOAD_ENDPOINT = "/download_logs"

# Identifiants
USERNAME = "admin"
PASSWORD = "password123"

# Chemins
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
LOCAL_LOG_DIR = os.path.join(PROJECT_ROOT, "api", "production_logs")
LOCAL_LOG_FILE = os.path.join(LOCAL_LOG_DIR, "api_request_log.jsonl")
TEMP_LOG_FILE = os.path.join(LOCAL_LOG_DIR, "temp_downloaded_logs.jsonl")

def fetch_and_merge_logs():
    full_url = f"{API_URL}{DOWNLOAD_ENDPOINT}"
    print(f"🌍 Connexion à {full_url}...")

    try:
        # 1. Téléchargement dans un fichier TEMPORAIRE
        response = requests.get(
            full_url, 
            auth=HTTPBasicAuth(USERNAME, PASSWORD),
            stream=True
        )

        if response.status_code == 200:
            os.makedirs(LOCAL_LOG_DIR, exist_ok=True)
            
            # Sauvegarde temporaire
            with open(TEMP_LOG_FILE, 'wb') as f:
                response.raw.decode_content = True
                shutil.copyfileobj(response.raw, f)
            
            print("📥 Fichier téléchargé. Début de la fusion...")

            # 2. Logique de FUSION (Merge) intelligente
            # On charge d'abord les lignes existantes dans un "Set" (pour la rapidité)
            existing_lines = set()
            if os.path.exists(LOCAL_LOG_FILE):
                with open(LOCAL_LOG_FILE, 'r', encoding='utf-8') as f:
                    for line in f:
                        existing_lines.add(line.strip())
            
            new_lines_count = 0
            
            # On ouvre le fichier local en mode 'a' (Append/Ajout)
            with open(LOCAL_LOG_FILE, 'a', encoding='utf-8') as f_out:
                # On lit le fichier qu'on vient de télécharger
                with open(TEMP_LOG_FILE, 'r', encoding='utf-8') as f_in:
                    for line in f_in:
                        clean_line = line.strip()
                        # Si la ligne n'est pas vide ET n'existe pas déjà chez nous
                        if clean_line and clean_line not in existing_lines:
                            f_out.write(clean_line + '\n')
                            existing_lines.add(clean_line) # On l'ajoute au set pour éviter les doublons internes
                            new_lines_count += 1
            
            # 3. Nettoyage
            os.remove(TEMP_LOG_FILE)
            print(f"✅ Fusion terminée ! {new_lines_count} nouvelles lignes ajoutées.")
            print(f"📂 Total stocké dans : {LOCAL_LOG_FILE}")

        elif response.status_code == 404:
            print("⚠️  Aucun log sur le serveur pour l'instant.")
        elif response.status_code == 401:
            print("❌ Erreur d'authentification.")
        else:
            print(f"❌ Erreur HTTP : {response.status_code}")

    except Exception as e:
        print(f"❌ Erreur de connexion : {e}")

if __name__ == "__main__":
    fetch_and_merge_logs()