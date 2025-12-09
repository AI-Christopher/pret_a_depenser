# Utilisation d'une image Python légère et récente
FROM python:3.12-slim

# Définition du répertoire de travail dans le conteneur
WORKDIR /app

# Installation des dépendances système nécessaires (libgomp pour LightGBM/XGBoost)
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Copie des fichiers de dépendances
# Note: On générera un requirements.txt via uv pour la compatibilité Docker standard
COPY requirements.txt .

# Installation des paquets Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code de l'API et des fichiers nécessaires
COPY api/ ./api/

# Copie du dossier src si votre modèle en dépend (custom transformers etc.)
# COPY src/ ./src/ 
# (Décommentez la ligne ci-dessus si votre pickle a besoin de classes définies dans src)

# Exposition du port (Hugging Face Spaces attend le port 7860 par défaut)
EXPOSE 7860

# Commande de démarrage
# On change le port pour 7860 et l'host pour 0.0.0.0
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]