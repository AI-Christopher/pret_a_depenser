# Utilisation d'une image Python légère et récente
FROM python:3.12-slim

# Empêche Python de garder les logs en mémoire
ENV PYTHONUNBUFFERED=1

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

# 1. On crée le dossier de logs manuellement
RUN mkdir -p /app/api/production_logs

# 2. On donne la permission "777" (Tout le monde peut écrire) à ce dossier.
# Sans ça, l'utilisateur Hugging Face ne peut pas créer le fichier jsonl.
RUN chmod -R 777 /app/api/production_logs
    
# Exposition du port (Hugging Face Spaces attend le port 7860 par défaut)
EXPOSE 7860

# Commande de démarrage
# On change le port pour 7860 et l'host pour 0.0.0.0
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]