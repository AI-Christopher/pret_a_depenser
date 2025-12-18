import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Chemins
DATA_PATH = "datas/02_preprocess/datas.csv" # Vérifie ce chemin !
MODEL_DIR = "api/model_files"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

print("🔄 Chargement des données...")
df = pd.read_csv(DATA_PATH)

# Préparation simple (Adapte si ton training était plus complexe)
X = df.drop(columns=['TARGET'])
y = df['TARGET']

# Gestion des caractères spéciaux JSON pour LightGBM
import re
X = X.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("⚙️ Entraînement du modèle (Version Scikit-Learn actuelle)...")
# Pipeline simplifiée basée sur ton projet
pipeline = ImbPipeline(steps=[
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', LGBMClassifier(random_state=42, verbose=-1))
])

pipeline.fit(X_train, y_train)

print(f"💾 Sauvegarde du modèle compatible dans {MODEL_PATH}...")
os.makedirs(MODEL_DIR, exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Terminé ! Ton modèle est maintenant synchro avec ton environnement.")