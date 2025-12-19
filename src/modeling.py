"""Module de modélisation et suivi d'expérimentations avec MLflow.

Ce module centralise:
- l'évaluation des modèles (métriques adaptées au déséquilibre),
- le logging d'artefacts (matrice de confusion, importance des features),
- une fonction de coût métier (FN > FP),
- une boucle de validation croisée avec tracking MLflow,
- des options pour StandardScaler et SMOTE.

Fonctions principales:
- evaluate_model
- log_light_confusion_matrix
- log_feature_importance
- custom_business_cost_scorer
- train_and_track_model_with_cv
"""
import os
import pandas as pd
import numpy as np
import mlflow
from mlflow.models import infer_signature
import seaborn as sns
import matplotlib.pyplot as plt
 
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)


def evaluate_model(y_test, y_pred, y_pred_proba):
    """Calcule un ensemble de métriques de classification (adaptées au déséquilibre).

    Cette fonction retourne des métriques standard et utiles sur données déséquilibrées:
    - accuracy
    - roc_auc (AUC-ROC sur les probabilités)
    - precision, recall, f1 (sur y_pred)
    - pr_auc (Average Precision, aire sous la courbe Precision-Recall)

    Args:
        y_test (array-like): Vraies étiquettes de la classe cible.
        y_pred (array-like): Prédictions binaires (0/1), généralement après seuillage.
        y_pred_proba (array-like): Probabilités de la classe positive (colonnes [:, 1] si binaire).

    Returns:
        Dict[str, float]: Dictionnaire {nom_métrique: valeur}.
    """
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_test, y_pred_proba),
    }
    return metrics


def log_light_confusion_matrix(y_test, y_pred, title="Matrice de confusion", fname="cm.png"):
    """Génère et logge une matrice de confusion légère en artefact MLflow.

    Produit un graphique simple (heatmap) de la matrice de confusion puis:
    - sauvegarde l'image localement (fname),
    - logge le fichier dans l'expérience MLflow courante.

    Args:
        y_test (array-like): Vraies étiquettes.
        y_pred (array-like): Prédictions binaires (0/1).
        title (str): Titre du graphique.
        fname (str): Nom de fichier de sortie pour l'image.
    """
    plt.figure(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues', 
                cbar=False,
                xticklabels=['Remboursé', 'Défaut'], 
                yticklabels=['Remboursé', 'Défaut']
    )
    plt.title(title, fontsize=10)
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.tight_layout()
    plt.savefig(fname, dpi=100, bbox_inches='tight')
    try:
        mlflow.log_artifact(fname)
    except Exception:
        pass
    plt.close()
    try:
        os.remove(fname)
    except Exception:
        pass


def log_feature_importance(model, feature_names, top_n=30, title="Feature importance", fname="feature_importance.png"):
    """Log l’importance des features (si disponible) en tant qu’artefact MLflow.

    - Compatible avec modèles à base d’arbres (RandomForest, XGBoost/LightGBM).
    - Compatible avec un Pipeline sklearn: tente de récupérer `named_steps['model']`.
    - Trie par importance décroissante et affiche les top_n.

    Args:
        model: Estimateur sklearn (ou Pipeline) déjà entraîné.
        feature_names (List[str]): Noms des variables en entrée.
        top_n (int): Nombre de variables à afficher.
        title (str): Titre du graphique.
        fname (str): Nom du fichier de sortie.

    Notes:
        Si `feature_importances_` n’est pas disponible, la fonction ne logge rien.
    """
    try:
        importances = None
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "named_steps"):
            est = model.named_steps.get("model", None)
            if est is not None and hasattr(est, "feature_importances_"):
                importances = est.feature_importances_
        if importances is None:
            print("Pas de feature_importances_ disponible.")
            return
        imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False).head(top_n)
        plt.figure(figsize=(8, max(4, top_n * 0.25)))
        sns.barplot(x="importance", y="feature", data=imp_df, orient="h")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(fname)
        mlflow.log_artifact(fname)
        plt.close()
        print(f"Feature importance loggée: {fname}")
    except Exception as e:
        print(f"Impossible de logger la feature importance: {e}")


try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    _IMBLEARN_AVAILABLE = True
except Exception:
    _IMBLEARN_AVAILABLE = False


def train_and_track_model_with_cv(model_name, model_class, params, X_data, y_data, n_splits=5, apply_scaler=True, use_smote=False):
    """Entraîne un modèle avec StratifiedKFold et logge tous les résultats dans MLflow.

    Pipeline général:
    1) Split StratifiedKFold (préserve la proportion de classes).
    2) Construction dynamique du Pipeline:
       - scaler optionnel (StandardScaler) selon `apply_scaler`,
       - SMOTE optionnel (si imblearn dispo) selon `use_smote`,
       - modèle fourni par `model_class(**params)`.
    3) Entraînement, prédictions, calcul de métriques avec [`modeling.evaluate_model`](src/modeling.py).
    4) Logging MLflow:
       - paramètres (run parent),
       - métriques par fold (runs enfants),
       - métriques moyennes/écarts,
       - artefacts (matrice de confusion, feature importance si dispo),
       - enregistrement du Pipeline dans le Model Registry.

    Args:
        model_name (str): Nom lisible du modèle (utilisé pour nommer les runs).
        model_class (type): Classe sklearn compatible (ex: LogisticRegression, LGBMClassifier).
        params (Dict[str, Any]): Hyperparamètres à passer au constructeur du modèle.
        X_data (pd.DataFrame): Données d’entraînement (features).
        y_data (pd.Series | array-like): Cible binaire.
        n_splits (int): Nombre de folds StratifiedKFold.
        apply_scaler (bool): Ajout d’un StandardScaler dans le Pipeline.
        use_smote (bool): Ajout de SMOTE si disponible (via imblearn).

    Notes:
        - Les métriques retournées par [`modeling.evaluate_model`](src/modeling.py) incluent `pr_auc`
          (utile pour classes déséquilibrées).
        - Les artefacts (images .png) sont sauvegardés localement puis loggés dans MLflow.
    """
    print(f"\n--- Démarrage de l'expérimentation pour {model_name} ---")

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_metrics = []

    with mlflow.start_run(run_name=f"{model_name}_CV_Run_Scaler_{apply_scaler}") as parent_run:
        mlflow.log_params(params) # Log des paramètres du modèle une seule fois pour le run parent
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("n_splits_cv", n_splits)
        mlflow.set_tag("apply_scaler", str(apply_scaler)) # Log si le scaler a été appliqué
        mlflow.set_tag("data_split", "train_full") # Indique que c'est sur le dataset complet avant test_final

        for fold, (train_index, val_index) in enumerate(kf.split(X_data, y_data)):
            with mlflow.start_run(nested=True, run_name=f"{model_name}_Fold_{fold+1}") as child_run:
                X_train, X_val = X_data.iloc[train_index], X_data.iloc[val_index]
                y_train, y_val = y_data.iloc[train_index], y_data.iloc[val_index]

                # Création du pipeline
                if apply_scaler and use_smote and _IMBLEARN_AVAILABLE:
                    pipeline = ImbPipeline([
                        ('smote', SMOTE(random_state=42)),
                        ('scaler', StandardScaler()),
                        ('model', model_class(**params, random_state=42))
                    ])
                elif use_smote and _IMBLEARN_AVAILABLE:
                    pipeline = ImbPipeline([
                        ('smote', SMOTE(random_state=42)),
                        ('model', model_class(**params, random_state=42))
                    ])
                elif apply_scaler:
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('model', model_class(**params, random_state=42))
                    ])
                else:
                    pipeline = Pipeline([
                        ('model', model_class(**params, random_state=42))
                    ])

                pipeline.fit(X_train, y_train)

                y_pred = pipeline.predict(X_val)
                y_pred_proba = pipeline.predict_proba(X_val)[:, 1]

                metrics = evaluate_model(y_val, y_pred, y_pred_proba)
                fold_metrics.append(metrics)

                # Log des métriques pour chaque fold
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"fold_{fold+1}_{metric_name}", value)

                mlflow.set_tag("fold_number", fold + 1)
                mlflow.set_tag("status", "completed")
                mlflow.set_tag("use_smote", str(use_smote and _IMBLEARN_AVAILABLE))
                print(f"  {model_name} - Fold {fold+1} metrics logged. ROC AUC: {metrics['roc_auc']:.4f}")

        # Calcul et log des métriques moyennes sur tous les folds
        avg_metrics = {metric: np.mean([f[metric] for f in fold_metrics]) for metric in fold_metrics[0]}
        std_metrics = {metric: np.std([f[metric] for f in fold_metrics]) for metric in fold_metrics[0]}

        print(f"\n--- Métriques Moyennes pour {model_name} sur {n_splits} Folds ---")
        for metric_name, avg_value in avg_metrics.items():
            std_value = std_metrics[metric_name]
            mlflow.log_metric(f"avg_{metric_name}", avg_value)
            mlflow.log_metric(f"std_{metric_name}", std_value)
            print(f"  Avg {metric_name}: {avg_value:.4f} (+/- {std_value:.4f})")

        mlflow.log_dict(fold_metrics, "fold_metrics.json")

        input_example = None
        try:
            # Si X_data est un DataFrame
            input_example = X_data.iloc[:5]
        except Exception:
            # Sinon, tente une petite tranche numpy
            try:
                input_example = X_data[:5]
            except Exception:
                pass

        # Enregistrer le modèle (le pipeline entier)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name=f"{model_name}_final_pipeline_scaler_{apply_scaler}",
            registered_model_name=f"{model_name}_CreditScoring_CV_Scaler_{apply_scaler}",
            input_example=input_example,
            signature=infer_signature(X_data, pipeline.predict(X_data))
        )
        # NEW: log feature importance (si dispo)
        try:
            log_feature_importance(pipeline, feature_names=list(X_data.columns), title=f"{model_name} - Feature importance")
        except Exception:
            pass

        mlflow.set_tag("status", "final_cv_run_complete")

    print(f"--- Expérimentation pour {model_name} terminée et loggée dans MLflow. ---")
    return avg_metrics


def custom_business_cost_scorer(y_true, y_pred_proba, threshold=0.5, COST_FN=10000, COST_FP=1000):
    """Calcule un score de coût métier (FN plus coûteux que FP).

    Définition du coût:
    $$
    total\\_cost = FP \\cdot COST\\_{FP} + FN \\cdot COST\\_{FN}
    $$
    La fonction retourne le coût sous forme négative pour être compatible avec
    les API sklearn qui maximisent le score (plus le score est grand, mieux c’est).

    Args:
        y_true (array-like): Vraies étiquettes.
        y_pred_proba (array-like): Probabilités de la classe positive.
        threshold (float): Seuil de classification pour convertir les probabilités en 0/1.
        COST_FN (int | float): Coût d’un faux négatif.
        COST_FP (int | float): Coût d’un faux positif.

    Returns:
        float: Score négatif (i.e. -total_cost).
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calcul du coût total
    total_cost = (fp * COST_FP) + (fn * COST_FN)

    # Scikit-learn GridSearch maximise le score, donc nous retournons le négatif du coût
    # pour que GridSearchCV cherche à minimiser le coût (maximiser le -coût)
    return -total_cost
