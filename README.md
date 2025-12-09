# Documentation du pipeline de scoring

## Structure du projet
- Données: `datas/`
- Notebooks: `notebooks/` — exploration, tracking, comparaison modèles, HPO + optimisation de seuil
- Code source: `src/` — utilitaires de modélisation et fonctions métier
- Suivi d'expériences: `mlruns/` et `mlartifacts/`

## Modules clés
- Modélisation: [`src/modeling.py`](src/modeling.py)
  - [`modeling.evaluate_model`](src/modeling.py): calcule les métriques (incl. PR AUC)
  - [`modeling.log_light_confusion_matrix`](src/modeling.py): log de matrice de confusion
  - [`modeling.log_feature_importance`](src/modeling.py): importances des features (arbres)
  - [`modeling.custom_business_cost_scorer`](src/modeling.py): coût métier (FN > FP)
  - [`modeling.train_and_track_model_with_cv`](src/modeling.py): CV + tracking MLflow

## Lancer MLflow UI
Dans un terminal:
```sh
mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri ./mlruns
```
Puis ouvrez: http://localhost:5001

## Métriques et coût métier
- Jeu déséquilibré: suivre ROC AUC, PR AUC, Recall, Precision, F1.
- Coût métier:
  $total\_cost = FP \cdot COST\_{FP} + FN \cdot COST\_{FN}$
  Le score optimisé est $-total\_cost$ (convention sklearn: maximisation).

## Processus global
1) Préparation & feature engineering — voir [notebooks/01_exploration_donnees.ipynb](notebooks/01_exploration_donnees.ipynb)  
2) Tracking MLflow — voir [notebooks/02_MLflow_Basic_Tracking.ipynb](notebooks/02_MLflow_Basic_Tracking.ipynb)  
3) Comparaison multi-modèles — voir [notebooks/03_Model_Comparison_and_CV.ipynb](notebooks/03_Model_Comparison_and_CV.ipynb)  
4) HPO + optimisation du seuil — voir [notebooks/04_Hyperparameter_Tuning_and_Threshold_Optimization.ipynb](notebooks/04_Hyperparameter_Tuning_and_Threshold_Optimization.ipynb)

## Bonnes pratiques
- Fixer `random_state` pour la reproductibilité.
- Gérer le déséquilibre (class_weight, is_unbalance, SMOTE).
- Logger systématiquement paramètres/métriques/artefacts (MLflow).
- Documenter les décisions métier (seuil, coûts).