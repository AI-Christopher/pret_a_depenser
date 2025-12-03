import pandas as pd
import os
import re

def load_data(files_to_load:list)->dict:
    """
    Chargement des données à partir d'un fichier CSV.

    Parameters:
    files_to_load (list): Liste des fichiers CSV à charger.

    Returns:
    dict: Dictionnaire contenant les DataFrames chargés.
    """
    
    # Chemin vers le dossier contenant les fichiers de données
    data_path = "../datas/01_base/"

    # Dictionnaire pour stocker les DataFrames
    data = {}

    # Chargement des fichiers spécifiés
    for file in files_to_load:
        df_name = file.replace('.csv', '')
        file_path = os.path.join(data_path, file)

        if os.path.exists(file_path):
            data[df_name] = pd.read_csv(file_path, encoding='utf-8')
            print(f"Le fichier {file} a été chargé avec succès sous le nom '{df_name}'.")
        else:
            print(f"Le fichier {file} n'existe pas à l'emplacement spécifié: {file_path}")

    return data

def get_shape(df_dict:dict):
    """
    Obtient la forme (nombre de lignes et de colonnes) d'un DataFrame.

    Parameters:
    df_dict (dict): Dictionnaire contenant les DataFrames dont on veut connaître la forme.
    """
    for name, df in df_dict.items():
        print(f"DataFrame: {name}")
        print(f"Shape: {df.shape}")
        print("---------------------------------")

def get_head(df_dict:dict):
    """
    Obtient les premières lignes d'un DataFrame.

    Parameters:
    df_dict (dict): Dictionnaire contenant les DataFrames dont on veut connaître les premières lignes.

    Returns:
    None
    """
    for name, df in df_dict.items():
        print(f"DataFrame: {name}")
        print(df.head())
        print("---------------------------------")


def split_columns_by_uniques(df, threshold=10):
    """
    Retourne deux listes de colonnes 'object' selon le nombre de modalités uniques.
    - Colonnes avec < threshold modalités (list_onehot)
    - Colonnes avec >= threshold modalités (list_binaire)
    """
    uniques = df.select_dtypes(include="object").nunique()
    list_onehot = uniques[uniques < threshold].sort_values().index.tolist()
    list_binaire = uniques[uniques >= threshold].sort_values().index.tolist()
    print(f"Colonnes avec moins de {threshold} modalités : {list_onehot}")
    print(f"Colonnes avec {threshold} modalités ou plus : {list_binaire}")
    return list_onehot, list_binaire

def clear_cols_name(df):
    """
    Nettoie et renomme les colonnes d'un DataFrame en remplaçant les caractères non valides.
    """
    df = df.copy()
    def clean(s):
        s = re.sub(r'[^0-9A-Za-z_]', '_', str(s))
        s = re.sub(r'_+', '_', s).strip('_')
        return s or 'col'
    bad = [c for c in df.columns if re.search(r'[^0-9A-Za-z_]', str(c))]
    mapper = {}
    used = set(df.columns)
    for c in bad:
        new = clean(c)
        # assurer l’unicité
        if new != c:
            base, i = new, 1
            while new in used:
                new = f"{base}_{i}"
                i += 1
            mapper[c] = new
            used.add(new)
    return df.rename(columns=mapper), mapper