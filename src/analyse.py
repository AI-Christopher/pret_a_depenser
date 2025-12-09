import pandas as pd

def missing_values_table(df):
    """
    cette fonction permet d'afficher le nombre et le pourcentage de valeurs manquantes dans un DataFrame.
    Parameters:
    df (DataFrame): Le DataFrame à analyser
    Returns:
    DataFrame: Un DataFrame contenant le nombre et le pourcentage de valeurs manquantes pour chaque colonne.
    """

    # Total de valeurs manquantes
    mis_val = df.isnull().sum()

    # Pourcentage de valeurs manquantes
    mis_val_percent = 100 * df.isnull().sum() / len(df)
        
    # Création d'un tableau avec les deux informations
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
    # Renommer les colonnes
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Valeurs Manquantes', 1 : '% de Valeurs Totales'})

    # Trier le tableau par pourcentage de valeurs manquantes décroissant
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% de Valeurs Totales', ascending=False).round(1)

    # Afficher un résumé des informations
    print ("Le DataFrame sélectionné a " + str(df.shape[1]) + " colonnes.\n"      
        "Il y a " + str(mis_val_table_ren_columns.shape[0]) +
        " colonnes qui ont des valeurs manquantes.")

    # Retourner le tableau des valeurs manquantes
    return mis_val_table_ren_columns

def get_unique_values(df_dict:dict, columns_id:str):
    """
    cette fonction permet d'afficher le nombre et le pourcentage de valeurs uniques dans un DataFrame.
    Parameters:
    df (DataFrame): Le DataFrame à analyser
    Returns:
    DataFrame: Un DataFrame contenant le nombre et le pourcentage de valeurs uniques pour chaque colonne.
    """
    for name, df in df_dict.items():
        print(f"DataFrame: {name}")
        print(f"Vérification de l'unicité des valeurs dans la colonne '{columns_id}':")
        if columns_id in df.columns:
            #is_unique = df[columns_id].is_unique
            #if is_unique:
            print(f"Nb valeur unique: {df[columns_id].nunique()} sur {len(df)} lignes.")
            #else:
            #    print(f"La colonne '{columns_id}' ne contient pas des valeurs uniques.")
        else:
            print(f"La colonne '{columns_id}' n'existe pas dans le DataFrame '{name}'.")
        print("---------------------------------")

def get_count_unique_values(df: pd.DataFrame):
    """
    Affiche le nombre de valeurs uniques pour chaque colonne de type 'object', trié décroissant.
    """
    uniques = {col: df[col].nunique() for col in df.select_dtypes(include=['object']).columns}
    for col, count in sorted(uniques.items(), key=lambda x: x[1], reverse=True):
        print(f"{col} : {count}")

