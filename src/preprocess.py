import pandas as pd
from category_encoders.binary import BinaryEncoder
from sklearn.preprocessing import LabelEncoder

def binary_encoding(df: pd.DataFrame, list_bin: list) -> pd.DataFrame:
    """Applique un encodage binaire aux colonnes spécifiées dans list_bin.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données à encoder.
        list_bin (list): Liste des noms de colonnes à encoder.

    Returns:
        pd.DataFrame: Le DataFrame avec les colonnes encodées.
    """
    be = BinaryEncoder(cols=list_bin, return_df=True)
    df[list_bin] = df[list_bin].fillna("missing")
    df_encoded = be.fit_transform(df[list_bin])
    df = df.drop(columns=list_bin)
    df = pd.concat([df, df_encoded], axis=1)
    return df


def one_hot_encoding(df: pd.DataFrame, list_onehot: list, nan_as_cat: bool) -> pd.DataFrame:
    """Applique un encodage one-hot aux colonnes spécifiées dans list_onehot.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données à encoder.
        list_onehot (list): Liste des noms de colonnes à encoder.

    Returns:
        pd.DataFrame: Le DataFrame avec les colonnes encodées.
    """
    df = pd.get_dummies(df, columns=list_onehot, drop_first=True, dummy_na=nan_as_cat)
    return df


def label_encoding(df: pd.DataFrame, list_bin: list) -> pd.DataFrame:
    """Applique un encodage par label aux colonnes spécifiées dans list_bin.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données à encoder.
        list_bin (list): Liste des noms de colonnes à encoder.

    Returns:
        pd.DataFrame: Le DataFrame avec les colonnes encodées.
    """
    le = LabelEncoder()
    for bin_feature in list_bin:
        df[bin_feature] = le.fit_transform(df[bin_feature])
    return df