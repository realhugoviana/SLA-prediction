import pandas as pd
import os

tables = ['FVC', 'HAN', 'LAB', 'MUS', 'SVC', 'VIT']

def extraire_prefixes_colonnes(df: pd.DataFrame) -> set:
    """
    Analyse toutes les colonnes d'un DataFrame, extrait la substring
    avant le premier underscore ('_'), et retourne l'ensemble des
    préfixes uniques trouvés.

    Args:
        df (pd.DataFrame): Le DataFrame à analyser.

    Returns:
        set: Un ensemble (set) contenant tous les préfixes de colonnes uniques.
    """
    
    # 1. Récupérer la liste des noms de colonnes
    noms_colonnes = df.columns
    
    # Utilisation d'une compréhension de liste et d'un set pour l'efficacité
    prefixes = {col.split('_')[0] for col in noms_colonnes}
    
    return prefixes

def separate_datasets_by_tables(df, output_prefix, file_name):
    for table in tables:
        cols_to_keep = df.columns[
            df.columns.str.startswith(table) |
            df.columns.str.startswith('ALS') |
            df.columns.str.contains('Target')
        ]

        df_table = df[cols_to_keep].copy()

        os.makedirs(os.path.dirname(f'{output_prefix}/{table}/'), exist_ok=True)

        df_table.to_csv(f'{output_prefix}/{table}/{file_name}', index=False)

    cols_to_keep = df.columns[
        df.columns.str.startswith('ALS') |
        df.columns.str.contains('Target') |
        ~(df.columns.str.contains(r'\d{1,4}_\d{2,4}', na=False))
    ]

    df_no_delta = df[cols_to_keep].copy()

    os.makedirs(os.path.dirname(f'{output_prefix}/no_delta/'), exist_ok=True)

    df_no_delta.to_csv(f'{output_prefix}/no_delta/{file_name}', index=False)

def separate_datasets_no_delta_SVC(df, output_prefix, file_name):

    cols_to_keep = df.columns[
        df.columns.str.startswith('ALS') |
        df.columns.str.contains('Target') |
        ~(df.columns.str.contains(r'\d{1,4}_\d{2,4}', na=False)) |
        df.columns.str.startswith('SVC')
    ]

    df_no_delta_SVC = df[cols_to_keep].copy()

    os.makedirs(os.path.dirname(f'{output_prefix}/no_delta_SVC/'), exist_ok=True)

    df_no_delta_SVC.to_csv(f'{output_prefix}/no_delta_SVC/{file_name}', index=False)
