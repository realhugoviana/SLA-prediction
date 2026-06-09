import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Drops questions 10 and ALSFRS scores columns (not ALSFRS-R score)
# Input: Pandas Dataframe
# Output: Pandas Dataframe
def drop_unused_cols(df):
    cols_to_drop = df.columns[
        df.columns.str.contains(
            r'ALS_ALSFRS_Total|ALS_Q10|Test_Unit'
        )
    ]

    df = df.drop(columns=cols_to_drop)
    df = df.drop("subject_id", axis=1, errors='ignore')

    return df

# Drops stats
# Input: Pandas Dataframe
# Output: Pandas Dataframe
def drop_baseline(df):
    cols_to_keep = df.columns[
        df.columns.str.contains(
            r'Central'
        )
    ]

    df = df[cols_to_keep]

    return df

# Drops features with more than a certain percentage of missing values and keeps only features with the highest r2/eta2 with the target variable
# Input: Pandas Dataframe, Threshold percentage of missing values
# Output: Pandas Dataframe
def drop_features_r2_eta2(df, thresh):
    cols_r2_eta2 = r2_eta2(drop_features_thresh(df, df, thresh=thresh))["col"]

    df_dropped = drop_features_by_features(df, cols_r2_eta2.index[:100])
    
    return df_dropped

def encode_simple_categorical(X):
        X_encoded = X.copy()

        # Définitions des mappages standards
        bool_map = {
            "True": 1, "False": 0,
            "Yes": 1, "No": 0,
        }
        direction_map = {
            "Left": -1, "Equal": 0, "Right": 1
        }
        sexe_map = {
            "Male": 1, "Female": 0
        }

        # Mapping simple supplémentaire
        simple_maps = {
            "TRE_Study_Arm": {"Active": 1, "Placebo": 0},
            "ELE_el_escorial": {
                "Definite": 3,
                "Probable Laboratory Supported": 2,
                "Probable": 1,
                "Possible": 0
            },
            "DEM_Ethnicity": {"Hispanic or Latino": 1, "Non-Hispanic or Latino": 0, "Unknown": np.nan}
        }

        # Log des colonnes traitées
        encoded_cols = []
        skipped_cols = []

        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype == 'string':
                unique_vals = X[col].dropna().unique()

                # Si toutes les valeurs appartiennent au mapping booléen
                if all(str(v).strip() in bool_map for v in unique_vals):
                    X_encoded[col] = X[col].map(lambda x: bool_map.get(str(x).strip(), np.nan))
                    encoded_cols.append((col, "bool"))
                
                # Si toutes les valeurs appartiennent au mapping directionnel
                elif all(str(v).strip() in direction_map for v in unique_vals):
                    X_encoded[col] = X[col].map(lambda x: direction_map.get(str(x).strip(), np.nan))
                    encoded_cols.append((col, "direction"))
                
                # Si toutes les valeurs appartiennent au mapping sexe
                elif all(str(v).strip() in sexe_map for v in unique_vals):
                    X_encoded[col] = X[col].map(lambda x: sexe_map.get(str(x).strip(), np.nan))
                    encoded_cols.append((col, "sexe"))
                
                # Si la colonne a un mapping simple défini
                elif col in simple_maps:
                    mapping = simple_maps[col]
                    X_encoded[col] = X[col].map(lambda x: mapping.get(str(x).strip(), np.nan))
                    encoded_cols.append((col, "simple_map"))
                
                elif len(unique_vals) > 2:
                    X_encoded[col] = X[col]
                    X_encoded = pd.get_dummies(X_encoded, columns=[col])
                # Sinon, on ignore pour l’instant
                else:
                    skipped_cols.append(col)
                    X_encoded[col] = X[col]
            else:
                # Garder la colonne telle quelle
                X_encoded[col] = X[col]

        return X_encoded

def drop_features_thresh(df, df_ref, thresh):
    cols_als = df_ref.columns[df_ref.columns.str.startswith("ALS_")].tolist()
    cols_als.append("Target")
    cols_other = df_ref.columns.difference(cols_als)

    df_ref_temp = df_ref[cols_other].dropna(thresh=(thresh / 100) * len(df_ref), axis=1)
    
    cols_to_keep = df_ref_temp.columns.union(cols_als)

    cols_to_keep_df = cols_to_keep.intersection(df.columns)

    df_temp = df[cols_to_keep_df].copy()

    return df_temp

def drop_features_by_features(df, cols_to_keep):
    cols_als = df.columns[df.columns.str.startswith("ALS_")].tolist()
    cols_als.append("Target")

    cols_to_keep = cols_to_keep.union(cols_als)

    cols_to_keep_df = cols_to_keep.intersection(df.columns)

    df_temp = df[cols_to_keep_df].copy()

    return df_temp

# Computes r2 for numeric features and eta2 for categorical features, returns a dataframe with the results sorted by descending r2/eta2
# Input: Pandas Dataframe
# Output: Pandas Dataframe with columns "col" and "r2/eta2"
def r2_eta2(df):
    
    # toutes sauf ALS et target
    cols = [c for c in df.columns if not c.startswith("ALS") and c != "Target"]

    target = "Target"
    
    # dataframe pour stocker les résultats
    results = pd.DataFrame({"col": [], 
                    "r2/eta2": []})
    
    for col in cols:
        # Drop NA sur les deux colonnes
        temp_df = df[[col, target]].dropna()
            
        if temp_df.empty:
            val = -1
        else:
            if pd.api.types.is_numeric_dtype(temp_df[col]):
                # Variable continue → r2
                X = temp_df[[col]].values
                y = temp_df[target].values
                model = LinearRegression().fit(X, y)
                val = model.score(X, y)  # r2
            else:
                # Variable catégorielle → eta²
                groups = [temp_df[target][temp_df[col] == cat] for cat in temp_df[col].unique()]
                # eta² = SSB / SST
                ss_between = sum([len(g) * (g.mean() - temp_df[target].mean())**2 for g in groups])
                ss_total = ((temp_df[target] - temp_df[target].mean())**2).sum()
                val = ss_between / ss_total if ss_total != 0 else np.nan

        results = pd.concat([results, pd.DataFrame({"col": [col], "r2/eta2": [val]})], ignore_index=True)
    

    #supprimer les cols avec au moins une valeur -1 
    results = results[(results != -1).all(axis=1)]

    # trie les colonnes par ordre décroissant
    results = results.sort_values(by='r2/eta2', ascending=False)

    return results

def calculate_fill_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le taux de remplissage (ou taux de complétude) pour chaque colonne 
    d'un DataFrame donné. Le taux est calculé comme :
    (Nombre de valeurs non-nulles) / (Nombre total de lignes).

    Args:
        df (pd.DataFrame): Le DataFrame d'entrée.

    Returns:
        pd.DataFrame: Un nouveau DataFrame contenant le taux de remplissage 
                       pour chaque colonne, avec les colonnes originales comme index.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("L'argument doit être un DataFrame pandas.")
    
    # 1. Compter le nombre de valeurs non-nulles pour chaque colonne
    # df.count() retourne une Series où les valeurs sont les comptes non-nuls
    non_null_counts = df.count()

    # 2. Obtenir le nombre total de lignes dans le DataFrame
    total_rows = len(df)
    
    if total_rows == 0:
        print("Attention : Le DataFrame est vide.")
        return pd.DataFrame()


    # 3. Calculer le taux de remplissage (division des comptes par le total des lignes)
    fill_rates = non_null_counts / total_rows

    # 4. Convertir la Series résultante en un DataFrame pour une sortie structurée et lisible
    result_df = pd.DataFrame({
        'Feature': fill_rates.index, # Les noms des colonnes deviennent une colonne
        'Fill_Rate': fill_rates,     # Le taux est dans la deuxième colonne
        'num_non_null': non_null_counts, # Nombre de valeurs non-nulles
        'total_rows': total_rows    # Nombre total de lignes (pour référence)
    })
    
    return result_df

# Returns a list of the intervals contained in the dataset
# Input: Pandas Dataframe
# Output: List of strings
def get_intervals(df):
    extracted_series = df.columns.to_series().str.extract(r'(\d{1,4}_\d{2,4})')
    
    single_series = extracted_series.stack() 
    
    unique_intervals_array = single_series.dropna().unique()
    
    interval_list = list(unique_intervals_array) 

    def sort_key(interval):
        first_number_str = interval.split('_')[0]
        return int(first_number_str)

    sorted_intervals = sorted(interval_list, key=sort_key)

    print(f"Intervals found: {sorted_intervals}")
    return pd.Series(sorted_intervals)

# Creates a dataset with the specified target interval and the specified source intervals
# Input: Pandas Dataframe, List of intervals, Index of the first source interval, Index of the target interval
# Output: Pandas Dataframe
def create_dataset_from_intervals(df, intervals, start_i):

    intervals_to_keep = intervals[start_i:]

    cols_to_keep = df.columns[
        df.columns.str.contains(r'Target') |
        df.columns.str.contains('|'.join(intervals_to_keep)) |
        ~(df.columns.str.contains(r'\d{1,4}_\d{2,4}', na=False))
    ]

    df = df[cols_to_keep]

    df = df.dropna()

    return df

# Separate dataset with all intervals into multiple datasets of all combinaisons of consecutive intervalls and saves them to csv files outputs a dataframe of the size of the generated datasets
# Input: Pandas Dataframe, String prefix for the generated csv files
# Output: Pandas Dataframe
def separate_datasets_by_intervals(df, file_prefix='datasets/ALSFRS_R_FIXED/ALSFRS_R_FIXED'):
    intervals = get_intervals(df)

    datasets_size = pd.DataFrame({"dataset": [], 
                    "size": [],
                    "nb_cols": [],
                    "nb_cols_non_als": []})
    
    target_i = len(intervals)

    for start_i in range(target_i):
        df_temp = create_dataset_from_intervals(df, intervals, start_i)
        csv_name = f'{file_prefix}_{start_i*3}_{target_i*3}M.csv'
        df_temp.to_csv(csv_name, index=False)
        print(f"Dataset {csv_name} created.")

        calculate_fill_rates(df_temp).to_csv(f'{file_prefix}_fill_rate_{start_i*3}_{target_i*3}M.csv', index=False)

        datasets_size = pd.concat([
            datasets_size,
            pd.DataFrame({
                "dataset": [csv_name],
                "size": [len(df_temp)],
                "nb_cols": [len(df_temp.columns)],
                "nb_cols_non_als": [len(df_temp.columns) - sum(
                    1 for c in df_temp.columns
                    if c.startswith('ALS') or c == 'Target'
                )]
            })
        ], ignore_index=True)

    return datasets_size

# Main function to generate datasets from the original dataset and save them to csv files, also outputs a dataframe of the size of the generated datasets
# Input: Pandas Dataframe
# Output: Pandas Dataframe
def generate_datasets(df, file_prefix='datasets/ALSFRS_R_FIXED/ALSFRS_R_FIXED', drop_feat=True, thresh=80, baseline=False):
    if baseline:
        df = drop_baseline(df)

    intervals = get_intervals(df)
    target_i = len(intervals) - 1
    df.rename(columns={f'ALS_ALSFRS_R_Total_{intervals[target_i]}_Central': 'Target'}, inplace=True)

    df = drop_unused_cols(df)

    df = encode_simple_categorical(df)

    if drop_feat:
        df = drop_features_thresh(df, df, thresh=thresh)

    calculate_fill_rates(df).to_csv(f'{file_prefix}_fill_rate_0_{target_i*3}M.csv', index=False)

    datasets_size = separate_datasets_by_intervals(df, file_prefix)

    return datasets_size


# Preprocesses combined dataset and saves to csv file
# Input: Pandas Dataframe, String path to the output csv file
# Output: Pandas Dataframe
def preprocess_sliding_windows_dataset(df, file_name, drop_feat=True, thresh=80, baseline=False):
    if baseline:
        df = drop_baseline(df)

    df = drop_unused_cols(df)
    df.rename(columns={'ALS_ALSFRS_R_Total_Qi+1_Central': 'Target'}, inplace=True)

    df = encode_simple_categorical(df)

    if drop_feat:
        df = drop_features_thresh(df, df, thresh=thresh)

    df = df.dropna()

    df.to_csv(file_name, index=False)
    print(f"Dataset {file_name}.")

    return df