import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats

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

        # print("\n✅ Colonnes encodées simplement :")
        # for col, typ in encoded_cols:
        #     print(f"  - {col} ({typ})")

        # print("\n⚠️ Colonnes textuelles non traitées :")
        # for col in skipped_cols:
        #     print(f"  - {col} (exemples : {X[col].dropna().unique()[:5]})")

        return X_encoded

def drop_features_thresh(df, df_ref, thresh):
    cols_als = df_ref.columns[df_ref.columns.str.startswith("ALS_")].tolist()
    cols_als.append("Target")
    cols_other = df_ref.columns.difference(cols_als)

    df_ref_temp = df_ref[cols_other].dropna(thresh=(thresh / 100) * len(df_ref), axis=1)
    
    cols_to_keep = df_ref_temp.columns.union(cols_als)

    cols_to_keep_df = cols_to_keep.intersection(df.columns)

    cols_to_drop = df.columns.difference(cols_to_keep_df)

    # for col in cols_to_drop:
    #     print(f"Colonne {col} supprimée (taux de valeurs manquantes : {df_ref[col].isna().mean() * 100:.2f}%)")

    df_temp = df[cols_to_keep_df].copy()

    # als_cols = [c for c in df_temp.columns if c.startswith("ALS")]
    # df_temp = df_temp.dropna(subset=als_cols)

    return df_temp

def drop_features_by_features(df, cols_to_keep):
    cols_als = df.columns[df.columns.str.startswith("ALS_")].tolist()
    cols_als.append("Target")

    cols_to_keep = cols_to_keep.union(cols_als)

    cols_to_keep_df = cols_to_keep.intersection(df.columns)

    cols_to_drop = df.columns.difference(cols_to_keep_df)

    # for col in cols_to_drop:
    #     print(f"Colonne {col} supprimée")

    df_temp = df[cols_to_keep_df].copy()

    # df_temp = df_temp.dropna()

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
    print("\n✅ R2/eta2 calculés pour les colonnes :")
    for _, row in results.iterrows():
        print(f"  - {row['col']}: {row['r2/eta2']:.4f}")
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
    
    cols = [c for c in df.columns if not c.startswith("ALS") and c != "Target"]
    # df_temp = df[cols]
    df_temp = df
    # 1. Compter le nombre de valeurs non-nulles pour chaque colonne
    # df.count() retourne une Series où les valeurs sont les comptes non-nuls
    non_null_counts = df_temp.count()

    # 2. Obtenir le nombre total de lignes dans le DataFrame
    total_rows = len(df_temp)
    
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