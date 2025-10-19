import pandas as pd

raw_data = pd.read_csv('datasets/3_month_alsfrs_only[200-299].csv')

columns_to_keep = ['[0-99]Q1_Speech', 
                   '[0-99]Q2_Salivation',
                   '[0-99]Q3_Swallowing',
                   '[0-99]Q4_Handwriting',
                   '[0-99]Q5a_Cutting_without_Gastrostomy',
                   '[0-99]Q6_Dressing_and_Hygiene',
                   '[0-99]Q7_Turning_in_Bed',
                   '[0-99]Q8_Walking',
                   '[0-99]Q9_Climbing_Stairs',
                   '[0-99]Q10_Respiratory',
                   '[0-99]R_1_Dyspnea',
                   '[0-99]ALSFRS_R_Total',
                   '[100-199]Q1_Speech', 
                   '[100-199]Q2_Salivation',
                   '[100-199]Q3_Swallowing',
                   '[100-199]Q4_Handwriting',
                   '[100-199]Q5a_Cutting_without_Gastrostomy',
                   '[100-199]Q6_Dressing_and_Hygiene',
                   '[100-199]Q7_Turning_in_Bed',
                   '[100-199]Q8_Walking',
                   '[100-199]Q9_Climbing_Stairs',
                   '[100-199]Q10_Respiratory',
                   '[100-199]R_1_Dyspnea',
                   '[100-199]ALSFRS_R_Total',
                   '[200-299]Q1_Speech', 
                   '[200-299]Q2_Salivation',
                   '[200-299]Q3_Swallowing',
                   '[200-299]Q4_Handwriting',
                   '[200-299]Q5a_Cutting_without_Gastrostomy',
                   '[200-299]Q6_Dressing_and_Hygiene',
                   '[200-299]Q7_Turning_in_Bed',
                   '[200-299]Q8_Walking',
                   '[200-299]Q9_Climbing_Stairs',
                   '[200-299]Q10_Respiratory',
                   '[200-299]R_1_Dyspnea',
                   '[200-299]ALSFRS_R_Total']
stripped_data = raw_data[columns_to_keep]

stripped_data['[0-99]Q10_Respiratory'] = stripped_data['[0-99]Q10_Respiratory'].fillna(
    stripped_data['[0-99]R_1_Dyspnea']
)
stripped_data['[100-199]Q10_Respiratory'] = stripped_data['[100-199]Q10_Respiratory'].fillna(
    stripped_data['[100-199]R_1_Dyspnea']
)
stripped_data['[200-299]Q10_Respiratory'] = stripped_data['[200-299]Q10_Respiratory'].fillna(
    stripped_data['[200-299]R_1_Dyspnea']
)

# supprimer la colonne source si on ne veut garder que la colonne fusionnée
stripped_data = stripped_data.drop(columns=['[0-99]R_1_Dyspnea'])
stripped_data = stripped_data.drop(columns=['[100-199]R_1_Dyspnea'])
stripped_data = stripped_data.drop(columns=['[200-299]R_1_Dyspnea'])

# créer une colonne qui somme les questions de chaque période
stripped_data['[0-99]ALSFRS_R_Total'] = stripped_data[['[0-99]Q1_Speech',
                                                    '[0-99]Q2_Salivation',
                                                    '[0-99]Q3_Swallowing',
                                                    '[0-99]Q4_Handwriting',
                                                    '[0-99]Q5a_Cutting_without_Gastrostomy',
                                                    '[0-99]Q6_Dressing_and_Hygiene',
                                                    '[0-99]Q7_Turning_in_Bed',
                                                    '[0-99]Q8_Walking',
                                                    '[0-99]Q9_Climbing_Stairs', 
                                                    '[0-99]Q10_Respiratory']].sum(axis=1)

stripped_data['[100-199]ALSFRS_R_Total'] = stripped_data[['[100-199]Q1_Speech',
                                                     '[100-199]Q2_Salivation',
                                                     '[100-199]Q3_Swallowing',
                                                     '[100-199]Q4_Handwriting',
                                                     '[100-199]Q5a_Cutting_without_Gastrostomy',
                                                     '[100-199]Q6_Dressing_and_Hygiene',
                                                     '[100-199]Q7_Turning_in_Bed',
                                                     '[100-199]Q8_Walking',
                                                     '[100-199]Q9_Climbing_Stairs', 
                                                     '[100-199]Q10_Respiratory']].sum(axis=1)

stripped_data['[200-299]ALSFRS_R_Total'] = stripped_data[['[200-299]Q1_Speech',
                                                     '[200-299]Q2_Salivation',
                                                     '[200-299]Q3_Swallowing',
                                                     '[200-299]Q4_Handwriting',
                                                     '[200-299]Q5a_Cutting_without_Gastrostomy',
                                                     '[200-299]Q6_Dressing_and_Hygiene',
                                                     '[200-299]Q7_Turning_in_Bed',
                                                     '[200-299]Q8_Walking',
                                                     '[200-299]Q9_Climbing_Stairs', 
                                                     '[200-299]Q10_Respiratory']].sum(axis=1)


stripped_data = stripped_data.dropna(axis=0, how='any')

stats = stripped_data.describe(include='all').transpose()
stats['missing'] = stripped_data.isna().sum()

print(stats)

stripped_data_1 = stripped_data.copy()

for col in stripped_data_1.columns:
    if col == '[100-199]ALSFRS_R_Total':
        continue
    elif '[100-199]' in col or '[200-299]' in col:
        stripped_data_1.drop(col, axis=1, inplace=True)
stripped_data_1.rename(columns={'[100-199]ALSFRS_R_Total': 'Target'}, inplace=True)

stripped_data_2 = stripped_data.copy()

for col in stripped_data_2.columns:
    if col == '[200-299]ALSFRS_R_Total':
        continue
    elif '[200-299]' in col:
        stripped_data_2.drop(col, axis=1, inplace=True)
stripped_data_2.rename(columns={'[200-299]ALSFRS_R_Total': 'Target'}, inplace=True)

stripped_data.to_csv('datasets/formatted_all.csv', index=False)
stripped_data_1.to_csv('datasets/formated_1.csv', index=False)
stripped_data_2.to_csv('datasets/formated_2.csv', index=False)