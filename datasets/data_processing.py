import pandas as pd

df_raw = pd.read_csv('datasets/added_subscores.csv')

#######################################################
################### MERGING COLUMNS ###################
#######################################################

################### Merging Q5 ###################

df_q5_merged = df_raw.copy()
for interval in ['[0-99]', '[100-199]', '[200-299]', '[300-399]']:
    df_q5_merged[f'{interval}Q5_Cutting'] = df_q5_merged[f'{interval}Q5a_Cutting_without_Gastrostomy'].fillna(df_q5_merged[f'{interval}Q5b_Cutting_with_Gastrostomy'])

################### Merging Q10 & R1 ###################

df_q10_merged = df_q5_merged.copy()
for interval in ['[0-99]', '[100-199]', '[200-299]', '[300-399]']:
    df_q10_merged[f'{interval}Q10_Respiratory'] = df_q10_merged[f'{interval}Q10_Respiratory'].fillna(df_q10_merged[f'{interval}R_1_Dyspnea'])

################### Recalcul ALSFRS ###################

df_updated_scores = df_q10_merged.copy()
for interval in ['[0-99]', '[100-199]', '[200-299]', '[300-399]']:
    df_updated_scores[f'{interval}ALSFRS_Total'] = df_updated_scores[[f'{interval}Q1_Speech',
                                                                   f'{interval}Q2_Salivation',
                                                                   f'{interval}Q3_Swallowing',
                                                                   f'{interval}Q4_Handwriting',
                                                                   f'{interval}Q5_Cutting',
                                                                   f'{interval}Q6_Dressing_and_Hygiene',
                                                                   f'{interval}Q7_Turning_in_Bed',
                                                                   f'{interval}Q8_Walking',
                                                                   f'{interval}Q9_Climbing_Stairs',
                                                                   f'{interval}Q10_Respiratory']].sum(axis=1)
    
###########################################################
################### SEPERATING DATASETS ###################
###########################################################

################### ALSFRS_ONLY_ALL ###################

alsfrs_cols = [] 
for interval in ['[0-99]', '[100-199]', '[200-299]', '[300-399]']:
    alsfrs_cols.extend([f'{interval}Q1_Speech',
                        f'{interval}Q2_Salivation',
                        f'{interval}Q3_Swallowing',
                        f'{interval}Q4_Handwriting',
                        f'{interval}Q5_Cutting',
                        f'{interval}Q6_Dressing_and_Hygiene',
                        f'{interval}Q7_Turning_in_Bed',
                        f'{interval}Q8_Walking',
                        f'{interval}Q9_Climbing_Stairs',
                        f'{interval}Q10_Respiratory',
                        f'{interval}ALSFRS_Total'])
    
df_alsfrs_only = df_updated_scores[alsfrs_cols]

################### ALSFRS_ONLY[0-99] ###################

df_alsfrs_only_trimestre_1 = df_alsfrs_only.copy()

for col in df_alsfrs_only_trimestre_1.columns:
    if col == '[100-199]ALSFRS_Total':
        continue
    elif '[100-199]' in col or '[200-299]' in col or '[300-399]' in col:
        df_alsfrs_only_trimestre_1.drop(col, axis=1, inplace=True)
df_alsfrs_only_trimestre_1.rename(columns={'[100-199]ALSFRS_Total': 'Target'}, inplace=True)

################### ALSFRS_ONLY[100-199] ###################

df_alsfrs_only_trimestre_2 = df_alsfrs_only.copy()

for col in df_alsfrs_only_trimestre_2.columns:
    if col == '[200-299]ALSFRS_Total':
        continue
    elif '[200-299]' in col or '[300-399]' in col:
        df_alsfrs_only_trimestre_2.drop(col, axis=1, inplace=True)
df_alsfrs_only_trimestre_2.rename(columns={'[200-299]ALSFRS_Total': 'Target'}, inplace=True)

################### ALSFRS_ONLY[200-299] ###################

df_alsfrs_only_trimestre_3 = df_alsfrs_only.copy()

for col in df_alsfrs_only_trimestre_3.columns:
    if col == '[300-399]ALSFRS_Total':
        continue
    elif '[300-399]' in col:
        df_alsfrs_only_trimestre_3.drop(col, axis=1, inplace=True)
df_alsfrs_only_trimestre_3.rename(columns={'[300-399]ALSFRS_Total': 'Target'}, inplace=True)

##############################################
################### TO CSV ###################
##############################################

df_alsfrs_only_trimestre_1.to_csv('datasets/MLP_alsfrs_only[0-99].csv', index=False)
df_alsfrs_only_trimestre_2.to_csv('datasets/MLP_alsfrs_only[100-199].csv', index=False)
df_alsfrs_only_trimestre_3.to_csv('datasets/MLP_alsfrs_only[200-299].csv', index=False)

############################################
################### OSEF ###################
############################################


# alsfrs_cols = ['[0-99]Q1_Speech', 
#                 '[0-99]Q2_Salivation',
#                 '[0-99]Q3_Swallowing',
#                 '[0-99]Q4_Handwriting',
#                 '[0-99]Q5a_Cutting_without_Gastrostomy',
#                 '[0-99]Q5b_Cutting_with_Gastrostomy',
#                 '[0-99]Q6_Dressing_and_Hygiene',
#                 '[0-99]Q7_Turning_in_Bed',
#                 '[0-99]Q8_Walking',
#                 '[0-99]Q9_Climbing_Stairs',
#                 '[0-99]Q10_Respiratory',
#                 '[0-99]R_1_Dyspnea',
#                 '[0-99]ALSFRS_R_Total',
#                 '[100-199]Q1_Speech', 
#                 '[100-199]Q2_Salivation',
#                 '[100-199]Q3_Swallowing',
#                 '[100-199]Q4_Handwriting',
#                 '[100-199]Q5a_Cutting_without_Gastrostomy',
#                 '[100-199]Q5b_Cutting_with_Gastrostomy',
#                 '[100-199]Q6_Dressing_and_Hygiene',
#                 '[100-199]Q7_Turning_in_Bed',
#                 '[100-199]Q8_Walking',
#                 '[100-199]Q9_Climbing_Stairs',
#                 '[100-199]Q10_Respiratory',
#                 '[100-199]R_1_Dyspnea',
#                 '[100-199]ALSFRS_R_Total',
#                 '[200-299]Q1_Speech', 
#                 '[200-299]Q2_Salivation',
#                 '[200-299]Q3_Swallowing',
#                 '[200-299]Q4_Handwriting',
#                 '[200-299]Q5a_Cutting_without_Gastrostomy',
#                 '[200-299]Q5b_Cutting_with_Gastrostomy',
#                 '[200-299]Q6_Dressing_and_Hygiene',
#                 '[200-299]Q7_Turning_in_Bed',
#                 '[200-299]Q8_Walking',
#                 '[200-299]Q9_Climbing_Stairs',
#                 '[200-299]Q10_Respiratory',
#                 '[200-299]R_1_Dyspnea',
#                 '[200-299]ALSFRS_R_Total'
#                 '[300-399]Q1_Speech', 
#                 '[300-399]Q2_Salivation',
#                 '[300-399]Q3_Swallowing',
#                 '[300-399]Q4_Handwriting',
#                 '[300-399]Q5a_Cutting_without_Gastrostomy',
#                 '[300-399]Q5b_Cutting_with_Gastrostomy',
#                 '[300-399]Q6_Dressing_and_Hygiene',
#                 '[300-399]Q7_Turning_in_Bed',
#                 '[300-399]Q8_Walking',
#                 '[300-399]Q9_Climbing_Stairs',
#                 '[300-399]Q10_Respiratory',
#                 '[300-399]R_1_Dyspnea',
#                 '[300-399]ALSFRS_R_Total']

# alsfrs_data = raw_data[alsfrs_cols]


# stripped_data = stripped_data.dropna(axis=0, how='any')

# stats = stripped_data.describe(include='all').transpose()
# stats['missing'] = stripped_data.isna().sum()

# print(stats)

# stripped_data_1 = stripped_data.copy()

# for col in stripped_data_1.columns:
#     if col == '[100-199]ALSFRS_R_Total':
#         continue
#     elif '[100-199]' in col or '[200-299]' in col:
#         stripped_data_1.drop(col, axis=1, inplace=True)
# stripped_data_1.rename(columns={'[100-199]ALSFRS_R_Total': 'Target'}, inplace=True)

# stripped_data_2 = stripped_data.copy()

# for col in stripped_data_2.columns:
#     if col == '[200-299]ALSFRS_R_Total':
#         continue
#     elif '[200-299]' in col:
#         stripped_data_2.drop(col, axis=1, inplace=True)
# stripped_data_2.rename(columns={'[200-299]ALSFRS_R_Total': 'Target'}, inplace=True)

# stripped_data.to_csv('datasets/formatted_all.csv', index=False)
# stripped_data_1.to_csv('datasets/formated_1.csv', index=False)
# stripped_data_2.to_csv('datasets/formated_2.csv', index=False)