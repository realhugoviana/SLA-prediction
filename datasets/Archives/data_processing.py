import pandas as pd

df_raw = pd.read_csv('datasets/added_subscores.csv')

#######################################################
################### MERGING COLUMNS ###################
#######################################################

################### Merging Q5 ###################

df_q5_merged = df_raw.copy()
for interval in ['[0-99]', '[100-199]', '[200-299]', '[300-399]']:
    for stat in ['', '_moyenne', '_mediane', '_min', '_max', '_diff']:
        df_q5_merged[f'{interval}Q5_Cutting{stat}'] = df_q5_merged[f'{interval}Q5a_Cutting_without_Gastrostomy{stat}'].fillna(df_q5_merged[f'{interval}Q5b_Cutting_with_Gastrostomy{stat}'])

################### Merging Q10 & R1 ###################

df_q10_merged = df_q5_merged.copy()
for interval in ['[0-99]', '[100-199]', '[200-299]', '[300-399]']:
    df_q10_merged[f'{interval}Q10_Respiratory'] = df_q10_merged[f'{interval}Q10_Respiratory'].fillna(df_q10_merged[f'{interval}R_1_Dyspnea'])

#######################################################
################### RECALCUL ALSFRS ###################
#######################################################

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

################### ALSFRS_STATS_ALL ###################

alsfrs_stats_cols = []
for interval in ['[0-99]', '[100-199]', '[200-299]', '[300-399]']:
    alsfrs_stats_cols.extend([f'{interval}Q1_Speech',
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
    
    for stat in ['_moyenne', '_mediane', '_min', '_max', '_diff']:
        alsfrs_stats_cols.extend([f'{interval}Q1_Speech{stat}',
                                  f'{interval}Q2_Salivation{stat}',
                                  f'{interval}Q3_Swallowing{stat}',
                                  f'{interval}Q4_Handwriting{stat}',
                                  f'{interval}Q5_Cutting{stat}',
                                  f'{interval}Q6_Dressing_and_Hygiene{stat}',
                                  f'{interval}Q7_Turning_in_Bed{stat}',
                                  f'{interval}Q8_Walking{stat}',
                                  f'{interval}Q9_Climbing_Stairs{stat}',
                                  f'{interval}Q10_Respiratory{stat}'])
        

df_alsfrs_stats = df_updated_scores[alsfrs_stats_cols]

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
    
df_alsfrs_only = df_alsfrs_stats[alsfrs_cols]

################### ALSFRS_LABS_ALL ###################

# À compléter plus tard

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

################### ALSFRS_STATS[0-99] ###################

df_alsfrs_stats_trimestre_1 = df_alsfrs_stats.copy()

for col in df_alsfrs_stats_trimestre_1.columns:
    if col == '[100-199]ALSFRS_Total':
        continue
    elif '[100-199]' in col or '[200-299]' in col or '[300-399]' in col:
        df_alsfrs_stats_trimestre_1.drop(col, axis=1, inplace=True)
df_alsfrs_stats_trimestre_1.rename(columns={'[100-199]ALSFRS_Total': 'Target'}, inplace=True)

################### ALSFRS_STATS[100-199] ###################

df_alsfrs_stats_trimestre_2 = df_alsfrs_stats.copy()

for col in df_alsfrs_stats_trimestre_2.columns:
    if col == '[200-299]ALSFRS_Total':
        continue
    elif '[200-299]' in col or '[300-399]' in col:
        df_alsfrs_stats_trimestre_2.drop(col, axis=1, inplace=True)
df_alsfrs_stats_trimestre_2.rename(columns={'[200-299]ALSFRS_Total': 'Target'}, inplace=True)

################### ALSFRS_STATS[200-299] ###################

df_alsfrs_stats_trimestre_3 = df_alsfrs_only.copy()

for col in df_alsfrs_stats_trimestre_3.columns:
    if col == '[300-399]ALSFRS_Total':
        continue
    elif '[300-399]' in col:
        df_alsfrs_stats_trimestre_3.drop(col, axis=1, inplace=True)
df_alsfrs_stats_trimestre_3.rename(columns={'[300-399]ALSFRS_Total': 'Target'}, inplace=True)

################### ALSFRS_LABS[0-99] ###################

# À compléter plus tard

################### ALSFRS_LABS[100-199] ###################

# À compléter plus tard

################### ALSFRS_LABS[200-299] ###################

# À compléter plus tard

##############################################
################### TO CSV ###################
##############################################

################### ALSFRS_ONLY ###################

df_alsfrs_only_trimestre_1.to_csv('datasets/MLP_alsfrs_only[0-99].csv', index=False)
df_alsfrs_only_trimestre_2.to_csv('datasets/MLP_alsfrs_only[100-199].csv', index=False)
df_alsfrs_only_trimestre_3.to_csv('datasets/MLP_alsfrs_only[200-299].csv', index=False)

################### ALSFRS_STATS ###################

df_alsfrs_stats_trimestre_1.to_csv('datasets/MLP_alsfrs_stats[0-99].csv', index=False)
df_alsfrs_stats_trimestre_2.to_csv('datasets/MLP_alsfrs_stats[100-199].csv', index=False)
df_alsfrs_stats_trimestre_3.to_csv('datasets/MLP_alsfrs_stats[200-299].csv', index=False)

################### ALSFRS_LABS ###################

# À compléter plus tard