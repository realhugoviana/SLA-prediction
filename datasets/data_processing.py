import pandas as pd

df_all = pd.read_csv('datasets/Lucas/PROACT_ALSFRS_INTERVALLES.csv')

#######################################################
################### DROPPING ALSFRS ###################
#######################################################

df_alsfrs_r = df_all.dropna(subset=['ALS_ALSFRS_R_Total_0_90_Mean'])

# Retirer les sous-scores
for col in df_alsfrs_r.columns:
    if 'Bulbar' in col or 'Upper_Limb' in col or 'Lower_Limb' in col or 'Respiratory_Score' in col:
        df_alsfrs_r.drop(col, axis=1, inplace=True)
    
###########################################################
################### SEPERATING DATASETS ###################
###########################################################

intervalles = ['0_90', '90_180', '180_270', '270_360', '360_450']

for target_i in range(1, 5):
    for i in range(1, 2**target_i):
        sources = [int(x) for x in format(i, '04b')]
        intervalles_sources = [intervalles[j] for j in range(4) if sources[3-j] == 1]
        target = intervalles[target_i]

        df_temp = df_alsfrs_r.copy()

        for col in df_temp.columns:
            if col == f'ALS_ALSFRS_R_Total_{target}_Central':
                df_temp.rename(columns={col: 'Target'}, inplace=True)
            elif any(intervalle in col for intervalle in intervalles_sources):
                pass
            else:
                df_temp.drop(col, axis=1, inplace=True)
        
        df_temp = df_temp.dropna()
        
        df_temp.to_csv(f'datasets/MLP_alsfrs-r_T{"-T".join([str(j+1) for j in range(4) if sources[3-j] == 1])}_T{target_i+1}.csv', index=False)
