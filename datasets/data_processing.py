import pandas as pd

raw_data = pd.read_csv('datasets/3_month_alsfrs_only[0-99].csv')


print(raw_data.columns)
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
                   '[0-99]ALSFRS_R_Total',
                   '[100-199]ALSFRS_R_Total',]
stripped_data = raw_data[columns_to_keep]
stripped_data = stripped_data.rename(columns={'[100-199]ALSFRS_R_Total': 'Target'})

stripped_data.to_csv('datasets/[0-99]-train.csv', index=False)