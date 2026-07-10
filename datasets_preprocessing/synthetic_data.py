import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
from sklearn.model_selection import train_test_split

def noisy_sigmoid(x):
  return (1 / (1 + np.exp(0.7*x-5)))*40+5 + np.random.normal(0, 7, 1) + np.random.normal(0, 1, len(x))

def noisy_dataset():
  x = np.linspace(0, 15, 15)

  data = []
  for i in range(3000):
    random_offset = np.random.normal(0, 3, 1)
    row = noisy_sigmoid(x+random_offset)
    data.append(row)
  
  df = pd.DataFrame(data)
  columns = ['alsfrs_r_M-14', 'alsfrs_r_M-13', 'alsfrs_r_M-12', 'alsfrs_r_M-11', 'alsfrs_r_M-10', 'alsfrs_r_M-9', 'alsfrs_r_M-8', 'alsfrs_r_M-7', 'alsfrs_r_M-6', 'alsfrs_r_M-5', 'alsfrs_r_M-4', 'alsfrs_r_M-3', 'alsfrs_r_M-2', 'alsfrs_r_M-1', 'Target_M0']
  df.columns = columns
  return df

def noisy_dataset_missing_data(df):
  noisy_df = df

  arr = noisy_df.values
  for row in arr:
    n_holes = np.random.poisson(5)
    holes = np.random.choice(np.arange(15), size=n_holes, replace=False).tolist()
    row[holes] = np.nan
  
  return pd.DataFrame(arr, columns=noisy_df.columns)

def noisy_dataset_interpolation(df):
  noisy_df = noisy_dataset_missing_data(df)

  return noisy_df.interpolate(method='linear', axis=1, limit_direction='both')


# 1. Generate the data
print("--- Generating Dataset (3000 curves) ---")
df_noisy = noisy_dataset()

df_noisy_train, df_noisy_test = train_test_split(df_noisy, test_size=0.2, random_state=42)

test_columns = ['alsfrs_r_M-14', 'alsfrs_r_M-13', 'alsfrs_r_M-12', 'Target_M-11', 'Target_M-10', 'Target_M-9', 'Target_M-8', 'Target_M-7', 'Target_M-6', 'Target_M-5', 'Target_M-4', 'Target_M-3', 'Target_M-2', 'Target_M-1', 'Target_M0']
df_noisy_test.columns = test_columns
# df_noisy_missing_data = noisy_dataset_interpolation(df_noisy.copy())
os.makedirs("datasets/synthetic_data/", exist_ok=True)
df_noisy_train.to_csv('datasets/synthetic_data/synthetic_noisy_sigmoid_train_0_15M.csv', index=False)
df_noisy_test.to_csv('datasets/synthetic_data/synthetic_noisy_sigmoid_test_0_15M.csv', index=False)
# df_noisy_missing_data.to_csv('datasets/synthetic_data/synthetic_data_interpolate_0_15M.csv', index=False)


# 2. Calculate the Mean Curve (The underlying signal)
# The mean of all 3000 rows should approximate the original clean sigmoid shape.
mean_curve = df_noisy_test.mean(axis=0)

plt.figure(figsize=(15, 7))

# Plot 1: The Average Signal (The 'True' underlying curve)
plt.plot(mean_curve, label='Mean Curve (Approximate True Signal)', linewidth=3, color='red')


# Plot 2: Sample Individual Curves to show variability and noise
N_SAMPLES = 100 # How many random curves we want to plot for visualization

# Select N_SAMPLES unique indices from the DataFrame rows
sample_indices = random.sample(range(df_noisy_test.shape[0]), N_SAMPLES)

for i in sample_indices:
    row = df_noisy_test.iloc[i]
    plt.plot(row, alpha=0.1, color='skyblue', linewidth=1) # Use low opacity (alpha)

# --- Finishing the Plot ---

plt.title('Visualization of Noisy and Offset Sigmoid Curves')
plt.xlabel('X Value')
plt.ylabel('Signal Magnitude')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()