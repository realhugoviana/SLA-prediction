import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os

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
  columns = ['alsfrs_r_0_30', 'alsfrs_r_30_60', 'alsfrs_r_60_90', 'alsfrs_r_90_120', 'alsfrs_r_120_150', 'alsfrs_r_150_180', 'alsfrs_r_180_210', 'alsfrs_r_210_240', 'alsfrs_r_240_270', 'alsfrs_r_270_300', 'alsfrs_r_300_330', 'alsfrs_r_330_360', 'alsfrs_r_360_390', 'alsfrs_r_390_420', 'Target']
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
df_noisy_missing_data = noisy_dataset_interpolation(df_noisy.copy())
print(f"Dataset generated successfully: Shape {df_noisy_missing_data.shape}")
print(df_noisy_missing_data.isna().sum())
os.makedirs("datasets/synthetic_data/", exist_ok=True)
df_noisy.to_csv('datasets/synthetic_data/synthetic_data_0_15M.csv', index=False)
df_noisy_missing_data.to_csv('datasets/synthetic_data/synthetic_data_interpolate_0_15M.csv', index=False)


# 2. Calculate the Mean Curve (The underlying signal)
# The mean of all 3000 rows should approximate the original clean sigmoid shape.
mean_curve = df_noisy_missing_data.mean(axis=0)

plt.figure(figsize=(15, 7))

# Plot 1: The Average Signal (The 'True' underlying curve)
plt.plot(mean_curve, label='Mean Curve (Approximate True Signal)', linewidth=3, color='red')


# Plot 2: Sample Individual Curves to show variability and noise
N_SAMPLES = 500 # How many random curves we want to plot for visualization

# Select N_SAMPLES unique indices from the DataFrame rows
sample_indices = random.sample(range(df_noisy_missing_data.shape[0]), N_SAMPLES)

for i in sample_indices:
    row = df_noisy_missing_data.iloc[i]
    plt.plot(row, alpha=0.1, color='skyblue', linewidth=1) # Use low opacity (alpha)

# --- Finishing the Plot ---

plt.title('Visualization of Noisy and Offset Sigmoid Curves')
plt.xlabel('X Value')
plt.ylabel('Signal Magnitude')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()