import torch
import pandas as pd

DATA_PATH = 'datasets/formated_2.csv'
INPUT_SIZE = len(pd.read_csv(DATA_PATH).columns) - 1
OUTPUT_SIZE = 1
MAX_EPOCHS = 100

if torch.backends.mps.is_available():
    ACCELERATOR = "mps"
elif torch.cuda.is_available():
    ACCELERATOR = "gpu"
else:
    ACCELERATOR = "cpu"