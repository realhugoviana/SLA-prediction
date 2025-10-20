import torch

DATA_PATH = 'datasets/formated_1.csv'
INPUT_SIZE = 11
OUTPUT_SIZE = 1
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")