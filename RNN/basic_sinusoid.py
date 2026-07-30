import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import torch
import lightning as L
from sklearn.model_selection import train_test_split

from model import RNNmodel, AutoregressiveRNN
from data import DataModule, AutoregressiveDataModule
# Permets d'utiliser le GPU
if torch.backends.mps.is_available(): # MPS pour mac
    accelerator = "mps"
elif torch.cuda.is_available(): # Cuda sinon
    accelerator = "gpu"
else:
    accelerator = "cpu" # CPU si Cuda pas supporté

if __name__ == "__main__":

    L.seed_everything(42)

    # Generate a sine wave  
    time_steps = np.linspace(0, 100, 500)   # 500 points between 0 and 100  
    data = np.sin(time_steps) # Create a sine wave  
    data = data + 0.1 * np.random.normal(size=data.shape)  # Add some noise to the sine wave
    
    # Convert to DataFrame  
    df = pd.DataFrame(data, columns=['value']) 

    # plt.figure(figsize=(10, 4)) 
    # plt.plot(df['value']) 
    # plt.title("Sine Wave") 
    # plt.xlabel("Time Step") 
    # plt.ylabel("Value") 
    # plt.show() 
    
    data = df['value'].values  # Convert to numpy array for sequence creation 

    def create_sequences(data, seq_length):  
        series = [] 
        for i in range(len(data) - seq_length):  
            row = data[i:i+seq_length] # Sequence of length `seq_length` 
            series.append(row)

        df = pd.DataFrame(series)
        df.columns = [f't_M{i}' for i in range(-seq_length+1, 1, 1)] # Name columns as t_M20, t_M19, ..., t_M0

        return df


    SEQ_LENGTH = 21 # Number of past time steps to look at  

    df_sequences = create_sequences(data, SEQ_LENGTH)

    # df_train, df_test = train_test_split(df_sequences, test_size=0.2, shuffle=False) # Split the data into training and testing sets

    dm = DataModule(data="datasets/interpolation_baseline/sliding_windows.csv", batch_size=32, n_folds=-1)

    model = RNNmodel(1, 
                    output_dim=1, 
                    architecture="RNN",
                    n_layer=3, 
                    n_units=64, 
                    learning_rate=0.0012405199434288984, 
                    activation='relu', 
                    optimizer='Adam', 
                    criterion='Huber', 
                    weight_decay=2.126140471991739e-06, 
                    dropout=2.535060493253868e-05, 
                    bidirectional=True)

    trainer = L.Trainer(
                max_epochs=10, # Nombre d'epoch maximum
                accelerator=accelerator, # GPU si possible
                enable_checkpointing=False
            )

    trainer.fit(model, dm)
    trainer.validate(model, dm)

    autoregressive_model = AutoregressiveRNN(model)

    df_test = pd.read_csv("datasets/interpolation/test/df_test_13.csv")  # Load the test dataset
    df_test = df_test

    X_test_np = df_test.drop(columns=df_test.columns[df_test.columns.str.contains(r'Target')]).values.astype('float32')   # Features
    y_test_np = df_test[df_test.columns[df_test.columns.str.contains(r'Target')]].values.astype('float32') 
    
    print(f"X_test shape: {X_test_np.shape}, y_test shape: {y_test_np.shape}")

    # for i in range(5):  # Test on the first 5 samples
    #     j = np.random.randint(0, len(X_test_np))  # Randomly select an index

    #     X_test = torch.from_numpy(X_test_np[j]).float()
    #     y_test = torch.Tensor(y_test_np[j])
    #     print(f"X_test shape after unsqueeze: {X_test.shape}, y_test shape: {y_test.shape}")
    #     predictions = []
    #     with torch.no_grad():  
    #         y_hat, hx = model(X_test)
    #         predictions.append(y_hat.squeeze().numpy())  # Convert to numpy array for plotting

    #         for i in range(1, y_test.shape[0]):
    #                 y_hat, hx = model(y_hat, hx)
    #                 predictions.append(y_hat.squeeze().numpy())  # Convert to numpy array for plotting
            


    #     # Plot actual vs predicted  
    #     plt.figure(figsize=(10, 4))  
    #     plt.plot(y_test, label='Actual')  
    #     plt.plot(predictions, label='Predicted')  
    #     plt.title(f"RNN Prediction vs Actual for sample {j}")  
    #     plt.xlabel("Time Step")  
    #     plt.ylabel("Value")  
    #     plt.legend()  
    #     plt.show() 

    autoregressivedm = AutoregressiveDataModule(data="datasets/interpolation_baseline/test/df_test_6.csv", batch_size=32)

    trainer.test(autoregressive_model, datamodule=autoregressivedm)  # Test on the test dataset