import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import os

class LRDataset():
    def __init__(self, csv_path, feature_cols=None, target_col='Target', test_size=0.2, random_state=42):
        self.csv_path = csv_path # Fichier csv contenant les données
        self.target_col = target_col # Target
        self.test_size = test_size # Fraction du dataset pour le test
        self.random_state = random_state # Seed

        self.dataframe = pd.read_csv(self.csv_path)

        self.feature_cols = self.dataframe.columns[~self.dataframe.columns.str.contains(r'Target')]

        self.train_df, self.test_df = train_test_split(self.dataframe, test_size=self.test_size, random_state=self.random_state) # Découpage du df en train et test

    def get_train(self):
        X_train = self.train_df[self.feature_cols]
        y_train = self.train_df[self.target_col]

        return X_train, y_train
    
    def get_test(self):
        X_test = self.test_df[self.feature_cols]
        y_test = self.test_df[self.target_col]

        return X_test, y_test

def train_lr(csv_path, test_size, random_state, log_path):
    dataset = LRDataset(csv_path=csv_path, test_size=test_size, random_state=random_state)

    X_train, y_train = dataset.get_train()
    X_test, y_test = dataset.get_test()

    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = root_mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    perf = [
        ['Metrics', 'Value'],
        ['test_mae', f'{test_mae}'],
        ['test_rmse', f'{test_rmse}'],
        ['test_r2', f'{test_r2}']
    ]

    print("======================================")
    for ligne in perf:
        print(f"{ligne[0]:<20} | {ligne[1]:>5}")

    df = pd.DataFrame(perf)
    df.to_csv(log_path, index=False)

if __name__ == '__main__':
    csv_file = 'datasets/papaiz/papaiz_3M.csv'
    test_size = 0.2
    random_state = 42
    log_path = 'baseline/logs/linear_regression/papaiz_3M.csv'

    os.makedirs('baseline/logs/linear_regression/', exist_ok=True)
    train_lr(csv_file, test_size, random_state, log_path)