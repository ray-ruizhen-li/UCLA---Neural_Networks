import pandas as pd

def load_and_preprocess_data(data_path):
    
    # Import the data from 'credit.csv'
    df = pd.read_csv(data_path)
    # Converting the target variable into a categorical variable
    df['Admit_Chance']=(df['Admit_Chance'] >=0.8).astype(int)
    
    # Dropping columns
    df = df.drop(['Serial_No'], axis=1)
    
    return df