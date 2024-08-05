import pandas as pd

# create dummy features
def create_dummy_vars(df):
    data = pd.get_dummies(df, columns=['University_Rating','Research'])

    # Separate the input features and target variable
    X = df.drop('Admit_Chance', axis=1)
    y = df['Admit_Chance']

    return X, y