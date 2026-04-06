# model.py

import pandas as pd
from sklearn.tree import DecisionTreeRegressor

def train_model():
    # Load data
    df = pd.read_csv("cleaned_data.csv")

    # Features & target
    X = df.drop("Rating", axis=1)
    y = df["Rating"]

    # Train model
    model = DecisionTreeRegressor()
    model.fit(X, y)

    # Return model + feature columns
    return model, X.columns
