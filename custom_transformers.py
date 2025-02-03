import numpy as np 
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd 



def handle_outliers(X):
    X = X.copy()
    for col in X.columns:
        q1 = X[col].quantile(0.25)
        q3 = X[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        X[col] = X[col].clip(lower_bound, upper_bound)
    return X



# Function for label encoding and scaling
def label_encode_and_scale(X):
    X_encoded = X.copy()
    for col in X_encoded.columns:
        X_encoded[col] = LabelEncoder().fit_transform(X_encoded[col])
    return StandardScaler().fit_transform(X_encoded)


def fillna_median(X):
    X = X.copy()
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())  # Fill missing values with median
    return X

def square(X):
    return X**2