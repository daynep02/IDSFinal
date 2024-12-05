from sklearn.model_selection import train_test_split
import numpy as np
def build_features(data):

    X = data.drop(["Occupancy", 'date'], axis=1)
    y = data["Occupancy"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y)
    X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.125)
    return X_train, X_validate, X_test, y_train, y_validate, y_test