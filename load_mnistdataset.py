import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

def get_data():
    #X, y = datasets.load_digits(return_X_y = True)
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

    #convert pandas series to numpy arrays
    X = np.array(X.astype(float))
    y = np.array(y.astype(int))

    #standardize input set
    X /= 255

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    #must transpose inputs to feed correctly
    X_train = X_train.T
    X_test = X_test.T

    return X_train, y_train, X_test, y_test

def one_hot_encode(y):
    y_encoded = np.zeros((10,y.shape[0]))
    for i in range(y.shape[0]):
        y_encoded[int(y[i]),i] = 1

    return y_encoded