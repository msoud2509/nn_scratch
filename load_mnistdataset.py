import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

def get_data():
    '''gets mnist data from scikit-learn in the form of numpyarrays
       must one hot encode output data as demonstrated in main method in main.py
    '''

    
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
    '''converts vector of output values to matrix of size 10x(size of vector)
       must be converted to fit model and effectively compute loss
    '''
    
    y_encoded = np.zeros((10,y.shape[0]))
    for i in range(y.shape[0]):
        y_encoded[int(y[i]),i] = 1

    return y_encoded
