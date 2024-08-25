import numpy as np
from load_mnistdataset import get_data, one_hot_encode

class NeuralNetwork:
    def __init__(self, X, y, hiddenlayer_neurons, learning_rate):
        self.inputs = X
        self.y = y

        self.lr = learning_rate
        self.hiddenlayer_neurons = hiddenlayer_neurons

        self.weights1 = np.random.randn(self.hiddenlayer_neurons,self.inputs.shape[0]) * np.sqrt(1. / self.inputs.shape[0])
        self.weights2 = np.random.randn(y.shape[0],self.hiddenlayer_neurons) * np.sqrt(1. / self.hiddenlayer_neurons)

        self.biases1 = np.zeros((self.hiddenlayer_neurons, 1))
        self.biases2 = np.zeros((10, 1))

        #momentum constant
        self.beta = .8

        #initialize momentum variables for backprop
        self.V_dW1 = np.zeros(self.weights1.shape)
        self.V_db1 = np.zeros(self.biases1.shape)
        self.V_dW2 = np.zeros(self.weights2.shape)
        self.V_db2 = np.zeros(self.biases2.shape)


    def compute_crossentropy_loss(self, y):

        L_sum = np.sum(np.multiply(y, np.log(self.activationlayer2)))
        m = y.shape[1]
        L = -(1/m) * L_sum

        return L
    
    #activation functions
    def activation_SoftMax(self, x):
        assert 0 not in np.sum(np.exp(x), axis=0)
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    
    def crossentropy_SoftMax_derivative(self, output, y_train):
        return output - y_train
    
    def ReLu(self, x, derivative = False):
        if derivative:
            return np.where(x <= 0, 0, 1)
        return np.maximum(0, x)
    
    #a full forward pass through all layers
    def fullforward_pass(self, x):
        self.hiddenlayer = np.dot(self.weights1, x) + self.biases1
        self.activationlayer1 = self.ReLu(self.hiddenlayer)
        self.outputlayer = np.dot(self.weights2, self.activationlayer1) + self.biases2
        self.activationlayer2 = self.activation_SoftMax(self.outputlayer)
    

    

    def backward(self, x, y):
        #get gradient for second set of weights and biases
        dL_dZ2 = self.crossentropy_SoftMax_derivative(self.activationlayer2, y)
        dL_dW2 = (1./y.shape[1]) * np.matmul(dL_dZ2, self.activationlayer1.T)
        dL_db2 = (1./y.shape[1]) * np.sum(dL_dZ2, axis=1, keepdims=True)
        
        #get gradient for first set of weights and biases
        dL_dA1 = np.matmul(self.weights2.T, dL_dZ2)
        dL_dZ1 = dL_dA1 * self.ReLu(self.hiddenlayer, derivative = True)
        dL_dW1 = (1./y.shape[1]) * np.matmul(dL_dZ1, x.T)
        dL_db1 = (1./y.shape[1]) * np.sum(dL_dZ1, axis=1, keepdims=True)

        #add momentum to gradient of weights/biases
        self.V_dW1 = (self.beta * self.V_dW1 + (1. - self.beta) * dL_dW1)
        self.V_db1 = (self.beta * self.V_db1 + (1. - self.beta) * dL_db1)
        self.V_dW2 = (self.beta * self.V_dW2 + (1. - self.beta) * dL_dW2)
        self.V_db2 = (self.beta * self.V_db2 + (1. - self.beta) * dL_db2)

        #update weights/biases
        self.weights2 -= self.lr * self.V_dW2
        self.biases2 -= self.lr * self.V_db2
        self.weights1 -= self.lr * self.V_dW1
        self.biases1 -= self.lr * self.V_db1

    def train(self):
        self.fullforward_pass(self.inputs)
        self.backward(self.inputs, self.y)