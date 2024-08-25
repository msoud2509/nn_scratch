from mnist_nn import NeuralNetwork
from load_mnistdataset import get_data, one_hot_encode

#only if you want to train in batches, else use the superclass in mnist_nn.py
class NN_batches(NeuralNetwork):
    def __init__(self, X, y, hiddenlayer_neurons, learning_rate, batch_size):
        super().__init__(X, y, hiddenlayer_neurons, learning_rate)
        '''x and y already shuffled from train test split, if want to shuffle again for batch split, use this code:
        
        perm = np.random.permutation(self.inputs.shape[1])
        self.inputs = self.inputs[:, perm]
        self.y = self.y[:, perm]'''

        self.num_batches = (X.shape[1] // batch_size)
        self.batch_size = batch_size

    def batch_forward_pass(self, index):
        #set the begin and end of each batch before feedforward
        begin = index * self.batch_size
        end = min(begin + self.batch_size, self.inputs.shape[1] - 1)
        X_batch = self.inputs[:, begin:end]
        
        super().fullforward_pass(X_batch)
    
    def batch_backward(self, index):
        #set the begin and end of each batch before backprop
        begin = index * self.batch_size
        end = min(begin + self.batch_size, self.inputs.shape[1] - 1)
        X_batch = self.inputs[:, begin:end]
        y_batch = self.y[:, begin:end]
        
        super().backward(X_batch, y_batch)
    
    def train_bybatch(self):
        #loop ignores remaining batches after dividing into sets of equal size of batch_size
        for i in range(self.num_batches):
            self.batch_forward_pass(i)
            self.batch_backward(i)