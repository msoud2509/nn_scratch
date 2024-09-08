import numpy as np
from load_mnistdataset import get_data, one_hot_encode
from mnist_nn_batches import NN_batches
from mnist_nn import NeuralNetwork

def main(hiddenlayer_neurons, learning_rate, epochs, batches = False, batch_size = 128):
    #fetch data
    #always use encoded sets when calculating loss
    X_train, y_train, X_test, y_test = get_data()
    y_train_encoded = one_hot_encode(y_train)

    #create instance of class and get initial (random) loss
    if batches:
        testNN = NN_batches(X_train, y_train_encoded, hiddenlayer_neurons, learning_rate, batch_size)
        print('Batch size is defaulted to 128 unless otherwise specified')
    else:
        testNN = NeuralNetwork(X_train, y_train_encoded, hiddenlayer_neurons, learning_rate)
    testNN.fullforward_pass(X_train)
    print('Initial loss (completely random): ', testNN.compute_crossentropy_loss(y_train_encoded))

    #train data
    print('training')
    if batches:
        num_iterations = epochs
        for i in range(num_iterations):
            if i == round(num_iterations / 4):
                print('25%')
            if i == round(num_iterations / 2):
                print('50%')
            if i == round(num_iterations * 3 / 4):
                print('75%')
            testNN.train_bybatch()
        print('training complete')
    else:
        num_iterations = epochs
        for i in range(num_iterations):
            if i == round(num_iterations / 4):
                print('25%')
            if i == round(num_iterations / 2):
                print('50%')
            if i == round(num_iterations * 3 / 4):
                print('75%')
            testNN.train()
        print('training complete')

    #if using batches, need to do a final full forward to match sets for computing training loss
    if batches:
        testNN.fullforward_pass(X_train)
    #get final loss
    print('Final train set loss: ', testNN.compute_crossentropy_loss(y_train_encoded))

    #compute accuracy, loss with test data
    testNN.fullforward_pass(X_test)
    outputs = np.argmax(testNN.activationlayer2,axis=0)

    y_test_encoded = one_hot_encode(y_test)
    accuracycount = 0
    for i in range(len(outputs)):
        if y_test_encoded[outputs[i]][i] == 1:
            accuracycount +=1

    print('Test set loss: ', testNN.compute_crossentropy_loss(y_test_encoded))

    #convert to percent cause it looks cooler
    print('Accuracy: ', (((accuracycount / len(outputs))) * 100), '%')

#example run
if __name__ == '__main__':
    main(hiddenlayer_neurons = 200, learning_rate = .99, epochs = 10, batches=True, batch_size=64)
