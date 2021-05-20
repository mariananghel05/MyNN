import numpy as np
import cv2



class Activation:
    def __init__(self):
        pass

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def ReLu(self, x):
        return np.maximum(0, x)


class Dense:
    def __init__(self, units = 8, shape = None):
        self.units = units
        #self.activation = activation
        self.shape = shape



class Sequential:
    def __init__(self, Layers, lr= 0.01):
        self.Layers = Layers
        self.lr = lr

    def summary(self):
        
        #Generating weights
        self.weights = []
        for i in range(len(self.Layers) - 1):
            weights = np.random.normal(0.0, pow(self.Layers[i+1].units, -0.5), (self.Layers[i+1].units, self.Layers[i].units))
            self.weights.append(weights)
        #return self.weights
    
    def predict(self, inputs):
        
        #Create first Hidden Layer input
        hidden_inputs = np.dot(self.weights[0], inputs)
        hidden_output = Activation().sigmoid(hidden_inputs)
        
        #Create for rest of Layers
        for i in range(len(self.Layers) - 2):
            hidden_inputs = np.dot(self.weights[i+1], hidden_output)
            hidden_output = Activation().sigmoid(hidden_inputs)
        return hidden_output
    
    
    def train(self, inputs_list, targets_list):
        
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        hidden_inputs = []
        hidden_output = []
        #Create first Hidden Layer input
        hidden_inputs.append(np.dot(self.weights[0], inputs))
        hidden_output.append(Activation().sigmoid(hidden_inputs[0]))
        
        #Create for rest of Layers
        for i in range(len(self.Layers) - 2):
            hidden_inputs.append(np.dot(self.weights[i+1], hidden_output[i]))
            hidden_output.append(Activation().sigmoid(hidden_inputs[i+1]))
        
         #error is the (target - actual)
        output_errors = targets - hidden_output[len(hidden_output)-1]
        hidden_errors = np.dot(self.weights[len(self.weights) - 1].T, output_errors)
        
        self.weights[len(self.weights)-1] += self.lr * np.dot((output_errors * hidden_output[len(hidden_output)-1] * (1.0 - hidden_output[len(hidden_output)-1])), np.transpose(hidden_output[len(hidden_output)-2]))
        
        for i in reversed(range(len(self.weights)-1)):
            #print("here")
            if(i > 0):
                self.weights[i] += self.lr * np.dot((hidden_errors * hidden_output[i] * (1.0 - hidden_output[i])), np.transpose(hidden_output[i-1]))
                hidden_errors = np.dot(self.weights[i].T, hidden_errors)
    
        self.weights[0] += self.lr * np.dot((hidden_errors * hidden_output[0] * (1.0 - hidden_output[0])), np.transpose(inputs))



#############################################################

data_file = open("mnist_train_100.csv", 'r')
data_list = data_file.readlines()        
data_file.close()

# load the mnist training data CSV file into a list
training_data_file = open("mnist_train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

import matplotlib.pyplot

all_values = data_list[0].split(',')
image_array = np.asfarray(all_values[1:]).reshape((28,28))
matplotlib.pyplot.imshow(image_array, cmap='Greys',interpolation='None')

scaled_input = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

targets = np.zeros(10) + 0.01
targets[int(all_values[0])] = 0.99


test_data_file = open("mnist_test_10.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()


model = Sequential([Dense(units = 784),
                    Dense(units = 100),
                    Dense(units = 10)], lr =0.3)


model.summary()
targetss = []
inputss = []
for record in training_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # scale and shift the inputs
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

    inputss.append(inputs)
    # create the target output values (all 0.01, except the desired label which
    targets = np.zeros(10) + 0
 
    targetss.append(targets)
    # all_values[0] is the target label for this record
    targets[int(all_values[0])] = 0.99
    model.train(inputs, targets)    
for i, j in zip(inputss, targetss):
    print(model.predict(i), j)
