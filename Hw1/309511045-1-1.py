import pandas as pd
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

class myNN():
    def __init__(self, architecture, learning_rate):
        self.architecture = architecture
        self.learning_rate = learning_rate
        self.params = {}
        self.init_layers()
        self.train_loss_history = []
        self.test_loss_history = []
        
    def init_layers(self):
        #np.random.seed(99)
        for index, layer in enumerate(self.architecture):
            layer_index = index + 1
            layer_input_size = layer['input_dim']
            layer_output_size = layer['output_dim']
            self.params['W' + str(layer_index)] = np.random.randn(layer_output_size, layer_input_size)*0.1
            self.params['b' + str(layer_index)] = np.random.randn(layer_output_size, 1)*0.1
            #print(self.params['W'+str(layer_index)].shape)
            #print(self.params['W'+str(layer_index)])
            #print(self.params['b'+str(layer_index)])
            #print("*******")
            
    def sigmoid(self, Z):
        return 1/(1+np.exp(-Z))
    
    def relu(self, Z):
        return np.maximum(0,Z)
    
    def linear(self, Z):
        return Z
    
    def sigmoid_backward(self, dA, Z):
        sig = self.sigmoid(Z)
        return dA*sig*(1-sig)
    
    def relu_backward(self, dA, Z):
        dZ = np.array(dA, copy = True)
        dZ[Z<=0] = 0
        return dZ
    
    def linear_backward(self, dA, Z):
        return dA
    
    def single_layer_FP(self, A_prev, W_curr, b_curr, activation='relu'):
        #print(W_curr.shape)
        #print(A_prev.shape)
        Z_curr = np.dot(W_curr, A_prev) + b_curr
        #print(Z_curr.shape)
        #print("------")
        if activation == 'relu':
            activation_func = self.relu
        elif activation == 'sigmoid':
            activation_func = self.sigmoid
        elif activation == 'linear':
            activation_func = self.linear
        else:
            raise Exception('Non-supported activation function')
        return activation_func(Z_curr), Z_curr
    
    def forward_propagation(self, x):
        memory = {}
        A_curr = x
        for index, layer in enumerate(self.architecture):
            layer_index = index + 1
            A_prev = A_curr
            activation_func = layer['activation']
            W_curr = self.params['W'+str(layer_index)]
            b_curr = self.params['b'+str(layer_index)]
            A_curr, Z_curr = self.single_layer_FP(A_prev, W_curr, b_curr, activation_func)
            memory["A"+str(index)] = A_prev
            memory["Z"+str(layer_index)] = Z_curr
        return A_curr, memory
        
    def get_loss_value(self, y, y_predict):
        #delta = y - y_predict
        #loss = np.dot(delta, delta.T)
        #return np.squeeze(loss)
        return np.power(y_predict - y, 2).sum()
    
    def single_layer_BP(self, dA_curr, W_curr, b_curr, Z_curr, A_prev, activation='relu'):
        if activation == 'relu':
            activation_func = self.relu_backward
        elif activation == 'sigmoid':
            activation_func = self.sigmoid_backward
        elif activation == 'linear':
            activation_func = self.linear_backward
        else:
            raise Exception('Non-supported activation function')
        dZ_curr = activation_func(dA_curr, Z_curr)
        dW_curr = np.dot(dZ_curr, A_prev.T)
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True)
        dA_prev = np.dot(W_curr.T, dZ_curr)
        return dA_prev, dW_curr, db_curr
    
    def back_propagation(self, y_predict, y, memory):
        grads_values = {}
        dA_prev = 2*(y_predict-y)
        #dA_prev = delta * np.ones(y.shape).reshape(1,-1)
        for layer_prev_index, layer in reversed(list(enumerate(self.architecture))):
            layer_cur_index = layer_prev_index+1
            activation_func = layer['activation']
            dA_curr = dA_prev
            A_prev = memory['A' + str(layer_prev_index)]
            Z_curr = memory['Z' + str(layer_cur_index)]
            W_curr = self.params['W'+str(layer_cur_index)]
            b_curr = self.params['b'+str(layer_cur_index)]
            dA_prev, dW_curr, db_curr = self.single_layer_BP(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation_func)
            grads_values['dW'+str(layer_cur_index)] = dW_curr
            grads_values['db'+str(layer_cur_index)] = db_curr
        return grads_values
    
    def update(self, grads_values):
        for index, layer in enumerate(self.architecture):
            layer_idx = index + 1
            self.params["W" + str(layer_idx)] -= self.learning_rate * grads_values["dW" + str(layer_idx)]        
            self.params["b" + str(layer_idx)] -= self.learning_rate * grads_values["db" + str(layer_idx)]
    
    def train(self, x_train, y_train, x_test, y_test, epochs, batch_size=16):
        for i in range(epochs):
            x_train, y_train = shuffle(x_train, y_train)
            for j in range(0 ,x_train.shape[0], batch_size):
                start = j
                end = j+batch_size
                if end >= x_train.shape[0]:
                    end = x_train.shape[0]-1
                x_min = x_train[start:end]
                y_min = y_train[start:end]
                y_predict, memory = self.forward_propagation(x_min.T)
                loss = self.get_loss_value(np.array([y_min]), y_predict)
                grads_values = self.back_propagation(y_predict, np.array([y_min]), memory)
                self.update(grads_values)
            y_train_predict = self.predict(x_train.T)
            y_test_predict = self.predict(x_test.T)
            train_loss = self.get_loss_value(np.array([y_train]), y_train_predict)
            test_loss = self.get_loss_value(np.array([y_test]), y_test_predict)
            train_acc = self.accuracy(np.array([y_train]), y_train_predict)
            test_acc = self.accuracy(np.array([y_test]), y_test_predict)
            self.train_loss_history.append(train_loss)
            self.test_loss_history.append(test_loss)
            if (i+1) % 100 == 0:
                print("Eposh : %4d/%d || training loss : %12lf || training eval : %6lf" % (i+1, epochs, train_loss, train_acc),end=" ")
                print("|| testing loss : %11lf || testing eval : %6lf" % (test_loss, test_acc))
    
    def show_parm(self):
        print(self.params)
    
    def accuracy(self, y, y_predict):
        power = np.power(y_predict - y, 2).mean()
        return np.sqrt(power)
    
    def predict(self, x):
        y_predict, memory = self.forward_propagation(x)
        return np.squeeze(y_predict)
    
    def get_loss_history(self):
        return self.train_loss_history, self.test_loss_history

def my_split_function(data, label, test_size):
    train_len = int(len(data)*(1-test_size))
    return (data[:train_len], data[train_len:], label[:train_len], label[train_len:])

def show_graph(d1, d2, labels, name, length = 0, locs='upper left'):
    plt.figure(name[0])
    t = range(len(d1))
    if length != 0:
        plt.plot(t[:length],d1[:length], 'r', label=labels[0])
        plt.plot(t[:length],d2[:length], 'b', label=labels[1])
    else:
        plt.plot(t, d1, 'r', label=labels[0])
        plt.plot(t, d2, 'b', label=labels[1])
    plt.title(name[0])
    plt.xlabel(name[1])
    plt.ylabel(name[2])
    plt.legend(loc=locs)
    plt.draw()

def original_train():
    nn_architecture = [
    {"input_dim": 16, "output_dim": 32, "activation": "relu"},
    {"input_dim": 32, "output_dim": 16, "activation": "relu"},
    {"input_dim": 16, "output_dim": 8, "activation": "relu"},
    {"input_dim": 8, "output_dim": 1, "activation": "linear"},]
    learning_rate = 0.0000001
    data = pd.read_csv('energy_efficiency_data.csv')
    heating = data.pop('Heating Load')
    data.pop('Cooling Load')
    data = pd.concat([data,pd.get_dummies(data['Orientation'], prefix='Orientation')],axis=1)
    data = pd.concat([data,pd.get_dummies(data['Glazing Area Distribution'], prefix='Glazing Area Distribution')],axis=1)
    data.drop(['Orientation'],axis=1, inplace=True)
    data.drop(['Glazing Area Distribution'],axis=1, inplace=True)
    my_train(data, heating, nn_architecture, learning_rate)

def for_ficture_selct_1():
    nn_architecture = [
    {"input_dim": 3, "output_dim": 32, "activation": "relu"},
    {"input_dim": 32, "output_dim": 16, "activation": "relu"},
    {"input_dim": 16, "output_dim": 8, "activation": "relu"},
    {"input_dim": 8, "output_dim": 1, "activation": "linear"},]
    learning_rate = 0.0000001
    data = pd.read_csv('energy_efficiency_data.csv')
    data.pop('Orientation')
    data.pop('Glazing Area Distribution')
    print(data.corr())
    data.pop('# Relative Compactness')
    data.pop('Wall Area')
    data.pop('Glazing Area')
    data.pop('Cooling Load')
    heating = data.pop('Heating Load')
    print("we choose higer corr such as Roof Area, Overall Height and Surface Aread\n****")
    my_train(data, heating, nn_architecture, learning_rate)

def for_ficture_selct_2():
    nn_architecture = [
    {"input_dim": 13, "output_dim": 32, "activation": "relu"},
    {"input_dim": 32, "output_dim": 16, "activation": "relu"},
    {"input_dim": 16, "output_dim": 8, "activation": "relu"},
    {"input_dim": 8, "output_dim": 1, "activation": "linear"},]
    learning_rate = 0.0000001
    data = pd.read_csv('energy_efficiency_data.csv')
    data = pd.concat([data,pd.get_dummies(data['Orientation'], prefix='Orientation')],axis=1)
    data = pd.concat([data,pd.get_dummies(data['Glazing Area Distribution'], prefix='Glazing Area Distribution')],axis=1)
    data.drop(['Orientation'],axis=1, inplace=True)
    data.drop(['Glazing Area Distribution'],axis=1, inplace=True)
    data.pop('# Relative Compactness')
    data.pop('Wall Area')
    data.pop('Glazing Area')
    data.pop('Cooling Load')
    heating = data.pop('Heating Load')
    print("if we think the cacategorical features are also import feature")
    print('So, we use Roof Area, Overall Height, Surface Aread, Orientation and Glazing Area Distribution\n****')
    my_train(data, heating, nn_architecture, learning_rate)

def for_ficture_selct_3():
    nn_architecture = [
    {"input_dim": 3, "output_dim": 32, "activation": "relu"},
    {"input_dim": 32, "output_dim": 16, "activation": "relu"},
    {"input_dim": 16, "output_dim": 8, "activation": "relu"},
    {"input_dim": 8, "output_dim": 1, "activation": "linear"},]
    learning_rate = 0.0000001
    data = pd.read_csv('energy_efficiency_data.csv')
    data.pop('Orientation')
    data.pop('Glazing Area Distribution')
    data.pop('Roof Area')
    data.pop('Surface Area')
    data.pop('Overall Height')
    data.pop('Cooling Load')
    heating = data.pop('Heating Load')
    print("we choose lower corr such as # Relative Compactness, Wall Area and Glazing Area\n****")
    my_train(data, heating, nn_architecture, learning_rate)

def my_train(data, heating, nn_architecture, learning_rate):
    x_train, x_test, y_train, y_test = my_split_function(data.to_numpy(), heating.to_numpy(), test_size=0.25)
    model = myNN(nn_architecture, learning_rate)
    model.train(x_train, np.array(y_train), x_test, np.array(y_test), 2000, batch_size=16)
    y_predict_train = model.predict(x_train.T)
    y_predict_test = model.predict(x_test.T)
    show_graph(y_predict_train, y_train, ['y_predict', 'y_train'], ['prediction for training data', '#th case', 'heating load'])
    show_graph(y_predict_test, y_test, ['y_predict', 'y_test'], ['prediction for testing data', '#th case', 'heating load'])
    train_loss, test_loss = model.get_loss_history()
    show_graph(train_loss, test_loss, ['train_loss', 'test_loss'], ['training curve', 'Epochs', ''], locs='upper right')
    plt.show(block=False)
    _ = input("Press [enter] to continue.")
    plt.close('all')

def regression():
    print('****\nOriginal training\n****')
    original_train()
    print('****\nWith feature selection')
    print("1st kind of selection")
    for_ficture_selct_1()
    print("****\n2nd kind of selection")
    for_ficture_selct_2()
    print('****\n3rd kind of selection')
    for_ficture_selct_3()

if __name__ == '__main__':
    regression()