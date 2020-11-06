import pandas as pd
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from mpl_toolkits.mplot3d import Axes3D

class myNN():
    def __init__(self, architecture, learning_rate):
        self.architecture = architecture
        self.learning_rate = learning_rate
        self.params = {}
        self.init_layers()
        self.train_loss_history = []
        self.test_loss_history = []
        self.latent_train_feature = []
        self.latent_test_feature = []
        
    def init_layers(self):
        for index, layer in enumerate(self.architecture):
            layer_index = index + 1
            layer_input_size = layer['input_dim']
            layer_output_size = layer['output_dim']
            self.params['W' + str(layer_index)] = np.random.randn(layer_output_size, layer_input_size)*0.1
            self.params['b' + str(layer_index)] = np.random.randn(layer_output_size, 1)*0.1
            
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
        Z_curr = np.dot(W_curr, A_prev) + b_curr
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
        n = y.shape[1]
        loss = -1/n*(np.dot(y, np.log(y_predict).T)+np.dot(1 - y, np.log(1-y_predict).T))
        return np.squeeze(loss)
    
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
        dA_prev = -(np.divide(y, y_predict) - np.divide(1-y, 1-y_predict))
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
            x_train_s, y_train_s = shuffle(x_train, y_train)
            for j in range(0 ,x_train.shape[0], batch_size):
                start = j
                end = j+batch_size
                if end >= x_train.shape[0]:
                    end = x_train.shape[0]-1
                x_min = x_train_s[start:end]
                y_min = y_train_s[start:end]
                y_predict, memory = self.forward_propagation(x_min.T)
                loss = self.get_loss_value(np.array([y_min]), y_predict)
                grads_values = self.back_propagation(y_predict, np.array([y_min]), memory)
                self.update(grads_values)
                
            y_train_predict, memory_train = self.predict(x_train.T)
            y_test_predict, memory_test = self.predict(x_test.T)
            train_loss = self.get_loss_value(np.array([y_train]), y_train_predict)
            test_loss = self.get_loss_value(np.array([y_test]), y_test_predict)
            self.train_loss_history.append(train_loss)
            self.test_loss_history.append(test_loss)
            train_acc = self.get_accuracy(y_train, y_train_predict)
            test_acc = self.get_accuracy(y_test, y_test_predict)
            if (i+1)%10 == 0:
                layer_index = len(self.architecture)-1
                A_train=  memory_train["Z"+str(layer_index)]
                A_test=  memory_test["Z"+str(layer_index)]
                self.latent_train_feature.append(A_train)
                self.latent_test_feature.append(A_test)
            if (i+1) % 100 == 0 and i!=0:
                print("Eposh : %4d/%d || training loss : %.4lf || traing acc : %.4lf" % (i+1, epochs, train_loss, train_acc), end=" ")
                print("|| testing loss : %.4lf || testing acc : %.4lf" % (test_loss, test_acc))
    
    def show_parm(self):
        print(self.params)
    
    def predict(self, x):
        y_predict, memory = self.forward_propagation(x)
        return np.squeeze(y_predict), memory
    
    def get_loss_history(self):
        return self.train_loss_history, self.test_loss_history

    def get_accuracy(self, y_true, y_predict):
        y_predict[y_predict>0.5] = 1
        y_predict[y_predict<=0.5] = 0
        count = 0
        for v1, v2 in zip(y_true, y_predict):
            if v1 == v2:
                count +=1
        return count/len(y_predict)
    def get_latent_feature(self):
        return [self.latent_train_feature, self.latent_test_feature]

def my_split_function(data, label, test_size):
    random.seed(500)
    data, label = shuffle(data, label)
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

def model_1():
    nn_architecture = [
    {"input_dim": 34, "output_dim": 17, "activation": "relu"},
    {"input_dim": 17, "output_dim": 3, "activation": "relu"},
    {"input_dim": 3, "output_dim": 1, "activation": "sigmoid"},
    ]
    learning_rate = 0.0003
    my_train(nn_architecture, learning_rate)

def model_2():
    nn_architecture = [
    {"input_dim": 34, "output_dim": 17, "activation": "relu"},
    {"input_dim": 17, "output_dim": 2, "activation": "relu"},
    {"input_dim": 2, "output_dim": 1, "activation": "sigmoid"},
    ]
    learning_rate = 0.0005
    my_train(nn_architecture, learning_rate, flag=False)

def draw_2D(model, y_train):
    train_loss, test_loss = model.get_loss_history()
    show_graph(train_loss, test_loss, ['train_loss', 'test_loss'], ['training curve', 'Epochs', ''], locs='upper right')
    latent_feature = model.get_latent_feature()
    fig, ax = plt.subplots()
    scatter = ax.scatter(latent_feature[0][1][0], latent_feature[0][1][1], c=y_train)
    classes = ['Class 1', 'Class 2']
    legend = plt.legend(handles=scatter.legend_elements()[0], labels=classes)
    fig.add_artist(legend)
    ax.title.set_text('2D feature of 10 epochs')

    fig, ax = plt.subplots()
    scatter = ax.scatter(latent_feature[0][99][0], latent_feature[0][99][1], c=y_train)
    classes = ['Class 1', 'Class 2']
    legend = plt.legend(handles=scatter.legend_elements()[0], labels=classes)
    ax.add_artist(legend)
    ax.title.set_text('2D feature of 1000 epochs')

def draw_3D(model, y_train):
    train_loss, test_loss = model.get_loss_history()
    show_graph(train_loss, test_loss, ['train_loss', 'test_loss'], ['training curve', 'Epochs', ''], locs='upper right')
    latent_feature = model.get_latent_feature()
    fig, ax = plt.subplots()
    ax = plt.axes(projection='3d')
    scatter = ax.scatter(latent_feature[0][1][0], latent_feature[0][1][1], latent_feature[0][1][2], c=y_train)
    classes = ['Class 1', 'Class 2']
    legend = plt.legend(handles=scatter.legend_elements()[0], labels=classes)
    ax.title.set_text('3D feature of 10 epochs')

    latent_feature = model.get_latent_feature()
    fig, ax = plt.subplots()
    ax = plt.axes(projection='3d')
    scatter = ax.scatter(latent_feature[0][99][0], latent_feature[0][99][1], latent_feature[0][99][2], c=y_train)
    classes = ['Class 1', 'Class 2']
    legend = plt.legend(handles=scatter.legend_elements()[0], labels=classes)
    ax.title.set_text('3D feature of 1000 epochs')

def my_train(nn_architecture, learning_rate, flag=True):
    data = pd.read_csv('ionosphere_data.csv', header=None)
    label = data.pop(34)
    for index, value in enumerate(label):
        if value == 'g':
            label[index] = 0
        else:
            label[index] = 1
    label = label.to_numpy(dtype='int8')

    x_train, x_test, y_train, y_test = my_split_function(data, label, test_size=0.2)
    model = myNN(nn_architecture, learning_rate)
    model.train(x_train, y_train, x_test, y_test, 1000, batch_size=16)
    if flag :
        draw_3D(model, y_train)
    else:
        draw_2D(model, y_train)
    plt.show(block=False)
    _ = input("Press [enter] to continue.")
    plt.close('all')


def classify():
    print('The dim of latent layer is 3')
    model_1()
    print('The dim of latent layer is 2')
    model_2()
    

if __name__ == '__main__':
    classify()