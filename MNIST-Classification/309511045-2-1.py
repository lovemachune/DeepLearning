import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import utils as np_utils
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

class my_callback(Callback):
    def __init__(self, data):
        self.x = data[0]
        self.y = data[1]
        self.acc = []
    def on_epoch_end(self, epoch, logs=None):
        predict = self.model.predict(self.x)
        acc = tf.metrics.categorical_accuracy(self.y,predict).numpy().mean()
        self.acc.append(acc)
    def get_history(self):
        return self.acc

def draw_learning_curve(train_acc, val_acc, test_acc):
    t = range(len(train_acc))
    plt.plot(t, train_acc, label='train')
    plt.plot(t, val_acc, label='val')
    plt.plot(t, test_acc, label='test')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy rate')
    plt.legend()
    plt.show()

def draw_loss(train_loss):
    t = range(len(train_loss))
    plt.plot(t, train_loss, label='Cross entropy')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def draw_distributions(layers):
    for i, layer in enumerate(layers):
        name = layer.name
        if not layer.get_weights() :
            continue
        weight, bias = layer.get_weights()
        plt.figure()
        plt.subplot(121)
        plt.hist(weight.flatten(), 100)
        plt.xlabel(name + '_weight')
        plt.ylabel('number')
        plt.subplot(122)
        plt.hist(bias.flatten(), 100)
        plt.xlabel(name + '_bias')
        plt.ylabel('number')
    plt.show()

def false_predict(x_test, Y_test, predict):
    plt.figure()
    for index, my_pred in enumerate(predict):
        pred_label = np.argmax(my_pred)
        if pred_label != Y_test[index]:
            plt.imshow(x_test[index], cmap='gray')
            img1 = x_test[index][np.newaxis]
            plt.title('label: %d, pred: %d' % (Y_test[index], pred_label))
            break
    cor_index = np.where(Y_test == pred_label)[0][0]
    plt.figure()
    plt.imshow(x_test[cor_index], cmap='gray')
    img2 = x_test[cor_index][np.newaxis]
    plt.title('label: %d, pred: %d' % (Y_test[cor_index], np.argmax(predict[cor_index])))
    return (img1, img2)

def feature_map(model, img):
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = Model(inputs=model.input,outputs=layer_outputs)
    activations = activation_model.predict(img)
    plt.figure()
    count = 0
    for index, layer in enumerate(model.layers):
        if layer.name[:3] == 'con':
            for i in range(5):
                plt.subplot(2, 5, i+1+count)
                feature = activations[index][0,:,:,i]
                plt.imshow(feature, cmap='gray')
            count += 5

def without_l2(input_size, kernel_size, stride):
    model = Sequential()  
    model.add(Conv2D(filters=32,  kernel_size=kernel_size, strides=stride, padding='same',  input_shape=input_size,  activation='relu'))   
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=32,  kernel_size=kernel_size, strides=stride, padding='same',  activation='relu'))   
    model.add(MaxPooling2D(pool_size=(2,2))) 
    model.add(Flatten()) 
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    model.summary()
    return model

def with_l2(input_size, kernel_size, stride):
    l2_rate = 0.00001
    model = Sequential()  
    model.add(Conv2D(filters=32,  kernel_size=kernel_size, strides=stride, padding='same',  input_shape=input_size,  activation='relu', kernel_regularizer=regularizers.l2(l=l2_rate)))   
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=32,  kernel_size=kernel_size, strides=stride, padding='same',  activation='relu', kernel_regularizer=regularizers.l2(l=l2_rate)))   
    model.add(MaxPooling2D(pool_size=(2,2))) 
    model.add(Flatten()) 
    model.add(Dense(10, activation='softmax', kernel_regularizer=regularizers.l2(l=l2_rate)))
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    model.summary()
    return model

def my_train(model, epochs, x_train, y_train, x_test, y_test, x_val, y_val):
    train_loss = []
    train_acc = []
    val_acc = []
    test_acc = []
    callback = my_callback([x_test, y_test])
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=64, shuffle=True, validation_data=(x_val, y_val), callbacks=[callback])
    train_loss = history.history['loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    test_acc = callback.get_history()
    return (train_loss, train_acc, val_acc, test_acc)

def get_data():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    x_train = X_train.reshape(-1, 28, 28, 1)/255
    x_test = X_test.reshape(-1, 28, 28, 1)/255
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)
    x_train = x_train[indices]
    Y_train = Y_train[indices]
    y_train = np_utils.to_categorical(Y_train)
    y_test = np_utils.to_categorical(Y_test)
    x_val = x_train[:5000]
    x_train = x_train[5000:]
    y_val = y_train[:5000]
    y_train = y_train[5000:]
    return (x_train, y_train, x_test, y_test, x_val, y_val, Y_test)

def show_graph_result(model, train_loss, train_acc, val_acc, test_acc, x_test, Y_test):
    draw_learning_curve(train_acc, val_acc, test_acc)
    draw_loss(train_loss)
    draw_distributions(model.layers)
    img1, img2 = false_predict(x_test, Y_test, model.predict(x_test))
    feature_map(model, img1)
    feature_map(model, img2)
    plt.show()

def normal_model(kernel_size = (3,3), stride = (1,1), flag = True):
    input_size = (28,28,1)
    print('kernel size = ', end='')
    print(kernel_size, end=', ')
    print('stride = ', end='')
    print(stride, end='\n\n')
    epochs = 100
    (x_train, y_train, x_test, y_test, x_val, y_val, Y_test) = get_data()
    model = without_l2(input_size, kernel_size, stride)
    train_loss, train_acc, val_acc, test_acc = my_train(model, epochs, x_train, y_train, x_test, y_test, x_val, y_val)
    if flag:
        show_graph_result(model, train_loss, train_acc, val_acc, test_acc, x_test, Y_test)
    _, accuracy = model.evaluate(x_test, y_test)
    print("\nAcc : %lf\n" % accuracy)
    print('------------------')

def diff_kernel_model():
    for i in range(2, 4):
        kernel_size = (i*2+1, i*2+1)
        stride = (1,1)
        normal_model(kernel_size = kernel_size, stride = stride, flag = False)
    for i in range(2,4):
        kernel_size = (3, 3)
        stride = (i, i)
        normal_model(kernel_size = kernel_size, stride = stride, flag = False)

def L2_model():
    input_size = (28,28,1)
    kernel_size = (3,3)
    stride = (1,1)
    epochs = 100
    (x_train, y_train, x_test, y_test, x_val, y_val, Y_test) = get_data()
    model = with_l2(input_size, kernel_size, stride)
    train_loss, train_acc, val_acc, test_acc = my_train(model, epochs, x_train, y_train, x_test, y_test, x_val, y_val)
    show_graph_result(model, train_loss, train_acc, val_acc, test_acc, x_test, Y_test)
    _, accuracy = model.evaluate(x_test, y_test)
    print("\nAcc : %lf\n" % accuracy)
    print('------------------')

def mnist_cnn():
    print('\n\n**** Normal model ****\n')
    normal_model()
    print("\n\n**** Train with different kernel size and stride ****\n")
    diff_kernel_model()
    print('\n\n**** Model with L2 ****\n')
    L2_model()

if __name__ == '__main__':
    mnist_cnn()