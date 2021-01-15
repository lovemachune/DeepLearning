import os
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow import keras



def draw_total_loss(total_loss):
    t = range(len(total_loss[0]))
    plt.plot(t, total_loss[0], label='1024,100')
    plt.plot(t, total_loss[1], label='1024,50')
    plt.plot(t, total_loss[2], label='512,100')
    plt.plot(t, total_loss[3], label='512,50')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

def draw_loss(train_loss):
    t = range(len(train_loss))
    plt.plot(t, train_loss, label='BPC')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def draw_learning_curve(train_acc, val_acc):
    t = range(len(train_acc))
    plt.plot(t, train_acc, label='train')
    plt.plot(t, val_acc, label='val')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy rate')
    plt.legend()
    plt.show()

def data_process(seq_length, text_as_int, BATCH_SIZE = 1, BUFFER_SIZE = 10000):
    # Create training examples / targets
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
    dataset = sequences.map(split_input_target)
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    DATASET_SIZE = len(list(dataset))
    train_size = int(0.9 * DATASET_SIZE)
    val_size = int(0.1 * DATASET_SIZE)
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)
    return train_dataset, val_dataset

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

def build_RNN_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim,batch_input_shape=[batch_size, None]))
    model.add(tf.keras.layers.SimpleRNN(rnn_units, return_sequences=True, stateful=True))
    model.add(tf.keras.layers.Dense(vocab_size))
    return model

def build_LSTM_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim,batch_input_shape=[batch_size, None]))
    model.add(tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True))
    model.add(tf.keras.layers.Dense(vocab_size))
    return model

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def generate_text(model, start_string):
    path_to_file = 'shakespeare.txt'
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
    vocab = sorted(set(text))
    char2idx = {u:i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    text_as_int = np.array([char2idx[c] for c in text])
    num_generate = 1000

    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    temperature = 1

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))

def RNN():
    path_to_file = 'shakespeare.txt'
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
    vocab = sorted(set(text))
    char2idx = {u:i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    text_as_int = np.array([char2idx[c] for c in text])
    total_loss = []
    embedding_dim = 256
    rnn_units = 1024
    seq_len = 100
    BATCH_SIZE = 1
    EPOCHS = 10

    for i in range (2):
        seq_len = 100
        for j in range (2):
            print('%d,%d' % (rnn_units, seq_len))
            train_dataset, val_dataset = data_process(seq_len, text_as_int, BATCH_SIZE = 1, BUFFER_SIZE = 10000)
            model = build_RNN_model(vocab_size=len(vocab),embedding_dim=embedding_dim,rnn_units=rnn_units,batch_size=BATCH_SIZE)
            checkpoint_dir = './training_RNN_' + str(i) + str(j)
            checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,save_weights_only=True)
            model.compile(optimizer='adam', loss=loss)
            model.summary()
            history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
            total_loss.append(history.history['loss'])
            seq_len = int(seq_len/2)
            print("Training error  rate : %lf" % model.evaluate(train_dataset))
            print("Validation error rate : %lf" % model.evaluate(val_dataset))
            print(generate_text(model, start_string=u"JULIET"))
            print('-----------------------')
        rnn_units = int(rnn_units/2)
        
    draw_loss(total_loss[2])
    draw_total_loss(total_loss)
    model = keras.models.load_model('RNN_model.h5')
    print('------------------------')
    print(generate_text(model, start_string=u"JULIET"))
    print('------------------------')

def LSTM():
    path_to_file = 'shakespeare.txt'
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
    vocab = sorted(set(text))
    char2idx = {u:i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    text_as_int = np.array([char2idx[c] for c in text])
    total_loss = []
    embedding_dim = 256
    rnn_units = 1024
    seq_len = 100
    BATCH_SIZE = 1
    EPOCHS = 10
    for i in range (2):
        seq_len = 100
        for j in range (2):
            print('%d,%d' % (rnn_units, seq_len))
            train_dataset, val_dataset = data_process(seq_len, text_as_int, BATCH_SIZE = 1, BUFFER_SIZE = 10000)
            model = build_LSTM_model(vocab_size=len(vocab),embedding_dim=embedding_dim,rnn_units=rnn_units,batch_size=BATCH_SIZE)
            checkpoint_dir = './training_LSTM_' + str(i) + str(j)
            checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,save_weights_only=True)
            model.compile(optimizer='adam', loss=loss)
            model.summary()
            history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
            total_loss.append(history.history['loss'])
            seq_len = int(seq_len/2)
            print("Training error  rate : %lf" % model.evaluate(train_dataset))
            print("Validation error rate : %lf" % model.evaluate(val_dataset))
            print(generate_text(model, start_string=u"JULIET"))
            print('-----------------------')
        rnn_units = int(rnn_units/2)

    draw_loss(total_loss[2])
    draw_total_loss(total_loss)
    model = keras.models.load_model('LSTM_model.h5')
    print('------------------------')
    print(generate_text(model, start_string=u"JULIET"))
    print('------------------------')

if __name__ == '__main__':
    RNN()
    LSTM()
