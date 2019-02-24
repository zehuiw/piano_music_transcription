## CS230 Deep Learning project ##

import numpy as np
import os
import os.path
import sys

np.random.seed(400)  

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
from keras.regularizers import l2
from keras import callbacks
from keras.callbacks import History, ModelCheckpoint, EarlyStopping
import tensorflow as tf

tf.app.flags.DEFINE_string("data_directory", "../Preprocessing/toy/", "data directory")
tf.app.flags.DEFINE_string("weights_dir", "../weights/", "weights directory")
tf.app.flags.DEFINE_string("model", "dnn", 'model type: dnn/rnn/cnn')

tf.app.flags.DEFINE_integer('mini_batch_size', 100, 'mini batch size')
tf.app.flags.DEFINE_integer('num_epochs' ,200, 'num epochs')
tf.app.flags.DEFINE_integer('input_size' ,252, 'input size')
tf.app.flags.DEFINE_integer('number_units' ,256, 'number units')
tf.app.flags.DEFINE_integer('number_layers' ,3, 'number layers')
tf.app.flags.DEFINE_integer('number_classes' ,88, 'number classes')
tf.app.flags.DEFINE_integer('best_accuracy' ,0, 'best acc')
tf.app.flags.DEFINE_integer('sequence_len', 100, 'rnn sequence length')
tf.app.flags.DEFINE_float('dropout', 0.2, 'dropout')
tf.app.flags.DEFINE_string('loss', 'mean_squared_error', 'loss function') # mean_squared_error, binary_crossentropy


FLAGS = tf.app.flags.FLAGS


def load_data(data_directory):
    print 'Load validation data...'
    X_val = np.load(data_directory + "dev/" + str(0) + "dev_X.npy" )
    y_val = np.load(data_directory + "dev/" + str(0) + "dev_y.npy" )

    # Count the number of files in the training folder 
    num_tr_batches = len([name for name in os.listdir(data_directory + "train/")])/2

    print 'Loading training data'
    for i in range(num_tr_batches):
        print "Batching..." + str(i) + "train_X.npy"
        X_train = np.array(np.load(data_directory + "train/" + str(i) + "train_X.npy" ))
        y_train = np.array(np.load(data_directory + "train/" + str(i) + "train_y.npy" ))
        if FLAGS.model == 'rnn':
            truncation = (X_train.shape[0] // 100) * 100
            X_train = np.reshape(X_train[0:truncation,:], (X_train.shape[0]//FLAGS.sequence_len, FLAGS.sequence_len, X_train.shape[1]))
            truncation = (y_train.shape[0] // 100) * 100
            y_train = np.reshape(y_train[0:truncation,:], (y_train.shape[0]//FLAGS.sequence_len, FLAGS.sequence_len, y_train.shape[1]))          
        if i == 0:
            X = X_train
            y = y_train
        else:
            X = np.concatenate((X,X_train), axis = 0)
            y = np.concatenate((y,y_train), axis = 0)
    print X.shape

    if FLAGS.model == 'rnn':
        truncation = (X_val.shape[0] // 100) * 100
        X_val = np.reshape(X_val[0:truncation,:], (X_val.shape[0]//FLAGS.sequence_len, FLAGS.sequence_len, X_val.shape[1]))
        truncation = (y_val.shape[0] // 100) * 100
        y_val = np.reshape(y_val[0:truncation,:], (y_val.shape[0]//FLAGS.sequence_len, FLAGS.sequence_len, y_val.shape[1]))


    return X_val, y_val, X, y

def build_dnn_model(model):
    print "Adding 1st layer of {} units".format(FLAGS.number_units)
    model.add(Dense(FLAGS.number_units, input_shape=(FLAGS.input_size,), kernel_initializer='normal', activation='relu'))
    model.add(Dropout(FLAGS.dropout))
    for i in range(FLAGS.number_layers-1):
        print "Adding %d" % (i+2) + "th layer of %d" % FLAGS.number_units + " units"
        model.add(Dense(FLAGS.number_units, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(FLAGS.dropout))

    print " Adding classification layer"
    model.add(Dense(FLAGS.number_classes, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss=FLAGS.loss, optimizer='adam', metrics=['accuracy'])

def build_rnn_model(model):
    model.add(LSTM(FLAGS.number_units, input_shape=(FLAGS.sequence_len, FLAGS.input_size), return_sequences = "True",kernel_initializer='normal', activation='tanh'))
    model.add(Dropout(FLAGS.dropout))
    for i in range(FLAGS.number_layers - 1):
        print "Adding {} units".format(str(i + 2) + "layer of" + str(FLAGS.number_units))
        model.add(LSTM(FLAGS.number_units,return_sequences = "True",kernel_initializer='normal', activation='tanh'))
        model.add(Dropout(FLAGS.dropout))
    print " Adding classification layer"
    model.add(Dense(FLAGS.number_classes, kernel_initializer='normal', activation='relu'))
    model.compile(loss=FLAGS.loss, optimizer='adam', metrics=['accuracy'])


def train(data_directory, weights_dir):
    X_val, y_val, X, y = load_data(data_directory)
    model = Sequential()
    history = History()
    if FLAGS.model == 'rnn':
        build_rnn_model(model)
    elif FLAGS.model == 'dnn':
        build_dnn_model(model)
    else:
        Abort("model name error")

    checkpointer = ModelCheckpoint(filepath= weights_dir + "weights.hdf5", verbose=1, save_best_only=True)
    early = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')

    training_log = open(weights_dir + "Training.log", "w")
    print 'Train . . .'

    save = model.fit(X, y, batch_size=FLAGS.mini_batch_size,epochs = FLAGS.num_epochs,validation_data=(X_val, y_val),verbose=1, callbacks=[checkpointer,early])
    training_log.write(str(save.history) + "\n")
    training_log.close()

def main():
    train(FLAGS.data_directory, FLAGS.weights_dir)


if __name__=="__main__":
    main()
