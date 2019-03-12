## CS230 Deep Learning project ##

import numpy as np
import os
import os.path
import sys

np.random.seed(400)  
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, LSTM, Conv2D, MaxPooling2D, Flatten
from keras import regularizers
from keras import callbacks
from keras.callbacks import History, ModelCheckpoint, EarlyStopping
import tensorflow as tf

tf.app.flags.DEFINE_string("data_directory", "../Preprocessing/toy/", "data directory")
tf.app.flags.DEFINE_string("weights_dir", "../weights/", "weights directory")
tf.app.flags.DEFINE_string("model", "", 'model type: dnn/rnn/cnn')

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
tf.app.flags.DEFINE_float('l2', 0.0, 'l2 regularization')
tf.app.flags.DEFINE_float('learning_rate', 0.0005, 'lr')

tf.app.flags.DEFINE_integer('window_size', 7, 'window size of cnn')
tf.app.flags.DEFINE_integer('num_filters', 50, 'number of filters per con layer')
tf.app.flags.DEFINE_integer('num_hidden_1', 1000, 'number of hidden units in FC')
tf.app.flags.DEFINE_integer('num_hidden_2', 200, 'number of hidden units in FC')
tf.app.flags.DEFINE_integer('pooling_size', 3, 'conv pooling window size')
tf.app.flags.DEFINE_string('checkpoint','', 'saved model dir') 
FLAGS = tf.app.flags.FLAGS

def slide_cnn_window(X_train):
    length, feature_length = X_train.shape
    # print 'before sliding:', X_train.shape
    padding = np.pad(X_train, ((FLAGS.window_size / 2,FLAGS.window_size / 2), (0,0)), 'constant', constant_values = 0)
    # print 'padding:', padding.shape
    for k in range(0, -FLAGS.window_size , -1):
        tmp = np.roll(padding, k, axis = 0)
        # print 'after rolling:', tmp.shape
        if k == 0:
            X_train = tmp[:length, :]
        else:
            X_train = np.concatenate((X_train, tmp[:length, :]), axis = 1)
        # print 'X_train:', X_train.shape
    X_train = np.reshape(X_train, (length, -1, feature_length, 1))
    return X_train

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

        # if FLAGS.model == 'cnn':
        #     X_train = slide_cnn_window(X_train)
        #     print 'training X_shape: ', X_train.shape

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

    if FLAGS.model == 'cnn':
        X_val = slide_cnn_window(X_val)
        print 'shape of X_val:', X_val.shape

    return X_val, y_val, X, y

def build_dnn_model(model):
    print "Adding 1st layer of {} units".format(FLAGS.number_units)
    model.add(Dense(FLAGS.number_units, input_shape=(FLAGS.input_size,), kernel_initializer='normal', activation='relu',  kernel_regularizer=regularizers.l2(FLAGS.l2)))
    model.add(Dropout(FLAGS.dropout))
    for i in range(FLAGS.number_layers-1):
        print "Adding %d" % (i+2) + "th layer of %d" % FLAGS.number_units + " units"
        model.add(Dense(FLAGS.number_units, kernel_initializer='normal', activation='relu', kernel_regularizer=regularizers.l2(FLAGS.l2)))
        model.add(Dropout(FLAGS.dropout))

    print " Adding classification layer"
    model.add(Dense(FLAGS.number_classes, kernel_initializer='normal', activation='sigmoid', kernel_regularizer=regularizers.l2(FLAGS.l2)))
    # Compile model
    model.compile(loss=FLAGS.loss, optimizer='adam', metrics=['accuracy'])

def build_cnn_model(model, opt):
    model.add(Conv2D(FLAGS.num_filters, kernel_size=(5,25), activation='relu', input_shape=(FLAGS.window_size, FLAGS.input_size, 1), kernel_regularizer=regularizers.l2(FLAGS.l2)))
    model.add(Dropout(FLAGS.dropout))
    model.add(MaxPooling2D(pool_size=(1, FLAGS.pooling_size)))

    model.add(Conv2D(FLAGS.num_filters, kernel_size=(3,5), activation='relu', kernel_regularizer=regularizers.l2(FLAGS.l2)))
    model.add(Dropout(FLAGS.dropout))
    model.add(MaxPooling2D(pool_size=(1, FLAGS.pooling_size)))

    model.add(Flatten())

    model.add(Dense(FLAGS.num_hidden_1, activation='relu', kernel_regularizer=regularizers.l2(FLAGS.l2)))
    model.add(Dropout(FLAGS.dropout))

    model.add(Dense(FLAGS.num_hidden_2,  activation='relu', kernel_regularizer=regularizers.l2(FLAGS.l2)))
    model.add(Dropout(FLAGS.dropout))

    model.add(Dense(FLAGS.number_classes, activation='sigmoid', kernel_regularizer=regularizers.l2(FLAGS.l2)))

    model.compile(loss=FLAGS.loss, optimizer=opt, metrics=['accuracy'])


def build_rnn_model(model):
    model.add(LSTM(FLAGS.number_units, input_shape=(FLAGS.sequence_len, FLAGS.input_size), return_sequences = "True",kernel_initializer='normal', activation='tanh', kernel_regularizer=regularizers.l2(FLAGS.l2)))
    model.add(Dropout(FLAGS.dropout))
    for i in range(FLAGS.number_layers - 1):
        print "Adding {} units".format(str(i + 2) + "layer of" + str(FLAGS.number_units))
        model.add(LSTM(FLAGS.number_units,return_sequences = "True",kernel_initializer='normal', activation='tanh', kernel_regularizer=regularizers.l2(FLAGS.l2)))
        model.add(Dropout(FLAGS.dropout))
    print " Adding classification layer"
    model.add(Dense(FLAGS.number_classes, kernel_initializer='normal', activation='relu', kernel_regularizer=regularizers.l2(FLAGS.l2)))
    model.compile(loss=FLAGS.loss, optimizer='adam', metrics=['accuracy'])

# handle training data that cannot fit in memory
def generator(features, labels, batch_size):
    # features already padded
    print 'padded features:', features.shape
    while True:
        for i in range(FLAGS.window_size / 2, FLAGS.window_size / 2 + features.shape[0], batch_size):
            features_batch = features[i - FLAGS.window_size / 2 : i + batch_size + FLAGS.window_size / 2, : ]
            features_batch = slide_cnn_window(features_batch)
            # print features_batch.shape
            features_batch = features_batch[FLAGS.window_size / 2 : - (FLAGS.window_size/ 2), :, :, :]
            # print 'batch:', features_batch.shape, labels[i : i + batch_size, :].shape
            # print ' idx:', i, i + batch_size, features_batch.shape, 
            yield features_batch, labels[i - FLAGS.window_size / 2: i - FLAGS.window_size / 2 + batch_size, :]

def train(data_directory, weights_dir):
    X_val, y_val, X, y = load_data(data_directory)
    model = Sequential()
    history = History()
    opt = Adam(lr=FLAGS.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    if FLAGS.checkpoint != '':
        model = load_model(FLAGS.checkpoint + "weights.hdf5")
    else:
        if FLAGS.model == 'rnn':
            build_rnn_model(model)
        elif FLAGS.model == 'dnn':
            build_dnn_model(model)
        elif FLAGS.model == 'cnn':
            build_cnn_model(model, opt)
        else:
            sys.exit('aa! errors!')

    # print model.summary()

    checkpointer = ModelCheckpoint(filepath= weights_dir + "weights.hdf5", verbose=1, save_best_only=True)
    early = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')

    training_log = open(weights_dir + "Training.log", "w")
    print 'Train . . .'
    if (FLAGS.model != 'cnn'): 
        save = model.fit(X, y, batch_size=FLAGS.mini_batch_size,epochs = FLAGS.num_epochs,validation_data=(X_val, y_val),verbose=1, callbacks=[checkpointer,early])
    else:
        # X: (None, feature_len), 
        # X_val: (None, window_size, feature_len, 1)
        print 'samples_per_epoch:', X.shape[0] / FLAGS.mini_batch_size
        padding = np.pad(X, ((FLAGS.window_size / 2,FLAGS.window_size / 2), (0,0)), 'constant', constant_values = 0)
        save = model.fit_generator(generator(padding, y, FLAGS.mini_batch_size), samples_per_epoch=X.shape[0] / FLAGS.mini_batch_size, epochs=FLAGS.num_epochs, validation_data=(X_val, y_val), verbose=1, callbacks=[checkpointer, early])

    training_log.write(str(save.history) + "\n")
    training_log.close()

def main():
    train(FLAGS.data_directory, FLAGS.weights_dir)

if __name__=="__main__":
    main()
