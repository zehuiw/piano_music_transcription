'''###### TRAIN 1: DNN - 3 layers - 150 unis per layer ######'''

import numpy as np
import os
import os.path
import sys
# import matplotlib.pyplot as plt

# We need to set the random seed so that we get ther same results with the same parameters
np.random.seed(400)  

# Import keras main libraries
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.regularizers import l2

mini_batch_size, num_epochs = 100, 50
input_size = 252
number_units = 256
number_layers = 3
number_classes = 88
size_samples = 100
data_directory = sys.argv[1] #.npy
weights_dir = sys.argv[2] #.hdf5
model_type = sys.argv[3] #cnn/dnn/rnn
X = []
y = []

# num_test_batches = len([name for name in os.listdir(data_directory )])/2
num_test_batches = 1

def slide_cnn_window(X_train):
    window_size = 7
    length, feature_length = X_train.shape
    # print 'before sliding:', X_train.shape
    padding = np.pad(X_train, ((window_size / 2,window_size / 2), (0,0)), 'constant', constant_values = 0)
    # print 'padding:', padding.shape
    for k in range(0, -window_size , -1):
        tmp = np.roll(padding, k, axis = 0)
        # print 'after rolling:', tmp.shape
        if k == 0:
            X_train = tmp[:length, :]
        else:
            X_train = np.concatenate((X_train, tmp[:length, :]), axis = 1)
        # print 'X_train:', X_train.shape
    X_train = np.reshape(X_train, (length, -1, feature_length, 1))
    return X_train

print 'Loading test data'
for i in range(num_test_batches):
    x_file_name =  "input_wave_file_X.npy"
    y_file_name =  "input_wave_file_y.npy"
    print "Batching..." + str(i) + x_file_name
    X_test = np.array(np.load(data_directory + str(i) + x_file_name ))
    y_test = np.array(np.load(data_directory + str(i) + y_file_name ))
    if i == 0:
        X = X_test
        y = y_test
    else:
        X = np.concatenate((X,X_test), axis = 0)
        y = np.concatenate((y,y_test), axis = 0)
    if model_type == 'cnn':
        X = slide_cnn_window(X)

# Load the model 
model = load_model(weights_dir + "weights.hdf5")
TP = 0
FP = 0
FN = 0

print "Predicting model. . . "
predictions = model.predict(X, batch_size=mini_batch_size, verbose = 1) 
predictions = np.array(predictions).round()
predictions[predictions > 1] = 1
np.save('{}predictions'.format(data_directory), predictions)
print "\nCalculating accuracy. . ."
TP = np.count_nonzero(np.logical_and( predictions == 1, y == 1 ))
FN = np.count_nonzero(np.logical_and( predictions == 0, y == 1 ))
FP = np.count_nonzero(np.logical_and( predictions == 1, y == 0 ))
print TP, FN, FP
if (TP + FN) > 0:
    R = TP/float(TP + FN)
    P = TP/float(TP + FP)
    A = 100*TP/float(TP + FP + FN)
    if P == 0 and R == 0:
	F = 0
    else: 
	F = 100*2*P*R/(P + R)
else: 
    A = 0
    F = 0
    R = 0
    P = 0

print '\n F-measure pre-processed: '
print F
print '\n Accuracy pre-processed: '
print A

