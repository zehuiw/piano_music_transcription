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
from train import slide_cnn_window

mini_batch_size, num_epochs = 100, 50
input_size = 252
number_units = 256
number_layers = 3
number_classes = 88
size_samples = 100
data_directory = sys.argv[1]
weights_dir = sys.argv[2]
dataset_type = sys.argv[3] #test on train/dev/test
model_type = sys.argv[4] #cnn/dnn/rnn
X = []
y = []

num_test_batches = len([name for name in os.listdir(data_directory )])/2

print 'Loading test data'
for i in range(num_test_batches):
    x_file_name = dataset_type + "_X.npy"
    y_file_name = dataset_type + "_y.npy"
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
np.save('{}predictions'.format(weights_dir), predictions)
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


main_data = open(weights_dir + "Accuracy.lst", "w")
main_data.write("R-pre = " + str("%.6f" % R) + "\n")
main_data.write("P-pre = " + str("%.6f" % P) + "\n")
main_data.write("A-pre = " + str("%.6f" % A) + "\n")
main_data.write("F-pre = " + str("%.6f" % F) + "\n")

main_data.close()

