from __future__ import print_function, division
from warnings import warn, filterwarnings

import random
import sys
import numpy as np

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Conv1D, GRU, Bidirectional
from keras.utils import plot_model
from gen import opends, mmax, gen_batch

import metrics

def create_model(input_window):
    '''Creates and returns the Neural Network
    '''
    model = Sequential()

    # 1D Conv
    model.add(Conv1D(16, 4, activation="linear", input_shape=(input_window,1), padding="same", strides=1))

    #Bi-directional LSTMs
    model.add(Bidirectional(GRU(64, return_sequences=True), merge_mode='concat'))
    model.add(Bidirectional(GRU(128, return_sequences=False), merge_mode='concat'))

    # Fully Connected Layers
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer='adam')
    plot_model(model, to_file='model.png', show_shapes=True)

    return model

key_name = 'fridge' # The string ID of the meter
input_window = 50 # Lookback parameter
threshold = 60 # On Power Threshold
test_building = 5 # ID of the building to be used for testing

# ======= Training phase

# Open train sets
X_train = np.load("dataset/trainsets/X-{}.npy".format(key_name))
y_train = np.load("dataset/trainsets/Y-{}.npy".format(key_name))
model = create_model(input_window)

# Train model and save checkpoints
epochs_per_checkpoint = 1
for epochs in range(0,1,epochs_per_checkpoint):
    model.fit(X_train, y_train, batch_size=128, epochs=epochs_per_checkpoint, shuffle=True)
    model.save("SYNTH-LOOKBACK-{}-ALL-{}epochs-1WIN.h5".format(key_name, epochs+epochs_per_checkpoint),model)

# ======= Disaggregation phase
mains, meter= opends(test_building,key_name)
X_test = mains
y_test = meter*mmax

# Predict data
X_batch, Y_batch = gen_batch(X_test, y_test, len(X_test)-input_window, 0, input_window)
pred = model.predict(X_batch)* mmax
pred[pred<0] = 0
pred = np.transpose(pred)[0]
# Save results
np.save('pred.results',pred)

# Calculate and show metrics
print("============ Recall Precision Accurracy F1 {}".format(metrics.recall_precision_accuracy_f1(pred, Y_batch,threshold)))
print("============ relative_error_total_energy {}".format(metrics.relative_error_total_energy(pred, Y_batch)))
print("============ mean_absolute_error {}".format(metrics.mean_absolute_error(pred, Y_batch)))
