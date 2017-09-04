""" This script generates train sets from several building data
"""
from __future__ import print_function, division
from warnings import warn, filterwarnings

import numpy as np
import urllib
import os

mmax = 5000 # Assumed maximum consumption value
def opends(building, meter):
    '''Opens dataset of synthetic data from Neural NILM

    Parameters
    ----------
    building : The integer id of the building
    meter : The string key of the meter

    Returns: np.arrays of data in the following order: main data, meter data
    '''
    if not os.path.exists("dataset"):
        download_dataset()

    path = "dataset/ground_truth_and_mains/"
    main_filename = "{}building_{}_mains.csv".format(path, building)
    meter_filename = "{}building_{}_{}.csv".format(path, building, meter)
    mains = np.genfromtxt(main_filename)
    meter = np.genfromtxt(meter_filename)
    mains = mains / mmax
    meter = meter / mmax
    up_limit = min(len(mains),len(meter))
    return mains[:up_limit], meter[:up_limit]

def download_dataset():
    print("Downloading dataset for the first time")
    testfile = urllib.URLopener()
    os.makedirs("dataset")
    testfile.retrieve("http://jack-kelly.com/files/neuralnilm/NeuralNILM_data.zip", "dataset/ds.zip")
    import zipfile

    zip_ref = zipfile.ZipFile('dataset/ds.zip', 'r')
    zip_ref.extractall('dataset')
    zip_ref.close()
    os.remove("dataset/ds.zip")
    os.makedirs("dataset/trainsets")


def gen_batch(mainchunk, meterchunk, batch_size, index,window_size):
    '''Generates batches from dataset

    Parameters
    ----------
    index : the index of the batch
    '''
    w = window_size
    offset = index*batch_size
    X_batch = np.array([ mainchunk[i+offset:i+offset+w]
                        for i in range(batch_size) ])

    Y_batch = meterchunk[w-1+offset:w-1+offset+batch_size]
    X_batch = np.reshape(X_batch, (len(X_batch), w ,1))

    return X_batch, Y_batch

if __name__ == "__main__":
    input_window = 50 # Lookback parameter
    train_size = 80639 # The length of the trainset
    key_name = 'fridge'  # The string ID of the meter
    house_keys = [1,2,4] # The buildings used for the train set
    all_x_train = np.empty((train_size*len(house_keys),input_window,1))
    all_y_train = np.empty((train_size*len(house_keys),))

    # Get data from every building
    for i,key in enumerate(house_keys):
        mains, meter = opends(key,key_name)
        X_train = mains[:train_size+input_window]
        y_train = meter[:train_size+input_window]
        X_batch, Y_batch = gen_batch(X_train, y_train, len(X_train)-input_window, 0, input_window)
        all_x_train[i*train_size:(i+1)*train_size] = X_batch
        all_y_train[i*train_size:(i+1)*train_size] = Y_batch


    np.save('dataset/trainsets/X-{}'.format(key_name),all_x_train)
    np.save('dataset/trainsets/Y-{}'.format(key_name),all_y_train)
