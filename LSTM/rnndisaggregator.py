from __future__ import print_function, division
from warnings import warn, filterwarnings

from matplotlib import rcParams
import matplotlib.pyplot as plt

import random
import sys
import pandas as pd
import numpy as np
import h5py

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Conv1D, LSTM, Bidirectional, Dropout
from keras.utils import plot_model

from nilmtk.utils import find_nearest
from nilmtk.feature_detectors import cluster
from nilmtk.disaggregate import Disaggregator
from nilmtk.datastore import HDFDataStore

class RNNDisaggregator(Disaggregator):
    '''Attempt to create a RNN Disaggregator

    Attributes
    ----------
    model : keras Sequential model
    meter_metadata : metadata of meter channel
    window_size : the size of window to use on the aggregate data
    mmax : the maximum value of the aggregate data
    gpu_mode : true if this is intended for gpu execution

    MIN_CHUNK_LENGTH : int
       the minimum length of an acceptable chunk
    '''

    def __init__(self, meter, window_size, gpu_mode=False):
        '''Initialize disaggregator
        DOES NOT TAKE INTO ACCOUNT EXISTANCE OF VAMPIRE POWER

        Parameters
        ----------
        window_size : the size of window to use on the aggregate data
        meter : a nilmtk.ElecMeter meter of the appliance to be disaggregated
        gpu_mode : true if this is intended for gpu execution
        '''
        self.MODEL_NAME = "LSTM"
        self.mmax = None
        self.window_size = window_size
        self.MIN_CHUNK_LENGTH = window_size
        self.gpu_mode = gpu_mode
        self.meter_metadata = meter
        self.model = self._create_model(self.window_size)

    def train(self, mains, meter, epochs=1, batch_size=128, **load_kwargs):
        '''Train

        Parameters
        ----------
        mains : a nilmtk.ElecMeter object for the aggregate data
        meter : a nilmtk.ElecMeter object for the meter data
        epochs : number of epochs to train
        **load_kwargs : keyword arguments passed to `meter.power_series()`
        '''

        main_power_series = mains.power_series(**load_kwargs)
        meter_power_series = meter.power_series(**load_kwargs)

        # Train chunks
        run = True
        mainchunk = next(main_power_series)
        meterchunk = next(meter_power_series)
        if self.mmax == None:
            self.mmax = mainchunk.max()

        while(run):
            mainchunk = self._normalize(mainchunk, self.mmax)
            meterchunk = self._normalize(meterchunk, self.mmax)

            self.train_on_chunk(mainchunk, meterchunk, epochs, batch_size)
            try:
                mainchunk = next(main_power_series)
                meterchunk = next(meter_power_series)
            except:
                run = False

    def train_on_chunk(self, mainchunk, meterchunk, epochs, batch_size):
        '''Train using only one chunk

        Parameters
        ----------
        mainchunk : chunk of site meter
        meterchunk : chunk of appliance
        epochs : number of epochs for training
        '''

        # Replace NaNs with 0s
        mainchunk.fillna(0, inplace=True)
        meterchunk.fillna(0, inplace=True)
        ix = mainchunk.index.intersection(meterchunk.index)
        mainchunk = mainchunk[ix]
        meterchunk = meterchunk[ix]

        num_of_batches = int(len(ix)/batch_size) - 1

        for e in range(epochs):
            print(e)
            batch_indexes = range(num_of_batches)
            random.shuffle(batch_indexes)

            for bi, b in enumerate(batch_indexes):
                print("Batch {} of {}".format(bi,num_of_batches), end="\r")
                sys.stdout.flush()
                X_batch, Y_batch = self.gen_batch(mainchunk, meterchunk, batch_size, b)
                self.model.train_on_batch(X_batch, Y_batch)
            print("\n")

        #self.model.fit(X_batch, Y_batch, batch_size=batch_size, nb_epoch=epochs, shuffle=True)

    def disaggregate(self, mains, output_datastore, **load_kwargs):
        '''Disaggregate mains according to the model learnt previously.

        Parameters
        ----------
        mains : nilmtk.ElecMeter
        output_datastore : instance of nilmtk.DataStore subclass
            For storing power predictions from disaggregation algorithm.
        **load_kwargs : key word arguments
            Passed to `mains.power_series(**kwargs)`
        '''

        load_kwargs = self._pre_disaggregation_checks(load_kwargs)

        load_kwargs.setdefault('sample_period', 60)
        load_kwargs.setdefault('sections', mains.good_sections())

        timeframes = []
        building_path = '/building{}'.format(mains.building())
        mains_data_location = building_path + '/elec/meter1'
        data_is_available = False

        for chunk in mains.power_series(**load_kwargs):
            if len(chunk) < self.MIN_CHUNK_LENGTH:
                continue
            print("New sensible chunk: {}".format(len(chunk)))

            timeframes.append(chunk.timeframe)
            measurement = chunk.name
            chunk2 = self._normalize(chunk, self.mmax)

            appliance_power = self.disaggregate_chunk(chunk2)
            appliance_power[appliance_power < 0] = 0
            appliance_power = self._denormalize(appliance_power, self.mmax)

            # Append prediction to output
            data_is_available = True
            cols = pd.MultiIndex.from_tuples([chunk.name])
            meter_instance = self.meter_metadata.instance()
            df = pd.DataFrame(
                appliance_power.values, index=appliance_power.index,
                columns=cols, dtype="float32")
            key = '{}/elec/meter{}'.format(building_path, meter_instance)
            output_datastore.append(key, df)

            # Append aggregate data to output
            mains_df = pd.DataFrame(chunk, columns=cols, dtype="float32")
            output_datastore.append(key=mains_data_location, value=mains_df)

        # Save metadata to output
        if data_is_available:
            self._save_metadata_for_disaggregation(
                output_datastore=output_datastore,
                sample_period=load_kwargs['sample_period'],
                measurement=measurement,
                timeframes=timeframes,
                building=mains.building(),
                meters=[self.meter_metadata]
            )

    def disaggregate_chunk(self, mains):
        '''In-memory disaggregation.

        Parameters
        ----------
        mains : pd.Series
        Returns
        -------
        appliance_powers : pd.DataFrame where each column represents a
            disaggregated appliance.  Column names are the integer index
            into `self.model` for the appliance in question.
        '''
        w = self.window_size
        up_limit = len(mains)

        mains.fillna(0, inplace=True)

        X_batch = np.array([ mains[i:i+w] for i in range(up_limit-(w-1)) ])
        X_batch = np.reshape(X_batch, (len(X_batch), w ,1))

        pred = self.model.predict(X_batch)
        pred = np.append(np.zeros(w-1), [x[0] for x in pred])
        column = pd.Series(pred, index=mains.index, name=0)

        appliance_powers_dict = {}
        appliance_powers_dict[0] = column
        appliance_powers = pd.DataFrame(appliance_powers_dict)
        return appliance_powers

    def gen_batch(self, mainchunk, meterchunk, batch_size, index):
        '''Generates batches from dataset

        Parameters
        ----------
        index : the index of the batch
        '''
        w = self.window_size
        offset = index*batch_size
        X_batch = np.array([ mainchunk[i+offset:i+offset+w]
                            for i in range(batch_size) ])
        Y_batch = meterchunk.values[w-1+offset:w-1+offset+batch_size]

        X_batch = np.reshape(X_batch, (len(X_batch), w ,1))

        return X_batch, Y_batch

    def import_model(self, filename):
        '''Loads keras model from h5

        Parameters
        ----------
        filename : filename for .h5 file

        Returns: Keras model
        '''
        self.model = load_model(filename)
        with h5py.File(filename, 'a') as hf:
            ds = hf.get('disaggregator-data').get('mmax')
            self.mmax = np.array(ds)[0]

    def export_model(self, filename):
        '''Saves keras model to h5

        Parameters
        ----------
        filename : filename for .h5 file
        '''
        self.model.save(filename)
        with h5py.File(filename, 'a') as hf:
            gr = hf.create_group('disaggregator-data')
            gr.create_dataset('mmax', data = [self.mmax])

    def _normalize(self, chunk, mmax):
        '''Normalizes timeseries

        Parameters
        ----------
        chunk : the timeseries to normalize
        max : max value of the powerseries

        Returns: Normalized timeseries
        '''
        tchunk = chunk / mmax
        return tchunk

    def _denormalize(self, chunk, mmax):
        '''Deormalizes timeseries
        Note: This is not entirely correct

        Parameters
        ----------
        chunk : the timeseries to denormalize
        max : max value used for normalization

        Returns: Denormalized timeseries
        '''
        tchunk = chunk * mmax
        return tchunk

    def _create_model(self, input_window):
        '''Creates the RNN module described in the paper
        '''
        model = Sequential()

        # 1D Conv

        model.add(Conv1D(16, 4, activation="linear", input_shape=(input_window,1), padding="same", strides=1))

        #Bi-directional LSTMs
        model.add(Bidirectional(LSTM(128, return_sequences=True), merge_mode='concat'))
        model.add(Bidirectional(LSTM(256, return_sequences=False), merge_mode='concat'))

        # Fully Connected Layers
        Dropout(0.2)
        model.add(Dense(128, activation='tanh'))
        Dropout(0.2)
        model.add(Dense(1, activation='linear'))

        model.compile(loss='mse', optimizer='adam')
        plot_model(model, to_file='model.png', show_shapes=True)

        return model
        
