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
from keras.layers import Dense, Conv1D, GRU, Bidirectional, Dropout, Flatten
from keras.utils import plot_model

from nilmtk.utils import find_nearest
from nilmtk.feature_detectors import cluster
from nilmtk.disaggregate import Disaggregator
from nilmtk.datastore import HDFDataStore

class ShortSeq2PointDisaggregator(Disaggregator):
    '''Attempt to create a RNN Disaggregator

    Attributes
    ----------
    model : keras Sequential model
    mmax : the maximum value of the aggregate data

    MIN_CHUNK_LENGTH : int
       the minimum length of an acceptable chunk
    '''

    def __init__(self, window_size=100):
        '''Initialize disaggregator
        '''
        self.MODEL_NAME = "WindowGRU"
        self.mmax = None
        self.MIN_CHUNK_LENGTH = window_size
        self.window_size = window_size
        self.model = self._create_model()

    def train(self, mains, meter, epochs=1, batch_size=128, **load_kwargs):
        '''Train

        Parameters
        ----------
        mains : a nilmtk.ElecMeter object for the aggregate data
        meter : a nilmtk.ElecMeter object for the meter data
        epochs : number of epochs to train
        batch_size : size of batch used for training
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
        batch_size : size of batch used for training
        '''

        # Replace NaNs with 0s
        mainchunk.fillna(0, inplace=True)
        meterchunk.fillna(0, inplace=True)
        ix = mainchunk.index.intersection(meterchunk.index)
        mainchunk = np.array(mainchunk[ix])
        meterchunk = np.array(meterchunk[ix])

        indexer = np.arange(self.window_size)[None, :] + np.arange(len(mainchunk)-self.window_size+1)[:, None]
        mainchunk = mainchunk[indexer]
        meterchunk = meterchunk[self.window_size-1:]
        mainchunk = np.reshape(mainchunk, (mainchunk.shape[0], mainchunk.shape[1],1))

        self.model.fit(mainchunk, meterchunk, epochs=epochs, batch_size=batch_size, shuffle=True)

    def train_across_buildings(self, mainlist, meterlist, epochs=1, batch_size=128, **load_kwargs):
        '''Train using data from multiple buildings

        Parameters
        ----------
        mainlist : a list of nilmtk.ElecMeter objects for the aggregate data of each building
        meterlist : a list of nilmtk.ElecMeter objects for the meter data of each building
        batch_size : size of batch used for training
        epochs : number of epochs to train
        **load_kwargs : keyword arguments passed to `meter.power_series()`
        '''

        assert len(mainlist) == len(meterlist), "Number of main and meter channels should be equal"
        num_meters = len(mainlist)

        mainps = [None] * num_meters
        meterps = [None] * num_meters
        mainchunks = [None] * num_meters
        meterchunks = [None] * num_meters

        # Get generators of timeseries
        for i,m in enumerate(mainlist):
            mainps[i] = m.power_series(**load_kwargs)

        for i,m in enumerate(meterlist):
            meterps[i] = m.power_series(**load_kwargs)

        # Get a chunk of data
        for i in range(num_meters):
            mainchunks[i] = next(mainps[i])
            meterchunks[i] = next(meterps[i])
        if self.mmax == None:
            self.mmax = max([m.max() for m in mainchunks])


        run = True
        while(run):
            # Normalize and train
            mainchunks = [self._normalize(m, self.mmax) for m in mainchunks]
            meterchunks = [self._normalize(m, self.mmax) for m in meterchunks]

            self.train_across_buildings_chunk(mainchunks, meterchunks, epochs, batch_size)

            # If more chunks, repeat
            try:
                for i in range(num_meters):
                    mainchunks[i] = next(mainps[i])
                    meterchunks[i] = next(meterps[i])
            except:
                run = False

    def train_across_buildings_chunk(self, mainchunks, meterchunks, epochs, batch_size):
        '''Train using only one chunk of data. This chunk consists of data from
        all buildings.

        Parameters
        ----------
        mainchunk : chunk of site meter
        meterchunk : chunk of appliance
        epochs : number of epochs for training
        batch_size : size of batch used for training
        '''
        num_meters = len(mainchunks)
        batch_size = int(batch_size/num_meters)
        num_of_batches = [None] * num_meters

        # Find common parts of timeseries
        for i in range(num_meters):
            mainchunks[i].fillna(0, inplace=True)
            meterchunks[i].fillna(0, inplace=True)
            ix = mainchunks[i].index.intersection(meterchunks[i].index)
            m1 = mainchunks[i]
            m2 = meterchunks[i]
            mainchunks[i] = m1[ix]
            meterchunks[i] = m2[ix]

            indexer = np.arange(self.window_size)[None, :] + np.arange(len(mainchunks[i].values)-self.window_size+1)[:, None]
            mainchunks[i] = mainchunks[i].values[indexer]
            meterchunks[i] = meterchunks[i].values[self.window_size-1:]

            num_of_batches[i] = int(len(ix)/batch_size) - 1

        for e in range(epochs): # Iterate for every epoch
            print(e)
            batch_indexes = list(range(min(num_of_batches)))
            random.shuffle(batch_indexes)

            for bi, b in enumerate(batch_indexes): # Iterate for every batch
                print("Batch {} of {}".format(bi,min(num_of_batches)), end="\r")
                sys.stdout.flush()
                X_batch = np.empty((batch_size*num_meters, self.window_size, 1))
                Y_batch = np.empty((batch_size*num_meters, 1))

                # Create a batch out of data from all buildings
                for i in range(num_meters):
                    mainpart = mainchunks[i]
                    meterpart = meterchunks[i]
                    mainpart = mainpart[b*batch_size:(b+1)*batch_size]
                    meterpart = meterpart[b*batch_size:(b+1)*batch_size]
                    X = np.reshape(mainpart, (batch_size, self.window_size, 1))
                    Y = np.reshape(meterpart, (batch_size, 1))

                    X_batch[i*batch_size:(i+1)*batch_size] = np.array(X)
                    Y_batch[i*batch_size:(i+1)*batch_size] = np.array(Y)

                # Shuffle data
                p = np.random.permutation(len(X_batch))
                X_batch, Y_batch = X_batch[p], Y_batch[p]

                # Train model
                self.model.train_on_batch(X_batch, Y_batch)
            print("\n")

    def disaggregate(self, mains, output_datastore, meter_metadata, **load_kwargs):
        '''Disaggregate mains according to the model learnt previously.

        Parameters
        ----------
        mains : a nilmtk.ElecMeter of aggregate data
        meter_metadata: a nilmtk.ElecMeter of the observed meter used for storing the metadata
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
            meter_instance = meter_metadata.instance()
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
                meters=[meter_metadata]
            )

    def disaggregate_chunk(self, mains):
        '''In-memory disaggregation.

        Parameters
        ----------
        mains : pd.Series of aggregate data
        Returns
        -------
        appliance_powers : pd.DataFrame where each column represents a
            disaggregated appliance.  Column names are the integer index
            into `self.model` for the appliance in question.
        '''
        up_limit = len(mains)

        mains.fillna(0, inplace=True)

        X_batch = np.array(mains)
        Y_len = len(X_batch)
        indexer = np.arange(self.window_size)[None, :] + np.arange(len(X_batch)-self.window_size+1)[:, None]
        X_batch = X_batch[indexer]
        X_batch = np.reshape(X_batch, (X_batch.shape[0],X_batch.shape[1],1))

        pred = self.model.predict(X_batch, batch_size=128)
        pred = np.reshape(pred, (len(pred)))
        column = pd.Series(pred, index=mains.index[self.window_size-1:Y_len], name=0)

        appliance_powers_dict = {}
        appliance_powers_dict[0] = column
        appliance_powers = pd.DataFrame(appliance_powers_dict)
        return appliance_powers

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

    def _create_model(self):
        '''Creates and returns the ShortSeq2Point Network
        Based on: https://arxiv.org/pdf/1612.09106v3.pdf
        '''
        model = Sequential()

        # 1D Conv
        model.add(Conv1D(30, 10, activation='relu', input_shape=(self.window_size,1), padding="same", strides=1))
        model.add(Dropout(0.5))
        model.add(Conv1D(30, 8, activation='relu', padding="same", strides=1))
        model.add(Dropout(0.5))
        model.add(Conv1D(40, 6, activation='relu', padding="same", strides=1))
        model.add(Dropout(0.5))
        model.add(Conv1D(50, 5, activation='relu', padding="same", strides=1))
        model.add(Dropout(0.5))
        model.add(Conv1D(50, 5, activation='relu', padding="same", strides=1))
        model.add(Dropout(0.5))        
        # Fully Connected Layers
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='linear'))

        model.compile(loss='mse', optimizer='adam')
        print(model.summary())
        plot_model(model, to_file='model.png', show_shapes=True)

        return model
