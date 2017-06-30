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
from keras.layers import Dense, Conv1D, GRU, Bidirectional, Dropout
from keras.utils import plot_model

from nilmtk.utils import find_nearest
from nilmtk.feature_detectors import cluster
from nilmtk.disaggregate import Disaggregator
from nilmtk.datastore import HDFDataStore

class GRUDisaggregator(Disaggregator):
    '''Attempt to create a GRU Disaggregator

    Attributes
    ----------
    model : keras Sequential model
    meter_metadata : metergroup of meters to be disaggregated
    window_size : the size of window to use on the aggregate data
    mmax : the maximum value of the aggregate data
    stateful : true if the gru layers are stateful
    gpu_mode : true if this is intended for gpu execution

    MIN_CHUNK_LENGTH : int
       the minimum length of an acceptable chunk
    '''

    def __init__(self, metergroup, window_size, stateful=False, gpu_mode=False):
        '''Initialize disaggregator
        DOES NOT TAKE INTO ACCOUNT EXISTANCE OF VAMPIRE POWER

        Parameters
        ----------
        window_size : the size of window to use on the aggregate data
        metergroup : a nilmtk.Metergroup with the appliances to be disaggregated
        gpu_mode : true if this is intended for gpu execution
        '''
        self.MODEL_NAME = "MYGRU"
        self.mmax = None
        self.window_size = window_size
        self.MIN_CHUNK_LENGTH = window_size
        self.gpu_mode = gpu_mode
        self.stateful = stateful
        self.meter_metadata = metergroup
        self.model = self._create_model(self.window_size)
        print(self.model.summary())

    def train(self, mains, metergroup, epochs=1, batch_size=128, **load_kwargs):
        '''Train

        Parameters
        ----------
        mains : a nilmtk.ElecMeter object for the aggregate data
        meter : a nilmtk.MeterGroup object for the meter data
        epochs : number of epochs to train
        **load_kwargs : keyword arguments passed to `meter.power_series()`
        '''

        main_power_series = mains.power_series(**load_kwargs)
        meter_power_series = []
        for i, meter in enumerate(metergroup.all_meters()):
            meter_power_series.append(meter.power_series(**load_kwargs))

        # Train chunks
        run = True
        mainchunk = next(main_power_series).fillna(0)
        meterchunks = []
        for i,meterps in enumerate(meter_power_series):
            meterchunks.append(next(meterps).fillna(0))


        if self.mmax == None:
            self.mmax = mainchunk.max()

        while(run):
            mainchunk = self._normalize(mainchunk, self.mmax)
            meterchunks = [self._normalize(m, self.mmax) for m in meterchunks]

            self.train_on_chunk(mainchunk, meterchunks, epochs, batch_size)
            try:
                mainchunk = next(main_power_series).fillna(0)
                meterchunks = []
                for i,meterps in enumerate(meter_power_series):
                    meterchunks[i] = next(meterps).fillna(0)
            except:
                run = False

    def train_on_chunk(self, mainchunk, meterchunks, epochs, batch_size):
        '''Train using only one chunk

        Parameters
        ----------
        mainchunk : chunk of site meter
        meterchunks : array of chunks of appliances
        epochs : number of epochs for training
        batch_size : size of batch for each train iteration
        '''

        w = self.window_size
        X_batch = None
        """
        # Align and trim
        up_limit =  min([m.index.max() for m in meterchunks] + [mainchunk.index.max()])
        down_limit =  max([m.index.min() for m in meterchunks] + [mainchunk.index.min()])
        mainchunk = mainchunk[down_limit:up_limit]
        meterchunks = [m[down_limit:up_limit] for m in meterchunks]
        arraylen = len(mainchunk.values)

        X_batch = np.array([ mainchunk.values[i:i+w] for i in range(arraylen-(w-1)) ])
        X_batch = np.reshape(X_batch, (len(X_batch), w ,1))
        Y_batch = np.array([ m[w-1:] for m in meterchunks ])
        Y_batch = np.transpose(Y_batch)

        if self.stateful:
            X_batch = X_batch[:-(len(X_batch)%batch_size)]
            Y_batch = Y_batch[:-(len(Y_batch)%batch_size)]
        print(X_batch.shape)
        print(Y_batch.shape)
        print(not self.stateful)

        self.model.fit(X_batch, Y_batch, batch_size=batch_size, nb_epoch=epochs, shuffle=(not self.stateful))
        """

        ixs = [m.index for m in meterchunks]
        ix = mainchunk.index.intersection(*ixs)
        mainchunk = mainchunk[ix]
        meterchunks = [m[ix] for m in meterchunks]
        num_of_batches = int(len(ix)/batch_size) - 1

        for e in range(epochs):
            print(e)
            batch_indexes = range(num_of_batches)
            if not self.stateful:
                random.shuffle(batch_indexes)

            for bi, b in enumerate(batch_indexes):
                print("Batch {} of {}".format(bi,num_of_batches), end="\r")
                sys.stdout.flush()
                X_batch, Y_batch = self.gen_batch(mainchunk, meterchunks, batch_size, b)
                self.model.train_on_batch(X_batch, Y_batch)
            print("\n")


    def disaggregate(self, mains, output_datastore, **load_kwargs):
        '''Disaggregate mains according to the model learnt previously.

        Parameters
        ----------
        mains : nilmtk.ElecMeter
        output_datastore : instance of nilmtk.DataStore subclass
            For storing power predictions from disaggregation algorithmself.
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
            for i,meter in enumerate(self.meter_metadata.all_meters()):
                data_is_available = True
                cols = pd.MultiIndex.from_tuples([chunk.name])
                meter_instance = meter.instance()
                df = pd.DataFrame(
                    appliance_power.get(i).values, index=appliance_power.get(i).index,
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
                meters=self.meter_metadata.all_meters()
            )

    def disaggregate_chunk(self, mains):
        """In-memory disaggregation.

        Parameters
        ----------
        mains : pd.Series
        Returns
        -------
        appliance_powers : pd.DataFrame where each column represents a
            disaggregated appliance.  Column names are the integer index
            into `self.model` for the appliance in question.
        """
        w = self.window_size
        up_limit = len(mains)

        mains.fillna(0, inplace=True)
        X_batch = np.array([ mains[i:i+w] for i in range(up_limit-(w-1)) ])

        X_batch = np.reshape(X_batch, (len(X_batch), w ,1))
        if self.stateful:
            X_batch = X_batch[:-(len(X_batch)%128)]

        print(X_batch.shape)
        pred = self.model.predict(X_batch, batch_size=128)
        pred = np.transpose(pred)

        appliance_powers_dict = {}
        for i,p in enumerate(pred):
            tmp = np.append(np.zeros(w-1), [x for x in p])

            if self.stateful:
                column = pd.Series(tmp, index=mains.index[:len(tmp)], name=0)
            else:
                column = pd.Series(tmp, index=mains.index, name=0)

            appliance_powers_dict[i] = column

        appliance_powers = pd.DataFrame(appliance_powers_dict)
        return appliance_powers

    def gen_batch(self, mainchunk, meterchunks, batch_size, index):
        '''Generates batches from dataset

        Parameters
        ----------
        index : the index of the batch
        '''
        w = self.window_size
        offset = index*batch_size
        X_batch = np.array([ mainchunk[i+offset:i+offset+w]
                            for i in range(batch_size) ])
        Y_batch = [m.values[w-1+offset:w-1+offset+batch_size] for m in meterchunks]

        X_batch = np.reshape(X_batch, (len(X_batch), w ,1))
        Y_batch = np.transpose(Y_batch)

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
        if self.stateful:
            model.add(Conv1D(16, 4, activation="relu", padding="same", strides=1, batch_input_shape=(128,input_window,1)))
        else:
            model.add(Conv1D(16, 4, activation="relu", padding="same", strides=1, input_shape=(input_window,1)))
        model.add(Conv1D(8, 4, activation="relu", padding="same", strides=1))

        #Bi-directional LSTMs
        model.add(Bidirectional(GRU(128, return_sequences=True, stateful=self.stateful), merge_mode='concat'))
        model.add(Bidirectional(GRU(256, return_sequences=False, stateful=self.stateful), merge_mode='concat'))

        # Fully Connected Layers
        Dropout(0.2)
        model.add(Dense(len(self.meter_metadata.all_meters()), activation='linear'))

        model.compile(loss='mse', optimizer='adam')
        plot_model(model, to_file='model.png', show_shapes=True)
        return model
