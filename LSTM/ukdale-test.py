from __future__ import print_function, division
import time

from matplotlib import rcParams
import matplotlib.pyplot as plt

from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from rnndisaggregator import RNNDisaggregator

print("========== OPEN DATASETS ============")
train = DataSet('../../Datasets/UKDALE/ukdale.h5')
test = DataSet('../../Datasets/UKDALE/ukdale.h5')

train.set_window(start="13-4-2013", end="1-1-2014")
test.set_window(start="24-5-2013", end="8-1-2013")

train_elec = train.buildings[1].elec
test_elec = test.buildings[2].elec

train_meter = train_elec.submeters()['kettle']
train_mains = train_elec.mains()
test_mains = test_elec.mains()
rnn = RNNDisaggregator(train_meter, 32, gpu_mode=True)


start = time.time()
print("========== TRAIN ============")
# Note that we have given the sample period to downsample the data to 8 seconds
rnn.train(train_mains, train_meter, epochs=4, batch_size=128, sample_period=8)
rnn.export_model("rnnwithdropout/UKDALE-RNN-h1-kettle.h5")
end = time.time()
print("Train =", end-start, "seconds.")


print("========== DISAGGREGATE ============")
disag_filename = 'rnnwithdropout/disag-out-kettle.h5'
output = HDFDataStore(disag_filename, 'w')
# Note that we have mentioned to disaggregate after converting to a sample period of 8 seconds
rnn.disaggregate(test_mains, output, sample_period=8)
output.close()
