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
test.set_window(start="1-1-2014", end="30-3-2014")

train_elec = train.buildings[1].elec
test_elec = test.buildings[1].elec
meter_key = 'microwave'

train_meter = train_elec.submeters()[meter_key]
train_mains = train_elec.mains()
test_mains = test_elec.mains()
rnn = RNNDisaggregator(gpu_mode=True)

start = time.time()
print("========== TRAIN ============")
epochs = 0
for i in range(1):
    rnn.train(train_mains, train_meter, epochs=5, sample_period=6)
    epochs += 5
    rnn.export_model("UKDALE-RNN-h1-{}-{}epochs.h5".format(meter_key, epochs))
    print("CHECKPOINT {}".format(epochs))
end = time.time()
print("Train =", end-start, "seconds.")


print("========== DISAGGREGATE ============")
disag_filename = "disag-out-h1-{}-{}epochs.h5".format(meter_key, epochs)
output = HDFDataStore(disag_filename, 'w')
rnn.disaggregate(test_mains, train_meter, output, sample_period=6)
output.close()
