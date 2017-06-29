from __future__ import print_function, division
import time

from matplotlib import rcParams
import matplotlib.pyplot as plt

from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from rnndisaggregator import RNNDisaggregator

print("========== OPEN DATASETS ============")
train = DataSet('../../Datasets/REDD/redd.h5')
test = DataSet('../../Datasets/REDD/redd.h5')

train.set_window(end="30-4-2011")
test.set_window(start="30-4-2011")

train_elec = train.buildings[1].elec
test_elec = test.buildings[1].elec

train_meter = train_elec.submeters()['fridge']
train_mains = train_elec.mains().all_meters()[0]
test_mains = test_elec.mains().all_meters()[0]
rnn = RNNDisaggregator(train_meter, 128, gpu_mode=True)


start = time.time()
print("========== TRAIN ============")
rnn.train(train_mains, train_meter, epochs=100, sample_period=60)
rnn.export_model("model-redd100.h5")
end = time.time()
print("Train =", end-start, "seconds.")


print("========== DISAGGREGATE ============")
disag_filename = 'disag-out.h5'
output = HDFDataStore(disag_filename, 'w')
rnn.disaggregate(test_mains, output, sample_period=60)
output.close()
