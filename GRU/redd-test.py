from __future__ import print_function, division
import time

from matplotlib import rcParams
import matplotlib.pyplot as plt

from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from nilmtk.elecmeter import ElecMeterID
from grudisaggregator import GRUdisaggregator

print("========== OPEN DATASETS ============")
train = DataSet('../../Datasets/REDD/redd.h5')
test = DataSet('../../Datasets/REDD/redd.h5')

train.set_window(end="30-4-2011")
test.set_window(start="30-4-2011")

train_elec = train.buildings[1].elec
test_elec = test.buildings[1].elec

meterkeys = ['microwave']
mlist = [ElecMeterID(train_elec[m].instance(), 1, 'REDD') for m in meterkeys]
train_meter = train_elec.submeters().from_list(mlist)
train_mains = train_elec.mains().all_meters()[0]
test_mains = test_elec.mains().all_meters()[0]
disagregator = GRUdisaggregator(train_meter, 128, gpu_mode=True)


start = time.time()
print("========== TRAIN ============")
# Note that we have given the sample period to downsample the data to 6 seconds

#disagregator.import_model("model-redd100a2-1gru.h5")
disagregator.train(train_mains, train_meter, epochs=4, sample_period=6)
disagregator.export_model("model-redd100a2-1gru.h5")
end = time.time()
print("Train =", end-start, "seconds.")

print("========== DISAGGREGATE ============")
disag_filename = 'disag-out.h5'
output = HDFDataStore(disag_filename, 'w')
# Note that we have mentioned to disaggregate after converting to a sample period of 6 seconds
disagregator.disaggregate(test_mains, output, sample_period=6)
output.close()
