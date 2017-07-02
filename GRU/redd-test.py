from __future__ import print_function, division
import time

from matplotlib import rcParams
import matplotlib.pyplot as plt

from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from nilmtk.elecmeter import ElecMeterID
from grudisaggregator import GRUdisaggregator

print("========== OPEN DATASETS ============")
train = DataSet('../../Datasets/REDD/redd.h5')
train.set_window(end="30-4-2011")
test = DataSet('../../Datasets/REDD/redd.h5')
test.set_window(start="30-4-2011")

train_building = 1
test_building = 1
sample_period = 6
train_elec = train.buildings[train_building].elec
test_elec = test.buildings[test_building].elec
meterkeys = ['microwave']

disagregator = GRUdisaggregator(len(meterkeys), 128, gpu_mode=True)

start = time.time()
print("========== TRAIN ============")
mlist = [ElecMeterID(train_elec[m].instance(), train_building, 'REDD') for m in meterkeys]
train_meter = train_elec.submeters().from_list(mlist)
train_mains = train_elec.mains().all_meters()[0]
test_mains = test_elec.mains().all_meters()[0]

disagregator.train(train_mains, train_meter, epochs=4, sample_period=sample_period)
end = time.time()
print("Train =", end-start, "seconds.")

print("========== DISAGGREGATE ============")
disag_filename = 'disag-out.h5'
output = HDFDataStore(disag_filename, 'w')
mlist = [ElecMeterID(test_elec[m].instance(), test_building, test_elec[m].dataset()) for m in meterkeys]
out_metadata = test_elec.submeters().from_list(mlist)

disagregator.disaggregate(test_mains, output, out_metadata, sample_period=sample_period)
output.close()
