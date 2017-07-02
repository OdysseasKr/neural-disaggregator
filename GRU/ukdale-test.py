from __future__ import print_function, division
import time

from matplotlib import rcParams
import matplotlib.pyplot as plt

from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from nilmtk.elecmeter import ElecMeterID
from grudisaggregator import GRUDisaggregator

print("========== OPEN DATASETS ============")
train = DataSet('../../Datasets/UKDALE/ukdale.h5')
train.set_window(start="13-4-2013", end="13-7-2013")
test = DataSet('../../Datasets/UKDALE/ukdale.h5')

train.clear_cache()

train_building = 1
test_building = 5
sample_period = 6
train_elec = train.buildings[train_building].elec
test_elec = test.buildings[test_building].elec
meterkeys = ['kettle']

disagregator = GRUDisaggregator(len(meterkeys), 64, stateful=False, gpu_mode=True)

start = time.time()
print("========== TRAIN ============")
mlist = [ElecMeterID(train_elec[m].instance(), train_building, train_elec[m].dataset()) for m in meterkeys]
train_meter = train_elec.submeters().from_list(mlist)
train_mains = train_elec.mains()

#disagregator.import_model("UKDALE-MYGRU-h1-kettle-6epochs-stateless.h5")
disagregator.train(train_mains, train_meter, epochs=4, batch_size=128, sample_period=sample_period)
end = time.time()
print("Train =", end-start, "seconds.")

print("========== DISAGGREGATE ============")
test_mains = test_elec.mains()
disag_filename = 'temp.h5'
output = HDFDataStore(disag_filename, 'w')
mlist = [ElecMeterID(test_elec[m].instance(), test_building, test_elec[m].dataset()) for m in meterkeys]
out_metadata = test_elec.submeters().from_list(mlist)
print(out_metadata)

disagregator.disaggregate(test_mains, output, out_metadata, sample_period=sample_period)
output.close()
