from __future__ import print_function, division
import time

from matplotlib import rcParams
import matplotlib.pyplot as plt

from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from nilmtk.elecmeter import ElecMeterID
from grudisaggregator import GRUdisaggregator

print("========== OPEN DATASETS ============")
train = DataSet('../../Datasets/UKDALE/ukdale.h5')
test = DataSet('../../Datasets/UKDALE/ukdale.h5')

train.clear_cache()

train.set_window(start="13-4-2013", end="13-7-2013")

train_elec = train.buildings[1].elec
test_elec = test.buildings[5].elec

meterkeys = ['kettle']
mlist = [ElecMeterID(train_elec[m].instance(), building, train_elec[m].dataset()) for m in meterkeys]
train_meter = train_elec.submeters().from_list(mlist)
train_mains = train_elec.mains()
test_mains = test_elec.mains()
disagregator = GRUdisaggregator(train_meter, 64, stateful=False, gpu_mode=True)


start = time.time()
print("========== TRAIN ============")
disagregator.train(train_mains, train_meter, epochs=4, batch_size=128, sample_period=6)
disagregator.export_model("UKDALE-MYGRU-h1-kettle-4epochs-stateless.h5")
end = time.time()
print("Train =", end-start, "seconds.")

print("========== DISAGGREGATE ============")
disag_filename = 'disag-out-h5-kettle.h5'
output = HDFDataStore(disag_filename, 'w')
disagregator.disaggregate(test_mains, output, sample_period=6)
output.close()
