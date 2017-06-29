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

#train.set_window(start="13-4-2013", end="13-6-2013")
#train.set_window(start="13-6-2013", end="13-8-2013")

#test.set_window(start="1-1-2014", end="30-1-2014")
#test.set_window(end="9-5-2013")

train_elec = train.buildings[building].elec
#train_elec2 = train2.buildings[1].elec
#train_elec3 = train3.buildings[5].elec
test_elec = test.buildings[2].elec

meterkeys = ['washer dryer']
mlist = [ElecMeterID(train_elec[m].instance(), building, train_elec[m].dataset()) for m in meterkeys]
train_meter = train_elec.submeters().from_list(mlist)
train_mains = train_elec.mains()
test_mains = test_elec.mains()
disagregator = GRUdisaggregator(train_meter, 64, stateful=False, gpu_mode=True)


start = time.time()
print("========== TRAIN ============")
# Note that we have given the sample period to downsample the data to 6 seconds

#disagregator.import_model("UKDALE-MYGRU-h1-washer-dryer-8epochs-stateless.h5")
disagregator.train(train_mains, train_meter, epochs=1, batch_size=128, sample_period=6)
disagregator.export_model("UKDALE-MYGRU-h1-washer-dryer-9epochs-stateless.h5")
end = time.time()
print("Train =", end-start, "seconds.")

print("========== DISAGGREGATE ============")
disag_filename = 'disag-out-h5-fridge.h5'
output = HDFDataStore(disag_filename, 'w')
# Note that we have mentioned to disaggregate after converting to a sample period of 6 seconds
disagregator.disaggregate(test_mains, output, sample_period=6)
output.close()
