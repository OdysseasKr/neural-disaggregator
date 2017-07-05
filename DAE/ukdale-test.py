from __future__ import print_function, division
import time

from matplotlib import rcParams
import matplotlib.pyplot as plt

from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from daedisaggregator import DAEDisaggregator

print("========== OPEN DATASETS ============")
train = DataSet('../../Datasets/UKDALE/ukdale.h5')
test = DataSet('../../Datasets/UKDALE/ukdale.h5')

train.set_window(start="13-4-2013", end="1-1-2014")
#test.set_window(start="1-1-2014", end="30-3-2014")
test.set_window(start="24-5-2013", end="8-1-2013")

train_elec = train.buildings[1].elec
test_elec = test.buildings[2].elec

train_meter = train_elec.submeters()['fridge']
test_meter = test_elec.submeters()['fridge']
train_mains = train_elec.mains()
test_mains = test_elec.mains()
dae = DAEDisaggregator(256, gpu_mode=True)


start = time.time()
print("========== TRAIN ============")
dae.train(train_mains, train_meter, epochs=40, sample_period=1)
dae.export_model("UKDALE-DAE-h1-Fridge2.h5")
end = time.time()
print("Train =", end-start, "seconds.")


print("========== DISAGGREGATE ============")
disag_filename = 'disag-out2.h5'
output = HDFDataStore(disag_filename, 'w')
dae .disaggregate(test_mains, output, test_meter, sample_period=1)
output.close()
