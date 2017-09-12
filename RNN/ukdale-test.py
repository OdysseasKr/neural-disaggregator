from __future__ import print_function, division
import time

from matplotlib import rcParams
import matplotlib.pyplot as plt

from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from rnndisaggregator import RNNDisaggregator
import metrics

print("========== OPEN DATASETS ============")
train = DataSet('../../Datasets/UKDALE/ukdale-utc.h5')
test = DataSet('../../Datasets/UKDALE/ukdale-utc.h5')

train.set_window(start="13-4-2014", end="1-1-2015")
test.set_window(start="1-1-2015", end="30-3-2015")

meter_key = 'kettle'
test_building = 1
train_elec = train.buildings[1].elec
test_elec = test.buildings[1].elec

train_meter = train_elec.submeters()[meter_key]
train_mains = train_elec.mains()
test_mains = test_elec.mains()
rnn = RNNDisaggregator()


start = time.time()
print("========== TRAIN ============")
epochs = 0
for i in range(1):
    print("CHECKPOINT {}".format(epochs))
    rnn.train(train_mains, train_meter, epochs=1, sample_period=6)
    epochs += 5
    rnn.export_model("UKDALE-RNN-h1-{}-{}epochs.h5".format(meter_key, epochs))
end = time.time()
print("Train =", end-start, "seconds.")


print("========== DISAGGREGATE ============")
disag_filename = "disag-out-h1-{}-{}epochs.h5".format(meter_key, epochs)
output = HDFDataStore(disag_filename, 'w')
rnn.disaggregate(test_mains, output, train_meter, sample_period=6)
output.close()

print("========== RESULTS ============")
result = DataSet(disag_filename)
res_elec = result.buildings[test_building].elec
rpaf = metrics.recall_precision_accuracy_f1(res_elec[meter_key], test_elec[meter_key])
print("============ Recall: {}".format(rpaf[0]))
print("============ Precision: {}".format(rpaf[1]))
print("============ Accuracy: {}".format(rpaf[2]))
print("============ F1 Score: {}".format(rpaf[2]))

print("============ Relative error in total energy: {}".format(metrics.relative_error_total_energy(res_elec[meter_key], test_elec[meter_key])))
print("============ Mean absolute error(in Watts): {}".format(metrics.mean_absolute_error(res_elec[meter_key], test_elec[meter_key])))
