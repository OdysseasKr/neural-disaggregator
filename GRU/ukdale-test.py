from __future__ import print_function, division
import time

from matplotlib import rcParams
import matplotlib.pyplot as plt

from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from grudisaggregator import GRUDisaggregator
import metrics

print("========== OPEN DATASETS ============")
train = DataSet('ukdale.h5')
test = DataSet('ukdale.h5')

train.set_window(start="13-4-2013", end="1-1-2014")
test.set_window(start="1-1-2014", end="30-3-2014")

train_building = 1
test_building = 1
sample_period = 6
meter_key = 'microwave'
train_elec = train.buildings[train_building].elec
test_elec = test.buildings[test_building].elec

train_meter = train_elec.submeters()[meter_key]
train_mains = train_elec.mains()
test_mains = test_elec.mains()
gru = GRUDisaggregator()

start = time.time()
print("========== TRAIN ============")
epochs = 0
for i in range(3):
    gru.train(train_mains, train_meter, epochs=5, sample_period=sample_period)
    epochs += 5
    gru.export_model("UKDALE-GRU-h{}-{}-{}epochs.h5".format(train_building,
                                                        meter_key,
                                                        epochs))
    print("CHECKPOINT {}".format(epochs))
end = time.time()
print("Train =", end-start, "seconds.")


print("========== DISAGGREGATE ============")
disag_filename = "disag-out.h5"
output = HDFDataStore(disag_filename, 'w')
gru.disaggregate(test_mains, output, train_meter, sample_period=sample_period)
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
