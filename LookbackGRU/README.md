# Recurrent (GRU) Network Energy Disaggregator with Lookback

A Recurrent Neural Network disaggregator that uses a window of data (lookback) as input instead of one sample of the timeseries.

The tests for this one were done on the Synthetic data from [www.doc.ic.ac.uk/∼dk3810/neuralnilm](www.doc.ic.ac.uk/∼dk3810/neuralnilm) and not on UK-DALE or REDD.

## Instructions

First run _gen.py_. It will download the dataset if needed and generate all the needed files. You can choose the appliance and the buildings inside the code.

Then run _synth-test.py_. This is the actual experiment. You can choose the number of epochs etc.
