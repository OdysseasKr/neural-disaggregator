# neural-disaggregator

Implementations of NILM disaggegregators using Neural Networks, using [NILMTK](https://github.com/NILMTK/NILMTK) and Keras.

Most of the architectures are based on [Neural NILM: Deep Neural Networks Applied to Energy Disaggregation](https://arxiv.org/pdf/1507.06594.pdf) by Jack Kelly and William Knottenbelt.

The implemented models are:
- Denoising autoencoder (DAE) as mentioned in [Neural NILM](https://arxiv.org/pdf/1507.06594.pdf) (see [example](https://github.com/OdysseasKr/neural-disaggregator/blob/master/DAE/DAE-example.ipynb))
- Recurrent network with LSTM neurons as mentioned in [Neural NILM](https://arxiv.org/pdf/1507.06594.pdf) (see [example](https://github.com/OdysseasKr/neural-disaggregator/tree/master/RNN/RNN-example.ipynb))
- Recurrent network with GRU. A variation of the LSTM network in order to compare the two types of RNNs (see [example](https://github.com/OdysseasKr/neural-disaggregator/blob/master/GRU/GRU-example.ipynb))
- Window GRU. A variation of the GRU network in that uses a window of data as input. As described in [Sliding Window Approach for Online Energy Disaggregation Using Artificial Neural Networks](https://dl.acm.org/citation.cfm?id=3201011) by Krystalakos, Nalmpantis and Vrakas (see [example](https://github.com/OdysseasKr/neural-disaggregator/blob/master/WindowGRU/Window-GRU-example.ipynb))
- Short Sequence to Point Network based on the architecture in [original paper](https://arxiv.org/abs/1612.09106) (see [example](https://github.com/OdysseasKr/neural-disaggregator/blob/master/ShortSeq2Point/ShortSeq2Point-example.ipynb))

_Note: If you are looking for the LookbackGRU folder, it has been moved to http://github.com/OdysseasKr/online-nilm. I try to keep this repo clean by using the same train and test sets for all architectures._
