# supervised_pretraining_GN_WS
* GraphAttn, PMAreadout, etc: https://github.com/SeongokRyu/ACGT
* weight standardization: https://github.com/ThomasEhling/Weight_Standardization/blob/master/src/MNIST_Fashion_classifier.ipynb
* group normalization: https://www.tensorflow.org/addons/api_docs/python/tfa/layers/GroupNormalization

REFERENCE: https://arxiv.org/pdf/1912.11370.pdf

## *TODO*
#### Pre-train model
- [ ] set optimal dataset pipeline
- [ ] Hyper-parameter tuning
- [ ] Data preparation
    - [ ] Data acquisition (ZINC, tox21, HIV, ...)
    - [ ] generalize get_dataset function
####Benchmark
- [ ] logging outputs
- [ ] add attention to prediction layer
