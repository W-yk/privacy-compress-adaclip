# Intro

This repository is an implementation of adaptive clip and gradient compression in differential privacy. Based from [TensorFlow Privacy library](https://github.com/tensorflow/privacy) . 
# Usage

## Setting up TensorFlow Privacy
Install TensorFlow Privacy repository into a directory of your choice:

```
git clone https://github.com/tensorflow/privacy
cd privacy
pip install -e .
```
## Add optimizers

Copy the files under `tensorflow_privacy/privacy/optimizers` to the same path in previously installed TensorFlow Privacy directory.

## Compression experiments

`python code/adult_compress.py` , for adult compress results.
`python code/adult.py`, for adult baseline . 

Same goes for mnist and cifar.

## Adaclip experiment
`python code/cifar_keras_adaclip.py`
