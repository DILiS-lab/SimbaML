# SimbaML

SimbaML is an all-in-one framework for integrating prior knowledge of ODE models into the ML process by synthetic data augmentation. It allows for the convenient generation of realistic synthetic data by sparsifying and adding noise. Furthermore, our framework provides customizable pipelines for various ML experiments, such as identifying needs for data collection and transfer learning.

![Overview of the SimbaML Framework](docs/source/_static/visualabstract.png)

# Installation

SimbaML requires Python 3.10 or newer and can be installed via pip:

```
pip install simba_ml
```

To be lightweight, SimbaML does not install PyTorch and TensorFlow per default. Both packages need to be installed manually by the user.

```
pip install pytorch-lightning>=1.9.0
```

```
pip install tensorflow>=2.10.0; platform_machine != 'arm64'
```

For further details on how to install Tensorflow on ARM-based MacOS devices, see: https://developer.apple.com/metal/tensorflow-plugin/


# Documentation

We provide detailed documentation for SimbaML here: https://simbaml.readthedocs.io/.
