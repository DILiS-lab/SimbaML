.. _installation:

Installation
============

SimbaML requires Python 3.10 or newer and can be installed via pip:

..  code-block:: bash
    
    pip install simba_ml

Check if installation was successfull by running:

.. code-block:: python

    import simba_ml
    simba_ml.__version__

Dependencies
------------

To be lightweight, SimbaML does not install PyTorch and TensorFlow per default. Both packages need to be installed manually by the user.

.. code-block:: python

pip install pytorch-lightning>=1.9.0

.. code-block:: python

pip install tensorflow>=2.10.0; platform_machine != 'arm64'

For further details on how to install Tensorflow on ARM-based MacOS devices, see: https://developer.apple.com/metal/tensorflow-plugin/