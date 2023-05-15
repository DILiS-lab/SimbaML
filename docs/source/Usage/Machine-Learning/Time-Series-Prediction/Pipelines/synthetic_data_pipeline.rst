.. _synthetic_data_pipeline:

Synthetic Data Pipeline
=======================

To enable scalable and easy-to-run machine learning experiments on time-series data, SimbaML offers multiple pipelines covering data pre-processing, training, and evaluation of ML models. The synthetic data piplines and mixed data pipelines can be run by passing them the corresponding config files.

Configure Synthetic Data Pipeline
---------------------------------

All provided machine learning pipelines of SimbaML can be configured based on config files. This way, users can change the models that are going to be trained, their hyperparameters, specify details for preprocessing and much more.

.. literalinclude:: synthetic_data_pipeline.toml
   :language: toml
   :linenos:

Start Synthetic Data Pipeline
-----------------------------

   $ simba_ml start-prediction synthetic_data --config-path synthetic_data_pipeline.toml