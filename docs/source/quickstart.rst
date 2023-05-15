Quickstart
==========

In the following we provide an example to show you SimbaML's main functionalities.

Data Generation
------------------------------

We generate noisy data using acustom ODE-system. In this case we chose a simple SIR model.

1. Import the necessary modules.

.. code-block:: python
  from simba_ml.simulation import (
      system_model,
      species,
      noisers,
      constraints,
      distributions,
      sparsifier as sparsifier_module,
      kinetic_parameters as kinetic_parameters_module,
      constraints,
      derivative_noiser as derivative_noisers,
      generators
  )

2. Define model name and entities.

.. code-block:: python
  name = "SIR"
  specieses = [
      species.Species(
          "Suspectible", distributions.NormalDistribution(1000, 100),
          contained_in_output=False, min_value=0,
      ),
      species.Species(
          "Infected", distributions.LogNormalDistribution(10, 2),
          min_value=0
      ),
      species.Species(
          "Recovered", distributions.Constant(0),
          contained_in_output=False, min_value=0)
  ]

3. Define kinetic parameters of the model.

.. code-block:: python
  kinetic_parameters: dict[str, kinetic_parameters_module.KineticParameter[float]] = {
      "beta": kinetic_parameters_module.ConstantKineticParameter(
          distributions.NormalDistribution(0.2, 0.05)
      ),
      "gamma": kinetic_parameters_module.ConstantKineticParameter(
          distributions.NormalDistribution(0.1, 0.01)
      ),
  }

4. Define the derivative function.

.. code-block:: python
  def deriv(
      _t: float, y: list[float], arguments: dict[str, float]
  ) -> tuple[float, float, float]:
      """Defines the derivative of the function at the point _.

      Args:
          y: Current y vector.
          arguments: Dictionary of arguments configuring the problem.

      Returns:
          Tuple[float, float, float]
      """
      S, I, _ = y
      N = sum(y)
      dS_dt = -arguments["beta"] * S * I / N
      dI_dt = arguments["beta"] * S * I / N - (arguments["gamma"]) * I
      dR_dt = arguments["gamma"] * I
      return dS_dt, dI_dt, dR_dt

5. Add noise to the ODE system and the output data.

.. code-block:: python
  noiser = noisers.AdditiveNoiser(distributions.LogNormalDistribution(0, 2))
  derivative_noiser = derivative_noisers.AdditiveDerivNoiser(
      distributions.NormalDistribution(0, 1)
  )

6. Add sparsifiers to remove constant suffix from generated data.

.. code-block:: python
  sparsifier1 = sparsifier_module.ConstantSuffixRemover(n=5, epsilon=1, mode="absolute")
  sparsifier2 = sparsifier_module.ConstantSuffixRemover(n=5, epsilon=0.1, mode="relative")
  sparsifier = sparsifier_module.SequentialSparsifier(
      sparsifiers=[sparsifier1, sparsifier2]
  )

7. Build the model. Generate 1000 timestamps per time series.

.. code-block:: python
  sm = constraints.SpeciesValueTruncator(
      system_model.SystemModel(
          name,
          specieses,
          kinetic_parameters,
          deriv=deriv,
          noiser=noiser,
          sparsifier=sparsifier,
          timestamps=distributions.Constant(1000),
      )
  )

8. Generate and store 100 csv files in custom path.

.. code-block:: python
  generators.TimeSeriesGenerator(sm).generate_csvs(100, "simulated_data")

Run ML Pipelines
----------------

We support multiple ML experiment pipelines, which can run by one command.
In this case we run the synthetic data pipeline that only uses the just generated data.
The details of the ML experiments get specified in the config file.
You find an examplary config for the synthetic data pipeline under :ref:`Usage/Machine Learning/Time-Series-Prediction/Pipelines/synthetic_data_pipeline`

.. code-block:: python
  from simba_ml.prediction.time_series.pipelines import synthetic_data_pipeline
  result_df = synthetic_data_pipeline.main("ml_config.toml")