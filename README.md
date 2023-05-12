# SimbaML

![Overview of the SimbaML framework](docs/source/_static/visualabstract.png)

SimbaML is an all-in-one framework for integrating prior knowledge of ODE models into the ML process by synthetic data augmentation. It allows for the convenient generation of realistic synthetic data by sparsifying and adding noise. Furthermore, our framework provides customizable pipelines for various ML experiments, such as identifying needs for data collection and transfer learning.

# Installation

SimbaML can be installed with pip.

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

# How to use SimbaML

## How to generate data

In the following, we demonstrate on the example of a SIR model how the user can define a custome ODE system in SimbaML and generate data with noise:

Imports

```
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
```

Define model name and entities

```
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
```

Define kinetic parameters of the model

```
kinetic_parameters: dict[str, kinetic_parameters_module.KineticParameter[float]] = {
    "beta": kinetic_parameters_module.ConstantKineticParameter(
        distributions.NormalDistribution(0.2, 0.05)
    ),
    "gamma": kinetic_parameters_module.ConstantKineticParameter(
        distributions.NormalDistribution(0.1, 0.01)
    ),
}
```

Define the derivative function

```
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
```

Add noise to the ODE system and the output data

```
noiser = noisers.AdditiveNoiser(distributions.LogNormalDistribution(0, 2))
derivative_noiser = derivative_noisers.AdditiveDerivNoiser(
    distributions.NormalDistribution(0, 1)
)
```

Add sparsifiers to remove constant suffix from generated data

```
sparsifier1 = sparsifier_module.ConstantSuffixRemover(n=5, epsilon=1, mode="absolute")
sparsifier2 = sparsifier_module.ConstantSuffixRemover(n=5, epsilon=0.1, mode="relative")
sparsifier = sparsifier_module.SequentialSparsifier(
    sparsifiers=[sparsifier1, sparsifier2]
)
```

Build the model. Generate 1000 timestamps per time series.

```
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
```

Generate and store 100 csv files in the provided path

```
generators.TimeSeriesGenerator(sm).generate_csvs(100, "simulated_data")
```

## How to run ML experiments

We support multiple ML experiment pipelines, which can run by one command.
The details of the experiment get specified in the config file.

```
from simba_ml.prediction.time_series.pipelines import synthetic_data_pipeline
result_df = synthetic_data_pipeline.main("ml_config.toml")
```

# Documentation

https://simbaml.readthedocs.io/
