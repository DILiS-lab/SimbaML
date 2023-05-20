"""Provides a model that predicts next timesteps from with a\
     pytorch lightning architecture."""
import abc
import dataclasses

import torch
from torch import utils
import numpy as np
import numpy.typing as npt
import pandas as pd
from torch.optim import Adam

from simba_ml.prediction.time_series.data_loader import window_generator
from simba_ml.prediction.time_series.models import model
from simba_ml.prediction import normalizer
from simba_ml.prediction.time_series.config import (
    time_series_config,
)

try:  # pragma: no cover
    import pytorch_lightning as pl
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "PyTorch Lightning is not installed. Please install it to "
        "use the PyTorchLightningModel."
    ) from e

if tuple(int(v) for v in pl.__version__.split(".")) < (  # type: ignore[attr-defined]
    1,
    9,
    0,
):  # pragma: no cover
    raise ImportError(
        "PyTorch Lightning version 1.9.0 or higher is required \
            for the PyTorchLightningModel."
    )


@dataclasses.dataclass
class ArchitectureParams:
    """Defines the parameters for the architecture."""

    units: int = 32
    activation: str = "relu"


@dataclasses.dataclass
class TrainingParams:
    """Defines the parameters for the training."""

    epochs: int = 10
    patience: int = 5
    batch_size: int = 32
    validation_split: float = 0.2
    verbose: int = 0
    accelerator: str = "auto"
    learning_rate: float = 0.001

@dataclasses.dataclass
class PytorchLightningModelConfig(model.ModelConfig):
    """Defines the configuration for the PytorchLightningModel."""

    name: str = "Pytorch Lightning Model"
    architecture_params: ArchitectureParams = dataclasses.field(
        default_factory=ArchitectureParams
    )
    training_params: TrainingParams = dataclasses.field(default_factory=TrainingParams)
    normalize: bool = True
    seed: int = 42


class PytorchLightningModel(model.Model):
    """Defines a Pytorch Lightning model to predict the next timestamps."""

    model_params: PytorchLightningModelConfig

    def __init__(
        self,
        time_series_params: time_series_config.TimeSeriesConfig,
        model_params: PytorchLightningModelConfig,
    ) -> None:
        """Initializes the model.

        Args:
            time_series_params: configuration for the time series that
                influence the training and archicture of the model.
            model_params: configuration for the model.

        Raises:
            TypeError: if input_length or output_length is not an integer.
        """
        super().__init__(time_series_params, model_params)
        self.set_seed(self.model_params.seed)
        if self.model_params.normalize:
            self.normalizer = normalizer.Normalizer()
        self.model_params = model_params
        self.model = self.get_model(time_series_params, model_params)

    @abc.abstractmethod
    def get_model(
        self,
        time_series_params: time_series_config.TimeSeriesConfig,
        model_params: PytorchLightningModelConfig,
    ) -> pl.LightningModule:
        """Returns the model.

        Args:
            time_series_params: configuration for the time series that
                influence the training and archicture of the model.
            model_params: configuration for the model.
        """

    def set_seed(self, seed: int) -> None:
        """Sets the seed for the model.

        Args:
            seed: seed to set.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

    def train(self, train: list[pd.DataFrame]) -> None:
        """Trains the model with the given data.

        Args:
            train: training data.
        """
        train = [array.astype(np.float32) for array in train]

        if self.model_params.normalize:
            train = self.normalizer.normalize_train_data(train, self.time_series_params)

        X_train, y_train = window_generator.create_window_dataset(
            train, self.time_series_params
        )

        train_data = list(zip(X_train, y_train))

        train_loader: utils.data.DataLoader[
            torch.FloatTensor  # pylint: disable=no-member
        ] = utils.data.DataLoader(
            train_data,  # type: ignore[arg-type]
            batch_size=self.model_params.training_params.batch_size,
            shuffle=True,
            num_workers=0,
        )

        trainer = pl.Trainer(
            max_epochs=self.model_params.training_params.epochs,
            log_every_n_steps=len(train_loader) // 2,
            accelerator=self.model_params.training_params.accelerator,
        )
        trainer.optimizers([Adam(params=self.model.parameters(), lr=self.model_params.training_params.learning_rate)])
        trainer.fit(self.model, train_loader)

    def predict(self, data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Predicts the next timestamps for every row (time series).

        Args:
            data: np.array, where each dataframe is a time series.

        Returns:
            np.array, where each value is a time series.
        """
        self.set_seed(self.model_params.seed)
        if self.model_params.normalize:
            data = self.normalizer.normalize_test_data(data)
        data_to_torch = torch.from_numpy(data).to(  # pylint: disable=no-member
            torch.float32  # pylint: disable=no-member
        )
        prediction = self.model(data_to_torch).cpu().detach().numpy()
        if self.model_params.normalize:
            prediction = self.normalizer.denormalize_prediction_data(prediction)
        return prediction
