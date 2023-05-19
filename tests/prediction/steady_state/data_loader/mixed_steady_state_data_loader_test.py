import numpy as np

from simba_ml.prediction.steady_state.config import steady_state_data_config
from simba_ml.prediction.steady_state.data_loader import mixed_data_loader


def test_x_test_does_not_change_when_called_multiple_times():
    cfg = steady_state_data_config.DataConfig(
        observed="/tests/prediction/steady_state/test_data/real/",
        synthethic="/tests/prediction/steady_state/test_data/simulated/",
        mixing_ratios=[1],
        start_value_params=["A"],
        prediction_params=["B"],
    )
    loader = mixed_data_loader.MixedDataLoader(cfg)
    before = loader.X_test
    after = loader.X_test
    assert np.array_equal(before, after)
    loader.load_data()
    after = loader.X_test
    assert np.array_equal(before, after)


def test_y_test_does_not_change_when_called_multiple_times():
    cfg = steady_state_data_config.DataConfig(
        observed="/tests/prediction/steady_state/test_data/real/",
        synthethic="/tests/prediction/steady_state/test_data/simulated/",
        mixing_ratios=[1],
        start_value_params=["A"],
        prediction_params=["B"],
    )
    loader = mixed_data_loader.MixedDataLoader(cfg)
    before = loader.y_test
    after = loader.y_test
    assert np.array_equal(before, after)
    loader.load_data()
    after = loader.y_test
    assert np.array_equal(before, after)


def test_train_validation_sets_does_not_change_when_called_multiple_times():
    cfg = steady_state_data_config.DataConfig(
        observed="/tests/prediction/steady_state/test_data/real/",
        synthethic="/tests/prediction/steady_state/test_data/simulated/",
        mixing_ratios=[1.0],
        start_value_params=["A"],
        prediction_params=["B"],
    )
    loader = mixed_data_loader.MixedDataLoader(cfg)
    assert len(loader.list_of_train_validation_sets) == 1
    before = loader.list_of_train_validation_sets
    assert len(loader.list_of_train_validation_sets) == 1
    after = loader.list_of_train_validation_sets
    compare_train_validation_sets(before, after)
    loader.load_data()
    after = loader.list_of_train_validation_sets
    compare_train_validation_sets(before, after)


def compare_train_validation_sets(before, after):
    for i, before_i in enumerate(before):
        for j, before_i_j in enumerate(before_i):
            for key in before_i_j.keys():
                for k, _ in enumerate(before_i_j[key]):
                    assert np.all(before_i_j[key][k] == after[i][j][key][k])
