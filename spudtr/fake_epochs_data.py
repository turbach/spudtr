import numpy as np
import pandas as pd

from spudtr.epf import EPOCH_ID, TIME
from patsy import balanced, demo_data


def _generate(
    n_epochs,
    n_samples,
    n_categories,
    n_channels,
    time=None,
    epoch_id=None,
    seed=None,
):
    """Return Pandas DataFrame with fake EEG data, and a list of channels."""

    if time is None:
        time = TIME

    if epoch_id is None:
        epoch_id = EPOCH_ID

    if seed is not None:
        np.random.seed(seed)

    total = n_epochs * n_samples * n_categories

    categories = np.array([f"cat{i}" for i in range(n_categories)])

    indices = {
        epoch_id: np.repeat(np.arange(n_epochs * n_categories), n_samples),
        time: np.tile(np.arange(n_samples), n_epochs * n_categories),
    }

    predictors = {
        "categorical": np.tile(np.repeat(categories, n_samples), n_epochs),
        "continuous": np.random.uniform(size=total),
    }

    channels = [f"channel{i}" for i in range(n_channels)]
    eeg = {
        channel: np.random.normal(loc=0, scale=30, size=total)
        for channel in channels
    }

    data = {**indices, **predictors, **eeg}

    df = pd.DataFrame(data)

    return df, channels


# fake a pd.DataFrame
def _get_df(n_a=2, n_b=3, n_epochs=4, n_streams=3):

    n_times = n_a * n_b
    n_obs = n_times * n_epochs

    epoch_id = np.repeat(range(n_epochs), n_times)
    time = np.tile(range(n_times), n_epochs)
    factors = np.repeat(
        pd.DataFrame(balanced(a=n_a, b=n_b)).to_numpy(), n_epochs, axis=0
    )
    data = np.arange(n_obs * n_streams).reshape(n_streams, n_obs).T

    df = pd.concat(
        [pd.DataFrame(arry) for arry in [epoch_id, time, factors, data]],
        axis=1,
    )
    df.columns = ["epoch_id", "time", "a", "b", "x", "y", "z"]
    return df
