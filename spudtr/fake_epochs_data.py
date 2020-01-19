import numpy as np
import pandas as pd

from spudtr.epf import EPOCH_ID, TIME


def _generate(
    n_epochs, n_samples, n_categories, n_channels, time, epoch_id, seed=None
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
