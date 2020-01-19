from pathlib import Path
import pytest

# import spudtr.epf as epf
import spudtr.fake_epochs_data as fake_data
from spudtr.epf import EPOCH_ID, TIME

# import epf as epf
# import fake_epochs_data as fake_data

import numpy as np
import pandas as pd


def test__generate():

    epochs_df, channels = fake_data._generate(
        n_epochs=10,
        n_samples=100,
        n_categories=2,
        n_channels=32,
        time=TIME,
        epoch_id=EPOCH_ID,
    )

    epochs_df, channels = fake_data._generate(
        n_epochs=10,
        n_samples=100,
        n_categories=2,
        n_channels=32,
        time=TIME,
        epoch_id=EPOCH_ID,
        seed=10,
    )
