from pathlib import Path
import pytest
import spudtr.epf as epf

import pdb

TEST_DATA_DIR = Path(__file__).parent / "data"

@pytest.mark.parametrize(
    "_f,h5_group",
    [
        ["sub000p3.epochs.h5", "p3"],
        ["sub000p5.epochs.h5", "p5"],
        ["sub000wr.epochs.h5", "wr"],
    ],
)
pdb.set_trace()
def test_hdf_read_epochs(_f, h5_group):
    pdb.set_trace()
    epochs_df = epf._hdf_read_epochs(TEST_DATA_DIR / _f, h5_group)

"""
def test_center_on():

    # fake some data
    # check center_on function with fake data
    import numpy as np
    import warnings
    randval = np.random.random()
    if randval > 0.5:
        pdb.set_trace()
        raise ValueError(f'randval > 0.5 {randval}')
    else:
        warnings.warn(f'randval <= 0.5 {randval}')
"""

