from pathlib import Path
import pytest
import pfunky.epf as epf

TEST_DATA_DIR = Path(__file__).parent / 'data'

@pytest.mark.parametrize(
    "_f,h5_group",
    [
        ["sub000p3.epochs.h5", "p3"],
        ["sub000p5.epochs.h5", "p5"],
        ["sub000wr.epochs.h5", "wr"],
    ],
)
def test_hdf_read_epochs(_f, h5_group):
    epochs_df = epf._hdf_read_epochs(TEST_DATA_DIR / _f, h5_group)
