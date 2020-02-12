import pytest
import spudtr.fake_epochs_data as fake_data
import numpy as np
from spudtr import epf, DATA_DIR, P3_F, P5_F, WR_F


@pytest.mark.parametrize(
    "_f,h5_group", [[P3_F, "p3"], [P5_F, "p5"], [WR_F, "wr"]]
)
def test_hdf_read_epochs(_f, h5_group):
    epf._hdf_read_epochs(DATA_DIR / _f, h5_group)


# test one file
def test_epochs_QC():
    epochs_df, channels = fake_data._generate(
        n_epochs=10,
        n_samples=100,
        n_categories=2,
        n_channels=32,
        time=epf.TIME,
        epoch_id=epf.EPOCH_ID,
    )

    eeg_streams = ["channel0", "channel1", "channel2", "channel3", "channel4"]
    epf._epochs_QC(epochs_df, eeg_streams)


def test_center_on():
    epochs_df, channels = fake_data._generate(
        n_epochs=10,
        n_samples=100,
        n_categories=2,
        n_channels=32,
        time=epf.TIME,
        epoch_id=epf.EPOCH_ID,
    )

    eeg_streams = ["channel0", "channel1", "channel2", "channel3", "channel4"]
    start, stop = 30, 60
    epochs_df_centeron = epf.center_eeg(epochs_df, eeg_streams, start, stop)

    # after center on, the mean inside interval should be zero
    qstr = f"{start} <= {epf.TIME} and {epf.TIME} < {stop}"
    after_mean = epochs_df_centeron.groupby([epf.EPOCH_ID]).apply(
        lambda x: x.query(qstr)[eeg_streams].mean(axis=0)
    )

    # a is afer_mean numpy array, and b is zero array same size as a

    a = after_mean.values
    b = np.zeros(after_mean.shape)

    # np.isclose(a,b)   #all false
    # np.isclose(a,b,atol=1e-05)  #most true, but some false

    # The absolute tolerance parameter: atol=1e-04
    # TorF = np.isclose(a,b,atol=1e-04)
    TorF = np.isclose(a, b)
    assert sum(sum(TorF)) == TorF.shape[0] * TorF.shape[1]


"""
    # fake some data
    # check center_on function with fake data
    import numpy as np
    import warnings
    randval = np.random.random()
    if randval > 0.5:
        pdb.set_trace()
        raise ValueError(f'randval > 0.5 {randval}')
    else:
        warnings.warn(f'randval <= 0.5 {randval})'
"""
