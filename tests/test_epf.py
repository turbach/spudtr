import pytest

# local HDF5 files to be deprecated in v0.0.11 with _hdf_read_epochs
from spudtr import DATA_DIR  # , P3_F, P5_F, WR_F

# Zenodo archive feather files used starting with v0.0.9
from spudtr import get_demo_df, WR_100_FEATHER, P5_1500_FEATHER
from spudtr import epf
import spudtr.fake_epochs_data as fake_data

from spudtr.epf import EPOCH_ID, TIME

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# test epochs Zenodo https://doi.org/10.5281/zenodo.3968485
# ------------------------------------------------------------
WR_DF_COLS = [
    "epoch_id",
    "time_ms",
    "sub_id",
    "eeg_artifact",
    "dblock_path",
    "log_evcodes",
    "log_ccodes",
    "dblock_srate",
    "ccode",
    "instrument",
    "repetition",
    "pres_position",
    "word_lag",
    "congruity",
    "event_code",
    "condition_id",
    "item_id",
    "category",
    "probe",
    "lle",
    "lhz",
    "MiPf",
    "LLPf",
    "RLPf",
    "LMPf",
    "RMPf",
    "LDFr",
    "RDFr",
    "LLFr",
    "RLFr",
    "LMFr",
    "RMFr",
    "LMCe",
    "RMCe",
    "MiCe",
    "MiPa",
    "LDCe",
    "RDCe",
    "LDPa",
    "RDPa",
    "LMOc",
    "RMOc",
    "LLTe",
    "RLTe",
    "LLOc",
    "RLOc",
    "MiOc",
    "A2",
    "HEOG",
    "rle",
    "rhz",
]

P5_DF_COLS = [
    "epoch_id",
    "time_ms",
    "sub_id",
    "eeg_artifact",
    "dblock_path",
    "log_evcodes",
    "log_ccodes",
    "dblock_srate",
    "ccode",
    "instrument",
    "bin",
    "click",
    "type",
    "lle",
    "lhz",
    "MiPf",
    "LLPf",
    "RLPf",
    "LMPf",
    "RMPf",
    "LDFr",
    "RDFr",
    "LLFr",
    "RLFr",
    "LMFr",
    "RMFr",
    "LMCe",
    "RMCe",
    "MiCe",
    "MiPa",
    "LDCe",
    "RDCe",
    "LDPa",
    "RDPa",
    "LMOc",
    "RMOc",
    "LLTe",
    "RLTe",
    "LLOc",
    "RLOc",
    "MiOc",
    "A2",
    "HEOG",
    "rle",
    "rhz",
]

print("downloading test epochs")
WR_100_FEATHER_DF = get_demo_df(WR_100_FEATHER)
assert WR_100_FEATHER_DF.shape == (12425, 51)
assert all(WR_100_FEATHER_DF.columns == WR_DF_COLS)

P5_1500_FEATHER_DF = get_demo_df(P5_1500_FEATHER)
assert P5_1500_FEATHER_DF.shape == (335250, 45)
assert all(P5_1500_FEATHER_DF.columns == P5_DF_COLS)

print("ok")


# @pytest.mark.parametrize(
#     "_f,h5_group", [[P3_F, "p3"], [P5_F, "p5"], [WR_F, "wr"]]
# )
# def test__hdf_read_epochs(_f, h5_group):
#     epoch_id, time = "epoch_id", "time_ms"
#     epochs_df = epf._hdf_read_epochs(
#         DATA_DIR / _f, h5_group, epoch_id=epoch_id, time=time,
#     )
#     with pytest.raises(ValueError) as excinfo:
#         epf._hdf_read_epochs(
#             DATA_DIR / _f, h5_group=None, epoch_id=epoch_id, time=time
#         )
#     assert "You have to give h5_group key" in str(excinfo.value)
#     _ = epf._epochs_QC(
#         epochs_df, epochs_df.columns.tolist(), epoch_id=epoch_id, time=time
#     )


# test default, alternative, and None epoch_id, time
@pytest.mark.parametrize(
    "_epoch_id,_time",
    [
        (epf.EPOCH_ID, epf.TIME),
        (epf.EPOCH_ID + "_ALT", epf.TIME + "_ALT"),
        pytest.param(None, None, marks=pytest.mark.xfail()),
    ],
)
def test__validate_epochs_df(_epoch_id, _time):

    # these should succeed
    epochs_df, channels = fake_data._generate(
        n_epochs=10,
        n_samples=100,
        n_categories=2,
        n_channels=32,
        time=_time,
        epoch_id=_epoch_id,
    )

    # defaults without the optional args
    if _epoch_id == epf.EPOCH_ID and _time == epf.TIME:
        epf._validate_epochs_df(epochs_df)

    # explicit args that match the data
    epf._validate_epochs_df(epochs_df, epoch_id=_epoch_id, time=_time)

    # these should fail ... mismatches
    with pytest.raises(Exception) as excinfo:
        xfail_epochs_df, channels = fake_data._generate(
            n_epochs=10,
            n_samples=100,
            n_categories=2,
            n_channels=32,
            time="xfail",
            epoch_id="xfail",
        )
        epf._validate_epochs_df(
            xfail_epochs_df, epoch_id=_epoch_id, time=_time
        )
    if not (
        excinfo.type is ValueError
        and "column not found" in excinfo.value.args[0]
    ):
        raise Exception(f"uncaught exception {excinfo}")


# test by using one file
def test_epochs_QC():
    epochs_df = WR_100_FEATHER_DF.copy()
    data_streams = ["MiPf", "MiCe", "MiPa", "MiOc"]
    epf._epochs_QC(
        epochs_df, data_streams, epoch_id="epoch_id", time="time_ms"
    )


def test_epochs_QC_fails():
    epochs_df = WR_100_FEATHER_DF.copy()
    data_streams = ["MiPf", "MiCe", "MiPa", "MiOc"]

    with pytest.raises(ValueError) as excinfo:
        epochs_df1 = [1, 2]
        epf._epochs_QC(epochs_df1, data_streams)
    assert "epochs_df must be a Pandas DataFrame." in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        data_streams1 = set(data_streams)
        epf._epochs_QC(epochs_df, data_streams1)
    assert "data_streams should be a list of strings." in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        data_streams1 = ["A"]
        epf._epochs_QC(epochs_df, data_streams1)
    assert (
        "data_streams should all be present in the epochs dataframe,"
        in str(excinfo.value)
    )


def test_raises_error_on_duplicate_channels():

    epochs_table, channels = fake_data._generate(
        n_epochs=10,
        n_samples=100,
        n_categories=2,
        n_channels=32,
        time=TIME,
        epoch_id=EPOCH_ID,
    )
    dupe_channel = channels[0]
    dupe_column = epochs_table[dupe_channel]
    bad_epochs_table = pd.concat([epochs_table, dupe_column], axis=1)

    with pytest.raises(ValueError) as excinfo:
        epf._epochs_QC(bad_epochs_table, channels)
    assert "Duplicate column names" in str(excinfo.value)


def test_epochs_unequal_snapshots():

    epochs_table, channels = fake_data._generate(
        n_epochs=10,
        n_samples=100,
        n_categories=2,
        n_channels=32,
        time=TIME,
        epoch_id=EPOCH_ID,
    )

    epochs_table.drop(epochs_table.index[42], inplace=True)
    with pytest.raises(ValueError) as excinfo:
        epf._epochs_QC(epochs_table, channels)
    assert "differs from previous snapshot" in str(excinfo.value)


def test_duplicate_values_of_epoch_id():
    epochs_table, channels = fake_data._generate(
        n_epochs=10,
        n_samples=100,
        n_categories=2,
        n_channels=32,
        time=TIME,
        epoch_id=EPOCH_ID,
    )
    epochs_table.loc[epochs_table[EPOCH_ID] == 16, [EPOCH_ID]] = 18

    with pytest.raises(ValueError) as excinfo:
        epf._epochs_QC(epochs_table, channels)
    assert "Duplicate values of epoch_id" in str(excinfo.value)


@pytest.mark.parametrize(
    "data_streams",
    [
        ["channel0", "channel1"],
        pytest.param("channel_xfail", marks=pytest.mark.xfail(strict=True)),
        pytest.param("[channel_xfail]", marks=pytest.mark.xfail(strict=True)),
    ],
)
@pytest.mark.parametrize(
    "_epoch_id",
    [
        "epoch_id",
        pytest.param("epoch_id_xfail", marks=pytest.mark.xfail(strict=True)),
    ],
)
@pytest.mark.parametrize(
    "_time",
    ["time", pytest.param("time_xfail", marks=pytest.mark.xfail(strict=True))],
)
def test_check_epochs(data_streams, _epoch_id, _time):
    """test UI wrapper for epochs QC"""

    # build default fake data
    epochs_table, channels = fake_data._generate(
        n_epochs=10,
        n_samples=100,
        n_categories=2,
        n_channels=2,
        time=TIME,
        epoch_id=EPOCH_ID,
    )

    # some succeed, some xfail
    epf.check_epochs(epochs_table, data_streams, _epoch_id, _time)


def test__find_subscript():
    time = "time"
    epochs_df = fake_data._get_df()
    times = epochs_df[time].unique()
    start = 6
    stop = 8
    with pytest.raises(ValueError) as excinfo:
        istart, istop = epf._find_subscript(times, start, stop)
    assert "start is too large" in str(excinfo.value)
    start = -3
    stop = -1
    with pytest.raises(ValueError) as excinfo:
        istart, istop = epf._find_subscript(times, start, stop)
    assert "stop is too small" in str(excinfo.value)
    start = 4
    stop = 2
    with pytest.raises(ValueError) as excinfo:
        istart, istop = epf._find_subscript(times, start, stop)
    assert "Bad rescaling slice" in str(excinfo.value)


def test_center_eeg():
    epochs_df = fake_data._get_df()

    # save a copy for demonstration
    # check_epochs_df = epochs_df.copy()

    # center two columns in place (for demonstration start and stop are epoch row indexes not times)
    eeg_streams = ["x", "z"]
    epoch_id = "epoch_id"
    time = "time"
    start = 0
    stop = 2

    # verify centering == 0 and report failures
    n_times = len(epochs_df[time].unique())
    n_epochs = len(epochs_df[epoch_id].unique())
    times = epochs_df[time].unique()
    istart, istop = epf._find_subscript(times, start, stop)
    center_idxs = np.array(
        [
            np.arange(istart + (i * n_times), istop + (i * n_times))
            for i in range(n_epochs)
        ]
    ).flatten()
    centered_epochs_df = epf.center_eeg(
        epochs_df, eeg_streams, start, stop, epoch_id=EPOCH_ID, time="time"
    )
    zero_mns = (
        centered_epochs_df.iloc[center_idxs, :]
        .groupby(epoch_id)[eeg_streams]
        .mean()
    )
    assert np.allclose(0, zero_mns)
    epf._epochs_QC(
        centered_epochs_df, eeg_streams, epoch_id=epoch_id, time=time
    )


def test_drop_bad_epochs():
    epoch_id = "epoch_id"
    time = "time_ms"

    epochs_df = WR_100_FEATHER_DF.copy()
    bads_column = "eeg_artifact"

    epochs_df_good = epf.drop_bad_epochs(
        epochs_df, bads_column, epoch_id=epoch_id, time=time
    )
    epochs_df_good["new_col"] = 0

    # get the group of time == 0
    group = epochs_df.groupby([time]).get_group(0)
    good_idx = list(group[epoch_id][group[bads_column] == 0])
    epochs_df_bad = epochs_df[~epochs_df[epoch_id].isin(good_idx)]
    assert (
        epochs_df_good.shape[0] + epochs_df_bad.shape[0] == epochs_df.shape[0]
    )
    epochs_df_good = epf.drop_bad_epochs(
        epochs_df, bads_column, epoch_id, time
    )
    epf._epochs_QC(
        epochs_df_good,
        epochs_df_good.columns.tolist(),
        epoch_id=epoch_id,
        time=time,
    )


def test_re_reference():

    # create a fake data
    epochs_df = pd.DataFrame(
        np.array([[0, -3, 1, 2, 3], [0, -2, 4, 5, 6], [0, -1, 7, 8, 9]]),
        columns=[EPOCH_ID, TIME, "a", "b", "c"],
    )

    eeg_streams = ["b", "c"]

    ref = ["a"]
    ref_type = "linked_pair"
    br_epochs_df = epf.re_reference(epochs_df, eeg_streams, ref, ref_type)
    assert list(br_epochs_df.b) == [1.5, 3.0, 4.5]
    ref = ["a"]
    ref_type = "new_common"
    br_epochs_df = epf.re_reference(epochs_df, eeg_streams, ref, ref_type)
    assert list(br_epochs_df.b) == [1, 1, 1]
    ref = ["a", "b"]
    ref_type = "common_average"
    br_epochs_df = epf.re_reference(epochs_df, eeg_streams, ref, ref_type)
    assert list(br_epochs_df.b) == [0.5, 0.5, 0.5]

    with pytest.raises(ValueError) as excinfo:
        ref1 = set(ref)
        br_epochs_df = epf.re_reference(epochs_df, eeg_streams, ref1, ref_type)
    assert "ref should be a list of strings" in str(excinfo.value)

    epf._epochs_QC(br_epochs_df, eeg_streams, epoch_id=EPOCH_ID, time=TIME)


@pytest.mark.parametrize(
    "eeg_streams,ref,ref_type,expected",
    [
        (["b", "c"], ["a"], "linked_pair", [1.5, 3.0, 4.5]),
        (["b", "c"], ["a"], "new_common", [1, 1, 1]),
        (["b", "c"], ["a", "b"], "common_average", [0.5, 0.5, 0.5]),
        pytest.param(
            ["b", "c"],
            ["a", "b"],
            "Xcommon_average",
            [0.5, 0.5, 0.5],
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ],
)
def test_re_reference_2(eeg_streams, ref, ref_type, expected):

    # create a fake data
    epochs_df = pd.DataFrame(
        np.array([[0, -3, 1, 2, 3], [0, -2, 4, 5, 6], [0, -1, 7, 8, 9]]),
        columns=[EPOCH_ID, TIME, "a", "b", "c"],
    )

    br_epochs_df = epf.re_reference(epochs_df, eeg_streams, ref, ref_type)

    assert list(br_epochs_df.b) == expected
    epf._epochs_QC(br_epochs_df, eeg_streams, epoch_id=EPOCH_ID, time=TIME)


@pytest.mark.parametrize(
    "trim_edges,df_shape", [(False, (335_250, 45)), (True, (253_896, 45))]
)
def test_fir_filter_epochs(trim_edges, df_shape):

    epoch_id = "epoch_id"
    time = "time_ms"

    epochs_df = P5_1500_FEATHER_DF.copy()

    eeg_cols = ["MiPf", "MiCe", "MiCe", "MiOc"]
    epf.check_epochs(
        epochs_df, data_streams=eeg_cols, epoch_id=epoch_id, time=time
    )

    _fp = dict(
        ftype="lowpass",
        cutoff_hz=12.5,
        width_hz=5,
        ripple_db=60,
        window="kaiser",
        sfreq=250,
    )

    filt_test_df = epf.fir_filter_epochs(
        epochs_df,
        eeg_cols,
        trim_edges=trim_edges,
        epoch_id=epoch_id,
        time=time,
        **_fp,
    )
    epf.check_epochs(
        filt_test_df, data_streams=eeg_cols, epoch_id=epoch_id, time=time
    )

    filt_times = filt_test_df[time].unique()
    if trim_edges is False:
        assert all(filt_times == epochs_df[time].unique())
        assert filt_test_df.shape == df_shape
    else:
        assert filt_test_df.shape == df_shape

        # slice epochs_df to match the filtered one
        qstr = f"{time} in @filt_times"
        epochs_df = epochs_df.query(qstr)
        assert filt_test_df.shape == df_shape == epochs_df.shape

    # check the filter changed all and only selected columns
    assert isinstance(filt_test_df, pd.DataFrame)
    assert all(filt_test_df.columns == epochs_df.columns)
    for col in epochs_df.columns:
        if col in eeg_cols:
            assert not all(epochs_df[col] == filt_test_df[col])
        else:
            assert all(epochs_df[col] == filt_test_df[col])
