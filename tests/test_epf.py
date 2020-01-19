from pathlib import Path
import pytest

import spudtr.epf as epf
import spudtr.fake_epochs_data as fake_data

# import epf as epf
# import fake_epochs_data as fake_data

import numpy as np
import pandas as pd

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
def test__hdf_read_epochs(_f, h5_group):
    epochs_df = epf._hdf_read_epochs(TEST_DATA_DIR / _f, h5_group)
    with pytest.raises(ValueError) as excinfo:
        epf._hdf_read_epochs(TEST_DATA_DIR / _f, h5_group=None)
    assert "You have to give h5_group key" in str(excinfo.value)


def test__validate_epochs_df():

    epochs_df, channels = fake_data._generate(
        n_epochs=10,
        n_samples=100,
        n_categories=2,
        n_channels=32,
        time="Time",
        epoch_id="Epoch_idx",
    )
    epf._validate_epochs_df(epochs_df)

    with pytest.raises(ValueError) as excinfo:
        epf._validate_epochs_df(epochs_df, epoch_id="Epoch_idx", time="T")
    assert "time column T not found" in str(excinfo.value)


# test by using one file
def test_epochs_QC():
    _f1, h5_group1 = "sub000wr.epochs.h5", "wr"
    epochs_df = epf._hdf_read_epochs(TEST_DATA_DIR / _f1, h5_group1)
    eeg_streams = ["MiPf", "MiCe", "MiPa", "MiOc"]
    epf._epochs_QC(epochs_df, eeg_streams)


def test_epochs_QC_fails():
    _f1, h5_group1 = "sub000wr.epochs.h5", "wr"
    epochs_df = epf._hdf_read_epochs(TEST_DATA_DIR / _f1, h5_group1)
    eeg_streams = ["MiPf", "MiCe", "MiPa", "MiOc"]

    with pytest.raises(ValueError) as excinfo:
        epochs_df1 = [1, 2]
        epf._epochs_QC(epochs_df1, eeg_streams)
    assert "epochs_df must be a Pandas DataFrame." in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        eeg_streams1 = set(eeg_streams)
        epf._epochs_QC(epochs_df, eeg_streams1)
    assert "eeg_streams should be a list of strings." in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        eeg_streams1 = ["A"]
        epf._epochs_QC(epochs_df, eeg_streams1)
    assert "eeg_streams should all be present in the epochs dataframe," in str(
        excinfo.value
    )


def test_raises_error_on_duplicate_channels():

    epochs_table, channels = fake_data._generate(
        n_epochs=10,
        n_samples=100,
        n_categories=2,
        n_channels=32,
        time="Time",
        epoch_id="Epoch_idx",
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
        time="Time",
        epoch_id="Epoch_idx",
    )

    epochs_table.drop(epochs_table.index[42], inplace=True)
    with pytest.raises(ValueError) as excinfo:
        epf._epochs_QC(epochs_table, channels)
    assert "differs from previous snapshot" in str(excinfo.value)


def test_Duplicate_values_of_epoch_id():
    epochs_table, channels = fake_data._generate(
        n_epochs=10,
        n_samples=100,
        n_categories=2,
        n_channels=32,
        time="Time",
        epoch_id="Epoch_idx",
    )
    epochs_table.loc[epochs_table["Epoch_idx"] == 16, ["Epoch_idx"]] = 18

    with pytest.raises(ValueError) as excinfo:
        epf._epochs_QC(epochs_table, channels)
    assert "Duplicate values of epoch_id" in str(excinfo.value)


def test_center_on():
    _f1, h5_group1 = "sub000wr.epochs.h5", "wr"
    epochs_df = epf._hdf_read_epochs(TEST_DATA_DIR / _f1, h5_group1)
    eeg_streams = ["MiPf", "MiCe", "MiPa", "MiOc"]
    start, stop = -50, 300

    with pytest.raises(ValueError) as excinfo:
        epf.center_eeg(epochs_df, eeg_streams, start, stop, atol=1e-6)
    assert "center_on is not successful" in str(excinfo.value)


def test_center_eeg_start_stop_time():
    epochs_df, channels = fake_data._generate(
        n_epochs=10,
        n_samples=100,
        n_categories=2,
        n_channels=32,
        time="Time",
        epoch_id="Epoch_idx",
    )
    start, stop = -999, 999
    eeg_streams = ["channel0", "channel1"]
    epf.center_eeg(epochs_df, eeg_streams, start, stop, atol=1e-04)


def test_drop_bad_epochs():
    _f1, h5_group1 = "sub000wr.epochs.h5", "wr"
    epochs_df = epf._hdf_read_epochs(TEST_DATA_DIR / _f1, h5_group1)
    epoch_id = "Epoch_idx"
    time = "Time"
    art_col = "log_flags"

    epochs_df_good = epf.drop_bad_epochs(epochs_df, art_col, epoch_id, time)
    epochs_df_good["new_col"] = 0

    # get the group of time == 0
    group = epochs_df.groupby([time]).get_group(0)
    good_idx = list(group[epoch_id][group[art_col] == 0])
    epochs_df_bad = epochs_df[~epochs_df[epoch_id].isin(good_idx)]
    assert (
        epochs_df_good.shape[0] + epochs_df_bad.shape[0] == epochs_df.shape[0]
    )
    epochs_df_good = epf.drop_bad_epochs(epochs_df)


def test_re_reference():

    # create a fake data
    epochs_df = pd.DataFrame(
        np.array([[0, -3, 1, 2, 3], [0, -2, 4, 5, 6], [0, -1, 7, 8, 9]]),
        columns=["Epoch_idx", "Time", "a", "b", "c"],
    )

    eeg_streams = ["b", "c"]

    rs = ["a"]
    ref_type = "bimastoid"
    br_epochs_df = epf.re_reference(epochs_df, eeg_streams, rs, ref_type)
    assert list(br_epochs_df.b) == [1.5, 3.0, 4.5]
    rs = ["a"]
    ref_type = "new_common"
    br_epochs_df = epf.re_reference(epochs_df, eeg_streams, rs, ref_type)
    assert list(br_epochs_df.b) == [1, 1, 1]
    rs = ["a", "b"]
    ref_type = "common_average"
    br_epochs_df = epf.re_reference(epochs_df, eeg_streams, rs, ref_type)
    assert list(br_epochs_df.b) == [0.5, 0.5, 0.5]

    with pytest.raises(ValueError) as excinfo:
        rs1 = set(rs)
        br_epochs_df = epf.re_reference(epochs_df, eeg_streams, rs1, ref_type)
    assert "rs should be a list of strings" in str(excinfo.value)


@pytest.mark.parametrize(
    "eeg_streams,rs,ref_type,expected",
    [
        (["b", "c"], ["a"], "bimastoid", [1.5, 3.0, 4.5]),
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
def test_re_reference_2(eeg_streams, rs, ref_type, expected):

    # create a fake data
    epochs_df = pd.DataFrame(
        np.array([[0, -3, 1, 2, 3], [0, -2, 4, 5, 6], [0, -1, 7, 8, 9]]),
        columns=["Epoch_idx", "Time", "a", "b", "c"],
    )

    br_epochs_df = epf.re_reference(epochs_df, eeg_streams, rs, ref_type)

    assert list(br_epochs_df.b) == expected
