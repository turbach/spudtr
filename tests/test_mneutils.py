import pytest
import numpy as np
import pandas as pd
import patsy
from spudtr import DATA_DIR, mneutils


@pytest.mark.parametrize(
    "_categories",
    [(["stim"]), pytest.param("stimx", marks=pytest.mark.xfail(strict=True)),],
)
def test__categories2event_id_data(_categories):

    # gold standard
    event_id_1_head = np.array(
        [
            [30500, 0, 1],
            [30875, 0, 1],
            [31250, 0, 1],
            [31625, 0, 1],
            [32000, 0, 1],
        ]
    )

    event_id_2_head = np.array(
        [[125, 0, 2], [500, 0, 2], [875, 0, 2], [1250, 0, 2], [1625, 0, 2],]
    )

    _f, _key = DATA_DIR / "gh_sub000p3.epochs.h5", "p3"
    epochs_df = pd.read_hdf(_f, _key).query("stim in ['target', 'standard']")
    epoch_id = "epoch_id"
    time = "time_ms"
    time_stamp = 0

    mne_event_id, mne_events = mneutils._categories2eventid(
        epochs_df, _categories, epoch_id, time, time_stamp
    )

    assert mne_event_id == {"stim[standard]": 1, "stim[target]": 2}

    # verify the head slices for both event ids
    assert np.array_equal(
        event_id_1_head,
        mne_events[[v for v in map(lambda x: x[2] == 1, mne_events)]][0:5],
    )

    assert np.array_equal(
        event_id_2_head,
        mne_events[[v for v in map(lambda x: x[2] == 2, mne_events)]][0:5],
    )

    # verify the original epoch_df row labels
    assert all(
        epochs_df.iloc[[ev[0] for ev in event_id_1_head], :]["stim"]
        == "standard"
    )
    assert all(
        epochs_df.iloc[[ev[0] for ev in event_id_2_head], :]["stim"]
        == "target"
    )


cat2event_id_params = [
    (dict(a=1), [{"a[a1]": 1}, [[0, 0, 1]]]),
    (dict(a=2), [{"a[a1]": 1, "a[a2]": 2}, [[0, 0, 1], [1, 0, 2]]]),
    (
        dict(a=3),
        [
            {"a[a1]": 1, "a[a2]": 2, "a[a3]": 3},
            [[0, 0, 1], [1, 0, 2], [2, 0, 3]],
        ],
    ),
    (
        dict(a=1, b=2),
        [{"a[a1]:b[b1]": 1, "a[a1]:b[b2]": 2}, [[0, 0, 1], [1, 0, 2]]],
    ),
    (
        dict(a=2, b=3),
        [
            {
                "a[a1]:b[b1]": 1,
                "a[a2]:b[b1]": 2,
                "a[a1]:b[b2]": 3,
                "a[a2]:b[b2]": 4,
                "a[a1]:b[b3]": 5,
                "a[a2]:b[b3]": 6,
            },
            [[0, 0, 1], [1, 0, 3], [2, 0, 5], [3, 0, 2], [4, 0, 4], [5, 0, 6]],
        ],
    ),
]


@pytest.mark.parametrize("_factors,_result", cat2event_id_params)
def test__categories2event_id(_factors, _result):

    # build a minimal spudtr epochs_df with one row per event
    epoch_id = "epoch_id"
    time = "time"
    time_stamp = 0

    events_df = pd.DataFrame(patsy.balanced(**_factors))
    events_df[epoch_id] = range(len(events_df))
    events_df[time] = time_stamp  # one event per epoch is atypical

    categories = list(_factors.keys())
    if len(categories) == 1:
        categories = categories[0]

    mne_event_id, mne_events = mneutils._categories2eventid(
        events_df, categories, epoch_id, time, time_stamp
    )
    assert mne_event_id == _result[0]
    assert np.array_equal(mne_events, _result[1])
