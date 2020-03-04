import numpy as np
import pandas as pd
import mne
from collections import OrderedDict


def streams2mne_digmont(eeg_streams):

    cap26 = pd.read_csv(
        "/home/qiz001/zq/Projects/spudtr/spudtr/data/cap26.csv"
    )
    missing_streams = set(eeg_streams) - set(cap26['stream'])
    if missing_streams:
        raise ValueError(f"eeg_streams not found in cap: {missing_streams}")

    df = cap26.set_index('stream').loc[eeg_streams, :].reset_index()
    ch_names = df.stream.to_list()
    pos = df[['x', 'y', 'z']].values
    dig_ch_pos = OrderedDict(zip(ch_names, pos))
    montage = mne.channels.make_dig_montage(
        ch_pos=dig_ch_pos, coord_frame='head'
    )
    return montage


def spudtr2mne(epochs_df, eeg_streams, time, epoch_id, sfreq):

    montage = streams2mne_digmont(eeg_streams)

    # create mne epochs from EpochsArray and show them
    info = mne.create_info(
        montage.ch_names, sfreq=sfreq, ch_types="eeg", montage=montage
    )

    epochs_data = []
    n_epochs_begin = 0
    n_epochs_end = max(epochs_df[epoch_id])
    n_epochs = n_epochs_end - n_epochs_begin

    # import pdb; pdb.set_trace()
    for epoch_i in range(n_epochs_begin, n_epochs_end):
        epoch1 = epochs_df[info['ch_names']][
            epochs_df.epoch_id == epoch_i
        ].to_numpy()
        epochs_data.append(epoch1.T)
    epochs = mne.EpochsArray(epochs_data, info=info)
    return epochs
