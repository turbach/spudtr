#!/usr/bin/env python3

from matplotlib import pyplot as plt
import pandas as pd
import mne
import tkinter
import tkinter.filedialog
import yaml
from tkinter import font as tkFont
from spudtr import mneutils, DATA_DIR

root = tkinter.Tk()
helv16 = tkFont.Font(family="Helvetica", size=16, weight=tkFont.BOLD)


def OpenFile():
    global f_eeg
    f_eeg = tkinter.filedialog.askopenfilename(
        parent=root,
        initialdir=DATA_DIR,
        title="Choose file",
        filetypes=[("feather files", "*.feather"), ("all files", "*.*")],
    )
    print(f_eeg)


config_file = DATA_DIR / "default.yml"


def OpenYaml_config():
    global config_file
    config_file = tkinter.filedialog.askopenfilename(
        parent=root,
        initialdir=DATA_DIR,
        title="Choose yaml file",
        filetypes=[("yaml files", "*.yml"), ("all files", "*.*")],
    )
    print(config_file)


b1 = tkinter.Button(
    root,
    text="Select a eeg file. *",
    height=3,
    width=20,
    font=helv16,
    fg="red",
    command=OpenFile,
)
b1.pack(fill="x")
b2 = tkinter.Button(
    root,
    text="Select a config yaml file. *",
    height=3,
    width=20,
    font=helv16,
    fg="red",
    command=OpenYaml_config,
)
b2.pack(fill="x")

epochs_plot = 0


def quit(n):
    global epochs_plot
    epochs_plot = n
    root.destroy()


button_plot1 = tkinter.Button(
    root,
    text="Create evokeds plot",
    height=3,
    width=20,
    font=helv16,
    fg="yellow",
    command=lambda: quit(1),
)
button_plot1.pack(fill="x")

button_plot2 = tkinter.Button(
    root,
    text="Create compare evokeds plot",
    height=3,
    width=20,
    font=helv16,
    fg="blue",
    command=lambda: quit(2),
)
button_plot2.pack(fill="x")

button_plot3 = tkinter.Button(
    root,
    text="Create epochs plot",
    height=3,
    width=20,
    font=helv16,
    fg="green",
    command=lambda: quit(3),
)
button_plot3.pack(fill="x")

button_plot4 = tkinter.Button(
    root,
    text="Create epochs image plot",
    height=3,
    width=20,
    font=helv16,
    fg="orange",
    command=lambda: quit(4),
)
button_plot4.pack(fill="x")

button_plot5 = tkinter.Button(
    root,
    text="Plot psd topomap",
    height=3,
    width=20,
    font=helv16,
    fg="black",
    command=lambda: quit(5),
)
button_plot5.pack(fill="x")

button_plot6 = tkinter.Button(
    root,
    text="Evokeds animation",
    height=3,
    width=20,
    font=helv16,
    fg="magenta",
    command=lambda: quit(6),
)
button_plot6.pack(fill="x")


root.wm_title("Spudtr epochs plots")
root.geometry("320x618")
root.mainloop()

with open(config_file, "r") as stream:
    config_data = yaml.safe_load(stream)

eeg_streams = config_data["eeg_streams"]
time = config_data["time"]
epoch_id = config_data["epoch_id"]
sfreq = config_data["sfreq"]
time_unit = config_data["time_unit"]
categories = config_data["categories"]
time_stamp = config_data["time_stamp"]
key = config_data["key"]

epochs = mneutils.read_spudtr_epochs(
    f_eeg, eeg_streams, categories, time_stamp, epoch_id, time, time_unit
)

mne_event_id = epochs.event_id
events_list = list(epochs.event_id.keys())

evokeds_dict = {cond: epochs[cond].average() for cond in mne_event_id}

if epochs_plot == 1:
    fig, ax = plt.subplots(len(evokeds_dict), 1, figsize=(10, len(evokeds_dict) * 3))
    if len(evokeds_dict) == 1:
        for x in evokeds_dict:
            evokeds_dict[x].plot(
                spatial_colors=True,
                gfp=True,
                picks="eeg",
                time_unit="ms",
                axes=ax,
                show=False,
            )
        ax.set_title(x)
    else:
        n = 0
        for x in evokeds_dict:
            n = n + 1
            evokeds_dict[x].plot(
                spatial_colors=True,
                gfp=True,
                picks="eeg",
                time_unit="ms",
                axes=ax[n - 1],
                show=False,
            )
            ax[n - 1].set_title(x)
    plt.show()

elif epochs_plot == 2:
    # compare evokeds for all evokes
    mne.viz.plot_compare_evokeds(
        evokeds_dict, picks="eeg", split_legend=False, axes="topo"
    )
elif epochs_plot == 3:
    for x in events_list:
        epochs[x].plot(
            picks="eeg", scalings="auto", show=False, n_channels=10, n_epochs=10,
        )
    plt.show()
elif epochs_plot == 4:
    # plotting channelwise information arranged into a shape of the channel array.
    for x in events_list:
        epochs[x].plot_topo_image(
            vmin=-3e7,
            vmax=2.0e7,
            title=x,
            sigma=2.0,
            fig_facecolor="w",
            font_color="k",
            show=False,
        )
    plt.show()
elif epochs_plot == 5:
    for x in events_list:
        fig = epochs[x].plot_psd_topomap(
            ch_type="eeg", normalize=True, cmap="RdBu_r", show=False
        )
        fig.suptitle(x)
        fig.set_figheight(2)
    plt.show()

elif epochs_plot == 6:
    times = epochs_df[time].unique() / 1000
    for x in evokeds_dict:
        fig, anim = evokeds_dict[x].animate_topomap(
            ch_type="eeg", times=times, frame_rate=2, time_unit="ms", show=False
        )
        fig.suptitle(x)
    plt.show()
