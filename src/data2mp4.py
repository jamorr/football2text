from functools import partial
import io
import time
from torchvision.transforms.functional import pil_to_tensor
import warnings
from PIL import Image
import matplotlib.animation
import matplotlib.axes
import matplotlib.axis
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import multiprocessing as mp
from multiprocessing import Pool

import torch

def get_play_by_frame_tensor(idx, team_plots, unique_teams, id_data, tracking_data):
    ids, tracks = id_data[idx], tracking_data[idx]
    for scat, team in zip(team_plots, unique_teams):
        scat.set_offsets(tracks[(ids[:, 3] == team), :2])
    return team_plots

def animate_play_tensor(dataloader_input, save_loc):
    id_tens, tracking_tens, play_idx = dataloader_input
    for id_data, tracking_data, idxs in zip(id_tens, tracking_tens, play_idx):
        gidx, pidx = idxs
        # Set up the figure for animation
        fig, ax = plt.subplots(figsize=(14.4, 6.4), layout="constrained")
        colors = ['#ff5733', '#ffbd33', '#dbff33']

        teams = id_data[
            0, :, 3
        ]
        team_plots = []
        for i in range(3):
            team_plots.append(ax.scatter([], [], s=100, c=colors[i]))

        los_msk = (id_data[0, :, 0] == -1)

        los = tracking_data[0, los_msk, 0]
        ax.axvline(los, c="k", ls=":")

        # plots a simple end zone
        for i in range(2, 10):
            ax.axvlin(i*10, c="b", ls="-")
        ax.axvline(10, c="k", ls="-")
        ax.axvline(110, c="k", ls="-")


        # takes out the legend (if you leave this, you'll get an annoying legend)
        ax.legend([]).set_visible(False)

        # takes out the left, top, and right borders on the graph
        sns.despine(left=True, bottom=True, right=True, top=True)


        # no y axis label
        ax.set_ylabel("")

        # no y axis tick marks
        ax.set_yticks([])
        # no x axis label
        ax.set_xlabel("")
        # no x axis tick marks
        ax.set_xticks([])

        # set the x and y graph limits to the entire field (from kaggle BDB page)
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 53.3)

        unique_teams = np.unique(teams)
        ani = animation.FuncAnimation(
            fig, get_play_by_frame_tensor, len(tracking_data),
            interval=1, repeat=False, blit=True,
            fargs=(team_plots, unique_teams, id_data, tracking_data)
        )
        plt.close()
        start = time.perf_counter()
        ani.save(save_loc/f"{gidx}-{pidx}.mp4", writer=animation.FFMpegWriter(fps=10, codec='h264'))
        print(f"Time to write image: {time.perf_counter() - start:.2f}")


if __name__ == '__main__':
    from dataset import NFLDataModule
    import pathlib
    which = "val"
    data_dir = pathlib.Path(__file__).parents[1]/"data"
    save_loc = data_dir/which/"mp4_data"
    team_names = pd.read_parquet(data_dir/which/"team_id_map.parquet")
    if not save_loc.exists():
        save_loc.mkdir()
    dmod = NFLDataModule(data_dir)
    dmod.setup(which)
    dataloader = dmod.val_dataloader()
    # for i, (id_data, tracking_data, play_idx) in enumerate(dataloader):
    #     start = time.perf_counter()
    #     animate_play_tensor(id_data, tracking_data, play_idx, save_loc,)
    #     print(f"Time per gif: {time.perf_counter()-start:.2f}")
    worker_count = int(0.9 * mp.cpu_count())
    animate_with_saveloc = partial(animate_play_tensor, save_loc=save_loc)
    mp.set_start_method('spawn')
    with Pool(worker_count) as p:
        p.map(animate_with_saveloc, dataloader)


