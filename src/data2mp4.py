import multiprocessing as mp
import pathlib
import time
from functools import partial
from multiprocessing import Pool

import matplotlib.animation
import matplotlib.animation as animation
import matplotlib.axes
import matplotlib.axis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

data_dir = pathlib.Path(__file__).parents[1] / "data"
TEAM_MAP = pd.read_parquet(data_dir / "team_id_map.parquet").to_dict()["category"]
TEAM_MAP[-1] = "football"
REV_TEAM_MAP = {name: idx for idx, name in TEAM_MAP.items()}

colors = {
    "ARI": ["#97233F", "#000000", "#FFB612"],
    "ATL": ["#A71930", "#000000", "#A5ACAF"],
    "BAL": ["#241773", "#000000"],
    "BUF": ["#00338D", "#C60C30"],
    "CAR": ["#0085CA", "#101820", "#BFC0BF"],
    "CHI": ["#0B162A", "#C83803"],
    "CIN": ["#FB4F14", "#000000"],
    "CLE": ["#311D00", "#FF3C00"],
    "DAL": ["#003594", "#041E42", "#869397"],
    "DEN": ["#FB4F14", "#002244"],
    "DET": ["#0076B6", "#B0B7BC", "#000000"],
    "GB": ["#203731", "#FFB612"],
    "HOU": ["#03202F", "#A71930"],
    "IND": ["#002C5F", "#A2AAAD"],
    "JAX": ["#101820", "#D7A22A", "#9F792C"],
    "KC": ["#E31837", "#FFB81C"],
    "LA": ["#003594", "#FFA300", "#FF8200"],
    "LAC": ["#0080C6", "#FFC20E", "#FFFFFF"],
    "LV": ["#000000", "#A5ACAF"],
    "MIA": ["#008E97", "#FC4C02", "#005778"],
    "MIN": ["#4F2683", "#FFC62F"],
    "NE": ["#002244", "#C60C30", "#B0B7BC"],
    "NO": ["#101820", "#D3BC8D"],
    "NYG": ["#0B2265", "#A71930", "#A5ACAF"],
    "NYJ": ["#125740", "#000000", "#FFFFFF"],
    "PHI": ["#004C54", "#A5ACAF", "#ACC0C6"],
    "PIT": ["#FFB612", "#101820"],
    "SEA": ["#002244", "#69BE28", "#A5ACAF"],
    "SF": ["#AA0000", "#B3995D"],
    "TB": ["#D50A0A", "#FF7900", "#0A0A08"],
    "TEN": ["#0C2340", "#4B92DB", "#C8102E"],
    "WAS": ["#5A1414", "#FFB612"],
    "football": ["#CBB67C", "#663831"],
}


def hex_to_rgb_array(hex_color):
    """take in hex val and return rgb np array"""
    return np.array(tuple(int(hex_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)))


def ColorDistance(hex1, hex2):
    """d = {} distance between two colors(3)"""
    if hex1 == hex2:
        return 0
    rgb1 = hex_to_rgb_array(hex1)
    rgb2 = hex_to_rgb_array(hex2)
    rm = 0.5 * (rgb1[0] + rgb2[0])
    d = abs(sum((2 + rm, 4, 3 - rm) * (rgb1 - rgb2) ** 2)) ** 0.5
    return d


def ColorPairs(team1, team2):
    color_array_1 = colors[team1]
    color_array_2 = colors[team2]
    # If color distance is small enough then flip color order
    if ColorDistance(color_array_1[0], color_array_2[0]) < 500:
        return {
            team1: [color_array_1[0], color_array_1[1]],
            team2: [color_array_2[1], color_array_2[0]],
            "football": colors["football"],
        }
    else:
        return {
            team1: [color_array_1[0], color_array_1[1]],
            team2: [color_array_2[0], color_array_2[1]],
            "football": colors["football"],
        }


def get_play_by_frame_tensor(idx, team_plots, unique_teams, id_data, tracking_data, ax):
    ids, tracks = id_data[idx], tracking_data[idx]
    for scat, team in zip(team_plots, unique_teams):
        loc_data = tracks[(ids[:, 3] == team), :2]
        scat.set_offsets(loc_data)
    else:
        # center the view on the football
        ax.set_xlim(loc_data[0,0]-26.65, loc_data[0,0]+26.65)

    return team_plots


def get_teams_colors(ax, teams):

    _, unique_teams = np.unique(teams, return_index=True)
    assert len(unique_teams) == 3
    unique_teams = list(sorted(unique_teams))
    t1_num = int(teams[unique_teams[0]])
    t2_num = int(teams[unique_teams[1]])
    football = -1
    if t1_num == football:
        t1_num, t2_num = t2_num, int(teams[unique_teams[2]])
    elif t2_num == football:
        t2_num = int(teams[unique_teams[2]])
    unique_teams = (t1_num, t2_num, football)

    colors = ColorPairs(TEAM_MAP[t1_num], TEAM_MAP[t2_num])
    team_plots = []
    for i in range(3):
        face_c, edge_c = colors[TEAM_MAP[unique_teams[i]]]
        team_plots.append(ax.scatter([], [], s=40, c=face_c, edgecolors=edge_c))
    return unique_teams, team_plots


def animate_play_tensor(dataloader_input, save_loc):
    id_tens, tracking_tens, play_idx, line_of_scrim, yards_to_first = dataloader_input
    for id_data, tracking_data, idxs, los, ytf in zip(
        id_tens, tracking_tens, play_idx, line_of_scrim, yards_to_first
    ):
        gidx, pidx = idxs
        # Set up the figure for animation
        fig, ax = plt.subplots(
            # figsize=(14.4, 6.4),
            figsize=(2.24, 2.24),
            layout="constrained")
        # fig.set_figwidth(224)
        # fig.set_figheight(224)
        # colors = ['#ff5733', '#ffbd33', '#dbff33']
        ax: matplotlib.axes.Axes
        teams: torch.Tensor = id_data[0, :, 3]
        try:
            unique_teams, team_plots = get_teams_colors(ax, teams)
        except AssertionError:
            print(f"Game - {gidx}, Play - {pidx} contains fewer than 3 teams in frame 0")
            return

        # ball_start_msk = (id_data[0, :, 0] == -1)
        # ball_start_pos = tracking_data[0, ball_start_msk, 0]

        first_down_line = los + ((-1) ** int(id_data[0, 0, 4]) * (-ytf))
        ax.axvline(first_down_line, c="y", ls="-")
        # plots every 10 yards
        for i in range(2, 11):
            ax.axvline(i * 10, c="b", ls=(5, (11 - i, i)), alpha=0.35)
            ax.axvline(i * 10, c="r", ls=(5, (i, 11 - i)), alpha=0.35)
        ax.axvline(10, c="k", ls="-")
        ax.axvline(110, c="k", ls="-")
        # plot line of scrimmage
        ax.axvline(los, c="k", ls=":")
        # remove plot box, legend, splines etc
        ax.legend([]).set_visible(False)
        sns.despine(left=True, bottom=True, right=True, top=True)
        ax.set_ylabel("")
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_xticks([])

        # set the x and y graph limits to a square image around the line of scrimmage
        ax.set_xlim(los-26.65, los+26.65)
        ax.set_ylim(0, 53.3)

        ani = animation.FuncAnimation(
            fig,
            get_play_by_frame_tensor,
            len(tracking_data),
            interval=1,
            repeat=False,
            blit=True,
            fargs=(team_plots, unique_teams, id_data, tracking_data, ax),
        )
        plt.close()
        ani.save(
            save_loc / f"{gidx}-{pidx}.mp4",
            writer=animation.FFMpegWriter(fps=10, codec="h264"),
        )


if __name__ == "__main__":
    from dataset import NFLDataModule

    # for i, data in enumerate(dataloader):
    #     animate_with_saveloc(data)
    #     if i == 3:
    #         print(f"Time to write images: {time.perf_counter() - start:.2f}")
    #         exit()
    mp.set_start_method("spawn")

    for which in ("val", "test", "train"):
        save_loc = data_dir / which / "mp4_data"
        team_names = pd.read_parquet(data_dir / which / "team_id_map.parquet")
        if not save_loc.exists():
            save_loc.mkdir()
        dmod = NFLDataModule(data_dir)
        dmod.setup(which)
        dataloader = getattr(dmod, f"{which}_dataloader")()
        animate_with_saveloc = partial(animate_play_tensor, save_loc=save_loc)
        print(f"Starting MP4 creation for {which} dataset...")
        start = time.perf_counter()
        worker_count = int(0.9 * mp.cpu_count())
        with Pool(worker_count) as p:
            p.map(animate_with_saveloc, dataloader)
        print(f"Time to write images: {time.perf_counter() - start:.2f}")
