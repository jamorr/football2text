import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# for mpl animation
import matplotlib.animation as animation
# from matplotlib import rc
# rc('animation', html='html5')

def get_play_by_frame(fid, ax, los, one_play):
    """
    take one frame from one play, plot a scatter plot image

    inputs:
      fid: frame ID
      ax: current matplotlib ax
      los: line of scrimmage (for aesthetics)
      one_play: pandas dataframe for one play

    output:
      seaborn axis level scatter plot
    """
    # clear current axis (or else you'll have a tracer effect)
    ax.cla()

    # get game and play IDs
    gid = one_play["gameId"].unique()[0]
    pid = one_play["playId"].unique()[0]

    # isolates a given frame within one play
    one_frame = one_play.loc[one_play["frameId"] == fid]

    # create a scatter plot, hard coded dot size to 100
    fig1 = sns.scatterplot(x="x", y="y", data=one_frame, hue="club", ax=ax, s=100)

    # plots line of scrimmage
    fig1.axvline(los, c="k", ls=":")

    # plots a simple end zone
    fig1.axvline(0, c="k", ls="-")
    fig1.axvline(100, c="k", ls="-")

    # game and play IDs as the title
    fig1.set_title(f"game {gid} play {pid}")

    # takes out the legend (if you leave this, you'll get an annoying legend)
    fig1.legend([]).set_visible(False)

    # takes out the left, top, and right borders on the graph
    sns.despine(left=True)

    # no y axis label
    fig1.set_ylabel("")

    # no y axis tick marks
    fig1.set_yticks([])

    # set the x and y graph limits to the entire field (from kaggle BDB page)
    fig1.set_xlim(-10, 110)
    fig1.set_ylim(0, 54)


def get_play_by_frame_tensor(frame_index, ax, los, tracking_data, id_data):

    ax.cla()
    colors = ['#747FE3', '#8EE35D', '#E37346']
    # isolates a given frame within one play
    player_positions = tracking_data[
        frame_index, :, :2
    ]  # Assuming the second to last index is players

    teams = id_data[
        frame_index, :, 3
    ]

    fig1 = sns.scatterplot(x=player_positions[:, 0], y=player_positions[:, 1], hue=teams, palette=colors, ax=ax, s=100)
    # plots line of scrimmage
    fig1.axvline(los, c="k", ls=":")

    # plots a simple end zone
    fig1.axvline(0, c="k", ls="-")
    fig1.axvline(100, c="k", ls="-")


    # takes out the legend (if you leave this, you'll get an annoying legend)
    fig1.legend([]).set_visible(False)

    # takes out the left, top, and right borders on the graph
    sns.despine(left=True)

    # no y axis label
    fig1.set_ylabel("")

    # no y axis tick marks
    fig1.set_yticks([])

    # set the x and y graph limits to the entire field (from kaggle BDB page)
    fig1.set_xlim(-10, 110)
    fig1.set_ylim(0, 54)


    # for player_position in player_positions:
    #     ax.plot(
    #         player_position[0],
    #         player_position[1],
    #         "o",
    #         color="red" if player_position[-1] == "home" else "blue",
    #     )

    # # Draw the ball
    # ax.plot(ball_position[0], ball_position[1], "o", color="brown")


def animate_play_tensor(id_tens, tracking_tens, play_idx, save_loc):
    for id_data, tracking_data, idxs in zip(id_tens, tracking_tens, play_idx):
        gidx, pidx = idxs
        # Set up the figure for animation
        fig, ax = plt.subplots(figsize=(14.4, 6.4))
        num_frames = id_data.shape[0]
        los_msk = (id_data[0, :, 0] == -1)
        los = tracking_data[0, los_msk, 0]
        ani = animation.FuncAnimation(
            fig,
            get_play_by_frame_tensor,
            frames=num_frames,
            fargs=(ax, los, tracking_data, id_data),
            interval=1,
            repeat=False,
        )
        plt.close()

        ani.save(save_loc/f"{gidx}-{pidx}.gif", writer="imagemagick", fps=10)

    return ani


if __name__ == '__main__':
    from dataset import NFLDataModule
    import pathlib
    which = "val"
    data_dir = pathlib.Path("../data")
    save_loc = data_dir/which/"gifs"
    team_names = pd.read_parquet(data_dir/which/"team_id_map.parquet")
    if not save_loc.exists():
        save_loc.mkdir()
    dmod = NFLDataModule(data_dir)
    dmod.setup(which)
    dataloader = dmod.val_dataloader()
    for id_data, tracking_data, play_idx in dataloader:
        start = time.perf_counter()
        animate_play_tensor(id_data, tracking_data, play_idx, save_loc,)
        print(f"Time per gif: {time.perf_counter()-start:.2f}")
        break

