import argparse
import glob
import os
import pathlib
import pickle
from argparse import Namespace
from functools import partial
from multiprocessing import Pool
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch_geometric as pyg
from torch_geometric.data import Data
from torch_geometric.data.lightning import (
    LightningDataset,
    LightningLinkData,
    LightningNodeData,
)
from torch_geometric.utils.mask import mask_select

# @inproceedings{rozemberczki2021pytorch,
#                 author = {Benedek Rozemberczki and Paul Scherer and Yixuan He and George Panagopoulos and Alexander Riedel and Maria Astefanoaei and Oliver Kiss and Ferenc Beres and Guzman Lopez and Nicolas Collignon and Rik Sarkar},
#                 title = {{PyTorch Geometric Temporal: Spatiotemporal Signal Processing with Neural Machine Learning Models}},
#                 year = {2021},
#                 booktitle={Proceedings of the 30th ACM International Conference on Information and Knowledge Management},
#                 pages = {4564–4573},
# }


def _get_time_windows(list_fts: list, time_span: float):
    """
    Get the time windows from the list of frame_timestamps
    Each window is a subset of frame_timestamps where its time span is not greater than "time_span"

    e.g.
    input:
        list_fts:    [902, 903, 904, 905, 910, 911, 912, 913, 914, 917]
        time_span:   3
    output:
        twd_all:     [[902, 903, 904], [905], [910, 911, 912], [913, 914], [917]]
    """

    twd_all = []

    start = end = 0
    while end < len(list_fts):
        while end < len(list_fts) and list_fts[end] < list_fts[start] + time_span:
            end += 1

        twd_all.append(list_fts[start:end])
        start = end

    return twd_all


def generate_graph(
    id_data, tracking_data, time_span
    # data_file: pathlib.Path, args: Namespace, path_graphs: pathlib.Path, data_split: str
):
    """
    Generate graphs of a single video
    Time span of each graph is not greater than "time_span"
    """

    # video_id = os.path.splitext(os.path.basename(data_file))[0]
    # with open(data_file, "rb") as f:
    #     data = pickle.load(f)  # nosec

    # Get a list of frame_timestamps
    # list_fts =
    # list_fts = sorted([float(frame_timestamp) for frame_timestamp in data.keys()])
    #nflId	[displayName]	frameId	[time]	jerseyNumber	[club,	playDirection]	x	y	s	a	dis	o	dir

    # Get the time windows where the time span of each window is not greater than "time_span"
    # twd_all = _get_time_windows(list_fts, args.time_span)
    twd_all = []
    for i in range(0, id_data.shape[3], time_span):
        twd_all.append((tuple(range(i, i+time_span)), id_data[:,:,:,i:i+time_span], tracking_data[:,:,:,i:i+time_span]))

    # Iterate over every time window
    num_graph = 0
    for frames, ids, track in twd_all:

        # Skip the training graphs without any temporal edges
        if ids.shape[1] == 1:
            continue

        # Get lists of the timestamps, features, coordinates, labels, person_ids, and global_ids for a given time window

        timestamp, feature, coord, label, person_id, global_id = [], [], [], [], [], []

        for ts, id, tr in (zip(frames, ids, track)):
            timestamp.append(ts)
            for entity in np.unique(ids[0, 0]):

                player_data = tr[:, (id[0, :, 0] == entity), :]
                feature.append()
                coord.append(player_data)
                # label.append(entity["label"])
                person_id.append(entity)
                global_id.append(entity["global_id"])

        # Get a list of the edge information: these are for edge_index and edge_attr
        node_source = []
        node_target = []
        edge_attr = []

        for i in range(len(timestamp)):
            for j in range(len(timestamp)):
                # Time difference between the i-th and j-th nodes
                time_diff = timestamp[i] - timestamp[j]

                # If the edge connection mode is csi, nodes having the same identity are connected across the frames
                # If the edge connection mode is cdi, temporally-distant nodes with different identities are also connected
                if args.ec_mode == "csi":
                    id_condition = person_id[i] == person_id[j]
                elif args.ec_mode == "cdi":
                    id_condition = True

                # The edge ij connects the i-th node and j-th node
                # Positive edge_attr indicates that the edge ij is backward (negative: forward)
                if time_diff == 0 or (abs(time_diff) <= args.tau and id_condition):
                    node_source.append(i)
                    node_target.append(j)
                    edge_attr.append(np.sign(time_diff))

        # x: features
        # c: coordinates of person_box
        # g: global_ids
        # edge_index: information on how the graph nodes are connected
        # edge_attr: information about whether the edge is spatial (0) or temporal (positive: backward, negative: forward)
        # y: labels
        graphs = Data(
            x=torch.tensor(np.array(feature, dtype=np.float32), dtype=torch.float32),
            c=torch.tensor(np.array(coord, dtype=np.float32), dtype=torch.float32),
            g=torch.tensor(global_id, dtype=torch.long),
            edge_index=torch.tensor(
                np.array([node_source, node_target], dtype=np.int64), dtype=torch.long
            ),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
            y=torch.tensor(np.array(label, dtype=np.float32), dtype=torch.float32),
        )

        num_graph += 1
        torch.save(graphs, os.path.join(path_graphs, f"{video_id}_{num_graph:04d}.pt"))

    return num_graph

def generate_graphs(dl:DataLoader, graph_dir:pathlib.Path, time_span:int = 5):
    start = time.perf_counter()

    for ids, tracking in dl:
        generate_graph(ids, tracking, time_span)

        exit()
        # print(len(batch))

        # for ids, tracking in batch:
        #     print(ids[0])
        #     exit()
        pass
    # print(tracking)
    print(f"Iteration time: {time.perf_counter()-start:.2f}s")

if __name__ == "__main__":
    from dataset import NFLDataModule
    data_dir = pathlib.Path(__file__).parents[1] / "data"
    print(data_dir)
    graph_dir = data_dir/"graphs"
    dmod = NFLDataModule(data_dir, include_str_types=True)
    dmod.setup('val')
    dl = dmod.val_dataloader()
    if not graph_dir.exists():
        graph_dir.mkdir()
    generate_graphs(dl, graph_dir)
    # """
    # Generate spatial-temporal graphs from the extracted features
    # """

    # parser = argparse.ArgumentParser()
    # # Default paths for the training process
    # parser.add_argument(
    #     "--root_data", type=str, help="Root directory to the data", default="./data"
    # )
    # parser.add_argument(
    #     "--features", type=str, help="Name of the features", required=True
    # )

    # # Two options for the edge connection mode:
    # # csi: Connect the nodes only with the same identities across the frames
    # # cdi: Connect different identities across the frames
    # parser.add_argument(
    #     "--ec_mode", type=str, help="Edge connection mode (csi | cdi)", required=True
    # )
    # parser.add_argument(
    #     "--time_span",
    #     type=float,
    #     help="Maximum time span for each graph in seconds",
    #     required=True,
    # )
    # parser.add_argument(
    #     "--tau",
    #     type=float,
    #     help="Maximum time difference between neighboring nodes in seconds",
    #     required=True,
    # )

    # args = parser.parse_args()

    # # Iterate over train/val splits
    # print("This process might take a few minutes")
    # for sp in ["train", "val"]:
    #     path_graphs = os.path.join(
    #         args.root_data,
    #         f"graphs/{args.features}_{args.ec_mode}_{args.time_span}_{args.tau}/{sp}",
    #     )
    #     os.makedirs(path_graphs, exist_ok=True)

    #     list_data_files = sorted(
    #         glob.glob(
    #             os.path.join(args.root_data, f"features/{args.features}/{sp}/*.pkl")
    #         )
    #     )

    #     with Pool(processes=20) as pool:
    #         num_graph = pool.map(
    #             partial(generate_graph, args=args, path_graphs=path_graphs, sp=sp),
    #             list_data_files,
    #         )

    #     print(
    #         f"Graph generation for {sp} is finished (number of graphs: {sum(num_graph)})"
    #     )
