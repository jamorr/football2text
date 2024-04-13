from typing import Any
import torch
from torch.utils.data import Dataset
from lightning.pytorch import LightningDataModule
import pathlib
import pandas as pd

class NFLDataset(Dataset):
    def __init__(self, data_path:pathlib.Path) -> None:
        super().__init__()
        assert data_path.exists()
        self.target = pd.read_parquet(data_path/'target.parquet')

        self.players = pd.read_parquet(data_path/'players.parquet')
        self.play_dir = data_path/'tracking_weeks'

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index) -> Any:
        target_play = self.target.iloc[index]
        play_data = pd.read_parquet(self.play_dir/f'gameId={target_play["gameId"]}'/f'playId={target_play['playId']}')
        players = self.players[self.players['nflId'].isin(play_data['nflId'].unique())]
        return play_data, players, target_play


class NFLDataModule(LightningDataModule):
    def __init__(self, data_path:pathlib.Path=pathlib.Path('../data'), splits:tuple[float]=(0.75, 0.05, 0.2)):
        super().__init__()
        self.data_path = data_path
        self.splits = splits


    def prepare_data(self) -> None:
        # assumes GLOBAL_RANK=0
        dataset = NFLDataset(self.data_path)




    def setup(self, which):
        if which in ("train", "fit"):
            pass

if __name__ == '__main__':
    import time
    data_dir = pathlib.Path('../data')
    print(data_dir.absolute())
    data = NFLDataset(data_dir)
    no_missing = 0
    start = time.perf_counter()
    for i in range(len(data)):
        try:
            data[i]
        except FileNotFoundError:
            no_missing += 1
            # print(f"No file for {i}")
    print(f" All files read in {time.perf_counter()-start:.2f}s")
    if no_missing == 0:
        print("Passed checks")
    else:
        print(f"Missing {no_missing} files out of {len(data)}...")