import shutil
from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule, seed_everything
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
    def __init__(
        self,
        data_path:pathlib.Path=pathlib.Path('../data'),
        splits:tuple[float, float, float]=(0.75, 0.05, 0.2),
        **loader_args

        ):
        super().__init__()
        self.data_path:pathlib.Path = data_path
        self.splits = splits
        self.loader_args = loader_args


    def prepare_data(self) -> None:
        # assumes GLOBAL_RANK=0
        train, val, test = self.splits
        assert sum(self.splits) == 1
        dataset = NFLDataset(self.data_path)
        missing_indices = []
        for i in range(len(dataset)):
            try:
                dataset[i]
            except FileNotFoundError:
                missing_indices.append(i)
        missing_dropped_df:pd.DataFrame = dataset.target.drop(missing_indices, axis='rows')
        msk = np.random.rand(len(missing_dropped_df)) < (train + val)
        train_val_df = missing_dropped_df.iloc[msk]
        val_msk = np.random.rand(len(train_val_df)) < (val / (train + val))

        train_df:pd.DataFrame = train_val_df.iloc[~val_msk]
        val_df:pd.DataFrame =  train_val_df.iloc[val_msk]
        test_df:pd.DataFrame = missing_dropped_df.iloc[~msk]
        del msk, train_val_df, val_msk

        self._move_files_to_split_directory("train", train_df)
        self._move_files_to_split_directory("test", test_df)
        self._move_files_to_split_directory("val", val_df)

    def _move_files_to_split_directory(self, split_name, split_df):
        split_dir:pathlib.Path = self.data_path/split_name/"tracking_weeks"
        split_df.to_parquet(self.data_path/f"{split_name}_target.parquet")

        for idx, data in split_df[['gameId', 'playId']].iterrows():
            game_dir_name = f"gameId={data['gameId']}"
            play_dir_name = f"playId={data['playId']}"
            src_path = data_dir/"tracking_weeks"/game_dir_name/play_dir_name
            assert (src_path).exists()
            dest_dir = split_dir/game_dir_name
            if not dest_dir.exists():
                dest_dir.mkdir(parents=True)
            shutil.move(src_path, dest_dir/play_dir_name)


    def setup(self, which):
        setattr(self, f"{which}_dataset",  NFLDataset(data_path=self.data_path/which))

    def train_dataloader(self) -> Any:
        return DataLoader(
            self.train_dataset
            **self.loader_args)

    def val_dataloader(self) -> Any:
        return DataLoader(
            self.val_dataset
            **self.loader_args)

    def test_dataloader(self) -> Any:
        return DataLoader(
            self.test_dataset
            **self.loader_args)


if __name__ == '__main__':
    import time
    data_dir = pathlib.Path('../data')
    seed_everything(37, True)
    print(data_dir.absolute())
    NFLDataModule(data_dir).prepare_data()
    # data = NFLDataset(data_dir)
    # no_missing = 0
    # start = time.perf_counter()
    # for i in range(len(data)):
    #     try:
    #         data[i]
    #     except FileNotFoundError:
    #         no_missing += 1
    #         # print(f"No file for {i}")
    # print(f" All files read in {time.perf_counter()-start:.2f}s")
    # if no_missing == 0:
    #     print("Passed checks")
    # else:
    #     print(f"Missing {no_missing} files out of {len(data)}...")