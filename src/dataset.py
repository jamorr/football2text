import pathlib
import shutil
from typing import Any

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import LightningDataModule, seed_everything
from torch.utils.data import DataLoader, Dataset



# TODO: #2 add padding to make stackable outputs/batchsize > 1
# TODO: #1 preprocess text into token ids
class NFLDataset(Dataset):
    def __init__(self, data_path:pathlib.Path, include_str_types:bool) -> None:
        super().__init__()
        assert data_path.exists()
        self.include_str_types = include_str_types
        self.play_dir = data_path/'tracking_weeks'

        self.target = pd.read_parquet(data_path/'target.parquet', dtype_backend='numpy_nullable') #.convert_dtypes(dtype_backend='numpy_nullable')
        self.players = pd.read_parquet(data_path/'players.parquet', dtype_backend='numpy_nullable') #.convert_dtypes(dtype_backend='numpy_nullable')
        self.id_cols = ['nflId', 'frameId', 'jerseyNumber', 'club', 'playDirection', 'event']
        self.tracking_cols = ['x', 'y', 's', 'a', 'dis', 'o', 'dir']
        # TODO: Add support for the mapping of ids to strings
        # self.teams = pd.read_parquet(data_path/'team_id_map.parquet', dtype_backend='numpy_nullable')
        # self.directions = pd.read_parquet(data_path/'direction_id_map.parquet', dtype_backend='numpy_nullable')
        # self.events = pd.read_parquet(data_path/'events_id_map.parquet', dtype_backend='numpy_nullable')

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index) -> Any:
        target_play = self.target.iloc[index]
        play_data = pd.read_parquet(
            self.play_dir/f'gameId={target_play["gameId"]}'/f'playId={target_play["playId"]}',
            dtype_backend='numpy_nullable',
            columns=['nflId', 'frameId', 'jerseyNumber', 'club', 'playDirection', 'event', 'x', 'y', 's', 'a', 'dis', 'o', 'dir']
        )
        # id columns

        play_data[self.id_cols] = play_data[self.id_cols].astype(np.int32)
        # tracking data columns
        play_data[self.tracking_cols] = play_data[self.tracking_cols].astype(np.float32)
        play_data = play_data[self.id_cols + self.tracking_cols]
        # organize into frames
        # print([group for _, group in play_data.groupby("frameId", as_index=True)][0])
        framewise_data = np.array([group.values for _, group in play_data.groupby("frameId", as_index=True)])
        int_cols = framewise_data[:, :, :len(self.id_cols)].astype(np.int32)  # Contains the first three columns
        float_cols = framewise_data[:, :, len(self.id_cols):].astype(np.float32)
        if self.include_str_types:
            players = self.players[self.players['nflId'].isin(play_data['nflId'].unique())]
            return int_cols, float_cols, players, target_play
        return int_cols, float_cols, torch.tensor((target_play["gameId"], target_play["playId"]), dtype=torch.int32)


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
        self.include_str_types = False


    def prepare_data(self) -> None:
        # assumes GLOBAL_RANK=0
        train, val, test = self.splits
        assert sum(self.splits) == 1

        events = (
            'autoevent_ballsnap',
            'autoevent_passforward',
            'autoevent_passinterrupted',
            'ball_snap',
            'first_contact',
            'fumble',
            'fumble_defense_recovered',
            'fumble_offense_recovered',
            'handoff',
            'lateral',
            'line_set',
            'man_in_motion',
            'out_of_bounds',
            'pass_arrived',
            'pass_forward',
            'pass_outcome_caught',
            'pass_outcome_touchdown',
            'pass_shovel',
            'penalty_accepted',
            'penalty_flag',
            'play_action',
            'qb_sack',
            'qb_slide',
            'run',
            'run_pass_option',
            'safety',
            'shift',
            'snap_direct',
            'tackle',
            'touchdown'
        )

        team_names = pd.read_csv(self.data_path/"games.csv")['homeTeamAbbr'].unique()
        play_direction = pd.Series(['left', 'right']).unique()

        tracking_df = pd.DataFrame()
        for file_path in data_dir.glob("tracking_week_*.csv"):
            print(f"Converting {file_path} into parquet")
            tracking_df = pd.read_csv(file_path, dtype_backend='pyarrow')
            tracking_df['playDirection'] = pd.Categorical(tracking_df['playDirection'], categories=play_direction).codes
            tracking_df['club'] = pd.Categorical(tracking_df['club'],categories=team_names).codes
            tracking_df['event'] = pd.Categorical(tracking_df['event'],categories=events).codes
            tracking_df.drop(["displayName", "time"], inplace=True, axis='columns')
            tracking_df.fillna(-1, inplace=True)
            tracking_df.to_parquet(self.data_path/"tracking_weeks", partition_cols=["gameId", "playId"])
        print("Creating map parquets from id to name")
        events_to_id_df = pd.DataFrame({'id': range(len(events)), 'category': events})
        events_to_id_df.to_parquet(data_dir/'events_id_map.parquet')

        id_to_team_df = pd.DataFrame({'id': range(len(team_names)), 'category': team_names})
        id_to_team_df.to_parquet(data_dir/'team_id_map.parquet')

        id_to_direction_df = pd.DataFrame({'id': range(len(play_direction)), 'category': play_direction})
        id_to_direction_df.to_parquet(data_dir/'direction_id_map.parquet')

        players_df = pd.read_csv(data_dir/'players.csv', dtype_backend='pyarrow')
        players_df.to_parquet(data_dir/'players.parquet')

        target_df = pd.read_csv(data_dir/'plays.csv', dtype_backend='pyarrow')
        target_df.to_parquet(data_dir/'target.parquet')
        print("Loading full dataset")
        dataset = NFLDataset(self.data_path, False)
        print("Removing examples with missing tracking data")
        missing_indices = []
        for i in range(len(dataset)):
            try:
                dataset[i]
            except FileNotFoundError:
                missing_indices.append(i)
        missing_dropped_df:pd.DataFrame = dataset.target.drop(missing_indices, axis='rows')
        print("Splitting dataset into Train/Val/Test")
        msk = np.random.rand(len(missing_dropped_df)) < (train + val)
        train_val_df = missing_dropped_df.iloc[msk]
        val_msk = np.random.rand(len(train_val_df)) < (val / (train + val))

        train_df:pd.DataFrame = train_val_df.iloc[~val_msk]
        val_df:pd.DataFrame =  train_val_df.iloc[val_msk]
        test_df:pd.DataFrame = missing_dropped_df.iloc[~msk]
        del msk, train_val_df, val_msk
        print("Moving data to train folder")
        self._move_files_to_split_directory("train", train_df)
        print("Moving data to test folder")
        self._move_files_to_split_directory("test", test_df)
        print("Moving data to val folder")
        self._move_files_to_split_directory("val", val_df)
        print("Deleting all leftover folders")
        shutil.rmtree(self.data_path/"tracking_weeks")


    def _move_files_to_split_directory(self, split_name, split_df):
        (self.data_path/split_name).mkdir()

        split_dir:pathlib.Path = self.data_path/split_name/"tracking_weeks"
        split_dir.mkdir()
        for idx, data in split_df[['gameId', 'playId']].iterrows():
            game_dir_name = f"gameId={data['gameId']}"
            play_dir_name = f"playId={data['playId']}"
            src_path = data_dir/"tracking_weeks"/game_dir_name/play_dir_name
            assert (src_path).exists()
            dest_dir = split_dir/game_dir_name
            if not dest_dir.exists():
                dest_dir.mkdir(parents=True)
            shutil.move(src_path, dest_dir/play_dir_name)

        split_df.to_parquet(self.data_path/split_name/"target.parquet")
        shutil.copy(self.data_path/"players.parquet", self.data_path/split_name/"players.parquet")
        shutil.copy(self.data_path/"team_id_map.parquet", self.data_path/split_name/"team_id_map.parquet")
        shutil.copy(self.data_path/"direction_id_map.parquet", self.data_path/split_name/"direction_id_map.parquet")
        shutil.copy(self.data_path/"events_id_map.parquet", self.data_path/split_name/"events_id_map.parquet")


    def setup(self, which):
        setattr(self, f"{which}_dataset",  NFLDataset(data_path=self.data_path/which, include_str_types=self.include_str_types))

    def train_dataloader(self) -> Any:
        return DataLoader(
            self.train_dataset,
            **self.loader_args)

    def val_dataloader(self) -> Any:
        return DataLoader(
            self.val_dataset,
            **self.loader_args)

    def test_dataloader(self) -> Any:
        return DataLoader(
            self.test_dataset,
            **self.loader_args)


if __name__ == '__main__':
    import time
    data_dir = pathlib.Path(__file__).parents[1] / "data"
    seed_everything(37, True)
    print(data_dir.absolute())
    dmod = NFLDataModule(data_dir)
    if not (data_dir/"train").exists():
        dmod.prepare_data()
    dmod.setup("train")
    data = dmod.train_dataloader()
    no_missing = 0
    print("Checking that all data in train set can be loaded")
    start = time.perf_counter()
    for d in data:
        pass
    print(f" All files read in {time.perf_counter()-start:.2f}s")
    print("Passed checks")