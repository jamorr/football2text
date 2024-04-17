import pathlib
import shutil
from typing import Any
from PIL import Image as PILImage

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import LightningDataModule, seed_everything
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_video
from towhee import ops
from towhee.models.clip.clip_utils import tokenize
from torchvision import transforms
from transformers import AutoTokenizer

class NFLTextDataset(Dataset):
    def __init__(self, data_path:pathlib.Path, which:str = "train",batch_size:int = 40) -> None:
        super().__init__()
        assert data_path.exists()
        self.data_dir = data_path
        self.play_dir = data_path/ which / 'tracking_weeks'
        self.target = pd.read_parquet(data_path/which / "target.parquet", dtype_backend='numpy_nullable', columns=['gameId', 'playId','playDescription']) #.convert_dtypes(dtype_backend='numpy_nullable')
        self.id_cols = ['nflId', 'frameId', 'jerseyNumber', 'club', 'playDirection', 'event']
        self.tracking_cols = ['x', 'y', 's', 'a', 'dis', 'o', 'dir']
        self.tokenizer = AutoTokenizer.from_pretrained("jkruk/distilroberta-base-ft-nfl")
        self.which = which
        print(self.play_dir)
        self.tracking_weeks = pd.read_parquet(
            self.play_dir,
            dtype_backend='numpy_nullable',
            columns=['gameId','playId', 'frameId']
            ).drop_duplicates(keep='last')

    def __len__(self):
        return len(self.tracking_weeks) #// self.batch_size
    def build_csvs(self):
        df = pd.DataFrame(self.target['playDescription']).rename({'playDescription':'text'},axis=1)
        df.to_csv(pathlib.Path(f"data/{self.which}.csv"))
    def __getitem__(self, index) -> Any:
        # target_play = self.target.iloc[index]['playDescription']
        # token = self.tokenizer(target_play,return_tensors='pt')
        # print(type(token),token)
        target_play = self.target.iloc[index]['playDescription']
        # token = self.tokenizer(target_play,return_tensors='pt')

        return target_play




if __name__ == "__main__":
    data_dir = pathlib.Path(__file__).parents[1].parent / "data"
    print(data_dir)
    for item in ['train','test','val']:
        print(f"Building {item}")
        ds = NFLTextDataset(data_dir,which=item)
        ds.build_csvs()
