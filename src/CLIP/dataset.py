import pathlib
import shutil
from typing import Any
from PIL import Image as PILImage

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import LightningDataModule, seed_everything
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_video, read_image
from towhee import ops
from towhee.models.clip.clip_utils import tokenize
from torchvision import transforms
from torchvision.io import write_jpeg, write_png
from transformers import AutoImageProcessor, ViTMAEModel
from tqdm import tqdm
from transformers import AutoTokenizer

class NFLTextDataset(Dataset):
    def __init__(self, data_path:pathlib.Path, which:str = "train") -> None:
        super().__init__()
        assert data_path.exists()

    def __len__(self):
        return len(self.target)

    def build_csvs(self):
        df = pd.DataFrame(self.target['playDescription']).rename({'playDescription':'text'},axis=1)
        df.to_csv(pathlib.Path(f"data/{self.which}.csv"))

    def __getitem__(self, index) -> Any:
        return self.target.iloc[index]['playDescription']


class NFLJPEGDataset(Dataset):
    def __init__(self, data_path:pathlib.Path, which:str = "train") -> None:
        super().__init__()
        if isinstance(data_path, str):
            data_path = pathlib.Path(data_path)
        assert data_path.exists()
        #Img
        self.data_dir:pathlib.Path = data_path
        self.img_path:pathlib.Path = data_path / "jpeg_data"
        self.img_list = list(self.img_path.glob("*.jpeg"))
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        #Text
        self.target = pd.read_parquet(data_path/ which / "target.parquet", dtype_backend='numpy_nullable', columns=['gameId', 'playId','playDescription']) #.convert_dtypes(dtype_backend='numpy_nullable')

        self.tokenizer = AutoTokenizer.from_pretrained("jkruk/distilroberta-base-ft-nfl")
        self.which = which
        #     self.tfms = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize(
        #         (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        # ])
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index) -> Any:
        img_file_path = self.img_list[index]
        file_name = img_file_path.name
        gameId, playId, frameID = file_name.split('-')[0:2]
        frameID = frameID[:-5] # Removing .jpeg

        # Getting play, and then indexing by frameID. If this doesn't work then we might need to reset_index
        self.play = self.target[(self.target["gameId"] == gameId)&(self.target["playId"] == playId)]
        text = self.play.iloc[frameID]['playDescription']
        return text,read_image(img_file_path)
        # THIS DOESNT WORK FOR SOME REASON
        # return read_image(str(self.img_list[index % len(self)].absolute()))
    

if __name__ == "__main__":
    data_dir = pathlib.Path(__file__).parents[1].parent / "data"
    # print(data_dir)
    # df = NFLJPEGDataset(data_dir)
    # print(len(df))

    x = "523.jpeg"
    print(x[:-5])