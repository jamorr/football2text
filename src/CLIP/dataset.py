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


class NFLJPEGDataset(Dataset):
    def __init__(self, data_path:pathlib.Path) -> None:
        super().__init__()
        if isinstance(data_path, str):
            data_path = pathlib.Path(data_path)
        assert data_path.exists()
        self.data_dir:pathlib.Path = data_path
        self.img_path:pathlib.Path = data_path / "jpeg_data"
        self.img_list = list(self.img_path.glob("*.jpeg"))
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")

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
        # THIS DOESNT WORK FOR SOME REASON
        return read_image(str(self.img_list[index % len(self)].absolute()))