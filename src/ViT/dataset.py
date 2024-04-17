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
from torchvision.io import write_jpeg, write_png
from transformers import AutoImageProcessor, ViTMAEModel
from tqdm import tqdm

class NFLImageDataset(Dataset):
    def __init__(self, data_path:pathlib.Path, img_size=224, write_mode:bool=False) -> None:
        super().__init__()
        assert data_path.exists()
        self.data_dir = data_path
        self.play_dir = data_path/'tracking_weeks'
        self.img_size = img_size
        self.write_mode = write_mode
        if write_mode and not (self.data_dir/"jpeg_data").exists():
            (self.data_dir/"jpeg_data").mkdir()

        self.target = pd.read_parquet(data_path/'target.parquet', dtype_backend='numpy_nullable', columns=['gameId', 'playId']) #.convert_dtypes(dtype_backend='numpy_nullable')
        self.id_cols = ['nflId', 'frameId', 'jerseyNumber', 'club', 'playDirection', 'event']
        self.tracking_cols = ['x', 'y', 's', 'a', 'dis', 'o', 'dir']
        # TODO: Add support for the mapping of ids to strings
        # self.teams = pd.read_parquet(data_path/'team_id_map.parquet', dtype_backend='numpy_nullable')
        # self.directions = pd.read_parquet(data_path/'direction_id_map.parquet', dtype_backend='numpy_nullable')
        # self.events = pd.read_parquet(data_path/'events_id_map.parquet', dtype_backend='numpy_nullable')
        self.tfms = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        self.tracking_weeks = pd.read_parquet(
            self.play_dir,
            dtype_backend='numpy_nullable',
            columns=['gameId','playId', 'frameId']
            ).drop_duplicates(keep='last')
        self.loaded_frames = None
        self.vid_idx = 0
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")

    def __len__(self):
        if self.write_mode:
            return len(self.target)
        return len(self.tracking_weeks) #// self.batch_size


    def _load_frames_batch(self):
        tup = self.target.iloc[self.vid_idx]
        self.vid_idx += 1
        file_path = self.data_dir/"mp4_data"/f'{tup.gameId}-{tup.playId}.mp4'
        frames, *_ = read_video(str(file_path.absolute()))
        # frames = frames.permute(0, 3, 1, 2)
        if not self.write_mode:
            self.loaded_frames = [frame for frame in frames]
        else:
            self.loaded_frames = [
                (frame, str((self.data_dir/"jpeg_data"/f'{tup.gameId}'/f'{tup.playId}'/f'{frame_id}.jpeg').absolute()))
                for frame_id, frame in enumerate(frames)]
            # for frame_id, frame in enumerate(frames):
            #     sloc = str()
            #     print(sloc)
            #     print(frame.dtype)
            #     exit()
            #     write_jpeg(frame, sloc, quality=90)
        # default batchsize is 30
        # idx = index*self.batch_size
        # plays = self.tracking_weeks.iloc[idx:idx+40]
        # fstack = torch.zeros((self.batch_size, self.img_size, self.img_size, 3))
        # num_frames = 0
        # for tup in plays.itertuples(False):
        #     file_path = self.data_dir/"mp4_data"/f'{tup.gameId}-{tup.playId}.mp4'
        #     frames, *_ = read_video(str(file_path.absolute()),end_pts=tup.frameId)
        #     num_frames += len(frames)
        #     fstack[0:tup.frameId] = frames
        # assert num_frames == self.batch_size
        # return fstack

    def __getitem__(self, index) -> Any:
        if not self.loaded_frames:
            self._load_frames_batch()
        return self.loaded_frames.pop()
        # out =  self.image_processor(self.loaded_frames.pop().squeeze())
        # print(len(out), type(out), end="\r")




if __name__ == "__main__":
    from time import perf_counter
    from PIL import Image
    for which in ("train", "test", "val"):
    # which = "val"
        data_dir = pathlib.Path(__file__).parents[2]/"data"/which
        dset = NFLImageDataset(data_dir, write_mode=True)
        start = perf_counter()
        for i, (frame, sloc) in tqdm(enumerate(dset)):
            im = Image.fromarray(frame.numpy())
            im.save(sloc)
            # sloc = ''.join([ch for ch in sloc])
            # write_png(frame, sloc)
            # print(i, end="\r")
        print(f"finished writing {which} dataset in {perf_counter()-start:.2f}s")