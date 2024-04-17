import pathlib
from typing import Any

import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
from tqdm import tqdm
from transformers import AutoImageProcessor


class NFLVisionTextDataset(Dataset):
    def __init__(self, data_path: pathlib.Path) -> None:
        super().__init__()
        if isinstance(data_path, str):
            data_path = pathlib.Path(data_path)
        assert data_path.exists()
        # Img
        self.data_dir: pathlib.Path = data_path
        self.img_path: pathlib.Path = data_path / "jpeg_data"
        self.img_list = list(self.img_path.glob("*.jpeg"))

        # Text
        self.target = pd.read_parquet(
            data_path / "target.parquet",
            dtype_backend="numpy_nullable",
            columns=["gameId", "playId", "playDescription"],
        )

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index) -> Any:
        img_file_path = self.img_list[index]
        file_name = img_file_path.name
        gameId, playId, *_ = file_name.split("-")[0:2]
        gameId, playId = int(gameId), int(playId)
        # Getting play, and then indexing by frameID. If this doesn't work then we might need to reset_index
        play = self.target[
            (self.target["gameId"] == gameId) & (self.target["playId"] == playId)
        ]

        return play["playDescription"], read_image(str(img_file_path.absolute()))


if __name__ == "__main__":
    data_dir = pathlib.Path("/media/jj_data/data")
    # data_dir = pathlib.Path(__file__).parents[1].parent / "data"
    print(data_dir)
    dset = NFLVisionTextDataset(data_dir/"val")
    for d in tqdm(dset):
        pass
