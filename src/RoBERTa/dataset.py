import pathlib
from typing import Any

import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class NFLTextDataset(Dataset):
    def __init__(self, data_path:pathlib.Path, which:str = "train") -> None:
        super().__init__()
        assert data_path.exists()
        self.data_dir = data_path
        self.target = pd.read_parquet(data_path/which / "target.parquet", dtype_backend='numpy_nullable', columns=['gameId', 'playId','playDescription']) #.convert_dtypes(dtype_backend='numpy_nullable')
        self.tokenizer = AutoTokenizer.from_pretrained("jkruk/distilroberta-base-ft-nfl")
        self.which = which
        print(self.target)
    def __len__(self):
        return len(self.target)

    def build_csvs(self):
        df = pd.DataFrame(self.target['playDescription']).rename({'playDescription':'text'},axis=1)
        df.to_csv(pathlib.Path(f"data/{self.which}.csv"))

    def __getitem__(self, index) -> Any:
        return self.target.iloc[index]['playDescription']




if __name__ == "__main__":
    data_dir = pathlib.Path(__file__).parents[1].parent / "data"
    print(data_dir)
    ds = NFLTextDataset(data_dir,"val")
    # for item in ['train','test','val']:
    #     print(f"Building {item}")
    #     ds = NFLTextDataset(data_dir,which=item)
    #     print(ds[10])
    #     # ds.build_csvs()
