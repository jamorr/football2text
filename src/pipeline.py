import pathlib
from towhee import pipe
import torchvision
from dataset import NFLDataset

#Takes video file & metadata

# preprocess = (
#     pipe.input('vid','text')
#         .map('vid', 'y', lambda x: x + 1)
#         .map('text', 'y', lambda x: x + 1)
#         .output('y')
# )


# training_configs = TrainingConfig(
#      xxx='some_value_xxx',
#      yyy='some_value_yyy'
# )

if __name__ == "__main__":
    data_dir = pathlib.Path(__file__).parents[1] / "data" / "val"
    dset = NFLDataset(data_dir, True)
    for i in range(len(dset)):
        dset[i]
