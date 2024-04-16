import pathlib
from collections.abc import Iterable
from typing import Any

import torchvision
from dataset import NFLDataset
from torch import Module, nn, optim
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from towhee import DataCollection, ops, pipe
from towhee.data.dataset.dataset import TorchDataSet, TowheeDataSet
from towhee.models.clip4clip import CLIP4Clip, create_model
from towhee.trainer.modelcard import MODEL_CARD_NAME, ModelCard
from towhee.trainer.trainer import Trainer
from towhee.trainer.training_config import TrainingConfig, dump_default_yaml

class C4CTrainer(Trainer):
    def __init__(
        self,
        model: Module = None,
        training_config: TrainingConfig = None,
        train_dataset: Dataset | TowheeDataSet = None,
        eval_dataset: Dataset | TowheeDataSet = None,
        train_dataloader: DataLoader | Iterable = None,
        eval_dataloader: DataLoader | Iterable = None,
        model_card: ModelCard = None,
    ):
        super().__init__(
            model,
            training_config,
            train_dataset,
            eval_dataset,
            train_dataloader,
            eval_dataloader,
            model_card,
        )
    def compute_loss(self, model: nn.Module, inputs: Any):
        self.set_train_mode(model)
        loss = model(*inputs)
        return loss



if __name__ == "__main__":
    model = CLIP4Clip(
        **dict(
            # vision
            embed_dim=512,
            image_resolution=224,
            vision_layers=12,
            vision_width=768,
            vision_patch_size=32,
            # text
            context_length=77,
            vocab_size=49408,
            transformer_width=512,
            transformer_heads=8,
            transformer_layers=12,
        )
    )
    which = "train"
    data_dir = pathlib.Path(__file__).parents[1] / "data"
    train_data = NFLDataset(data_dir / which, True)
    which = "val"
    eval_data = NFLDataset(data_dir / which, True)
    training_config = TrainingConfig(
        output_dir="../models",
        epoch_num=2,
        batch_size=1,
        print_steps=1,
    )
    trainer = C4CTrainer(
        model, training_config, train_dataset=train_data, eval_dataset=eval_data
    )
    # data_dir = pathlib.Path(__file__).parents[1] / "data" / "val"
    # dset = NFLDataset(data_dir, True)

    trainer.train()
    # for i in range(len(dset))[:10]:
    #     model(dset[i][0])
