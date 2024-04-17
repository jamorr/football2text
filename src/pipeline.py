import pathlib
from collections.abc import Iterable
from typing import Any

import torch
import torchvision
from dataset import NFLDataset
from torch.nn import Module
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from towhee import DataCollection, ops, pipe
from towhee.data.dataset.dataset import TorchDataSet, TowheeDataSet
from towhee.models.clip4clip import CLIP4Clip, create_model
from towhee.trainer.modelcard import MODEL_CARD_NAME, ModelCard
from towhee.trainer.trainer import Trainer
from towhee.trainer.training_config import TrainingConfig, dump_default_yaml
from towhee.trainer.utils.trainer_utils import STATE_CHECKPOINT_NAME, MODEL_NAME, set_seed, reduce_value, \
    is_main_process, send_to_device, unwrap_model, _construct_loss_from_config, _construct_optimizer_from_config, \
    _construct_scheduler_from_config
from towhee.models import clip4clip
from towhee.models.clip4clip.until_module import PreTrainedModel

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
        # print(inputs)
        loss = model.forward(*inputs)
        # print(loss)
        return loss
    
    
    @torch.no_grad()
    def compute_metric(self, model: nn.Module, inputs: Any) -> float:
        """
        Compute the step metric.
        It is recommended to subclass `Trainer` and override this method when deal with custom metric in custom task.
        When it is overridden, another method `compute_loss()` often needs to be overridden.

        Args:
            model (`nn.Module`):
                Pytorch model.
            inputs (`Any`):
                Input tensor collection.

        Returns:
            (`float`)
                Epoch average metric.
        """
        return 1
        model.eval()
        epoch_metric = None
        labels = inputs[1]
        outputs = model(*inputs)
        if self.metric is not None:
            self.metric.update(send_to_device(outputs, self.configs.device),
                               send_to_device(labels, self.configs.device))
            epoch_metric = self.metric.compute().item()
        return epoch_metric



if __name__ == "__main__":
    # model = CLIP4Clip(
    #     **dict(
    #         # vision
    #         embed_dim=512,
    #         image_resolution=224,
    #         vision_layers=12,
    #         vision_width=768,
    #         vision_patch_size=32,
    #         # text
    #         context_length=77,
    #         vocab_size=49408,
    #         transformer_width=512,
    #         transformer_heads=8,
    #         transformer_layers=12,
    #     )
    # )
    # model = clip4clip.create_model(model_name="clip_vit_b32", context_length=77, pretrained=False, device='cuda')

    model = ops.video_text_embedding.clip4clip(model_name='clip_vit_b32', modality='video', device='cpu').get_op()

    print(dir(model))
    exit()
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
    # print(ops.video_text_embedding.clip4clip(model_name='clip_vit_b32', modality='text')())
    # train_data[0][0]
    # for i in range(len(train_data))[:1]:
        # print(train_data[i][0])
