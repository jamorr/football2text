import pathlib
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    RobertaForMaskedLM,
    Trainer,
    TrainingArguments,
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    ViTImageProcessor,
    ViTMAEForPreTraining,
)
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from PIL import Image
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode

import transformers
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

import pandas as pd


dset = load_dataset(
    "/media/jj_data/data/CLIP",
    "default",
)
train_dset = dset["train"]
root_dir = pathlib.Path("/media/jj_data/")
models_dir = root_dir / "models"
vit_ver = "2/checkpoint-7896"
vit_pretrained = ViTMAEForPreTraining.from_pretrained(
    models_dir / "ViT" / vit_ver,
)

vit_model = vit_pretrained.vit  # type: ignore
vit_encoder_dir = models_dir / "ViT_encoder" / vit_ver
if not (vit_encoder_dir).exists():
    vit_encoder_dir.mkdir(parents=True)
    vit_model.save_pretrained(vit_encoder_dir)
image_processor = AutoImageProcessor.from_pretrained(models_dir / "ViT" / vit_ver)
tokenizer = AutoTokenizer.from_pretrained(models_dir / "roberta")
model = VisionTextDualEncoderModel.from_vision_text_pretrained(
    # models_dir / "ViT" / vit_ver, # type: ignore
    vit_encoder_dir,
    models_dir / "roberta",  # type: ignore
)
def transform_images(examples):
    images = [read_image(image_file, mode=ImageReadMode.RGB) for image_file in examples[image_column]]
    examples["pixel_values"] = [image_transformations(image) for image in images]
    return examples
train_dataset.set_transform(transform_images)
print(vit_model(read_image(train_dset[0]["pixel_values"]).unsqueeze(0)))