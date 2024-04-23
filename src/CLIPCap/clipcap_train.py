import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.nn.modules import Module
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer as Optimizer
from torch.utils.data import Dataset
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from clipcap_model import ClipCaptionModel, ClipCaptionPrefix, MappingType
from datasets import load_dataset
from PIL import Image
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode
from transformers import (
    AutoModel,
    GPT2Tokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    ViTImageProcessor,
    VisionTextDualEncoderModel,

)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    pretrained_clip_name_or_path: Optional[str] = field(
        default="/media/jj_data/models/CLIP/text_freeze1",
        metadata={
            "help": "Path to pretrained ViTForImageClassification or model identifier from huggingface.co/models"
        },
    )
    pretrained_gpt2_name_or_path: Optional[str] = field(
        default="gpt2",
        metadata={
            "help": "Path to pretrained ViTForImageClassification or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    image_processor_name_or_path: str = field(
        default=None, metadata={"help": "Name or path of preprocessor config."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    max_seq_len: Optional[int] = field(
        default=128, metadata={"help": "maximum length of a sequence"}
    )
    prefix_length: Optional[int] = field(
        default=10, metadata={"help": "maximum length of prefix sequence"}
    )
    prefix_length_clip: Optional[int] = field(
        default=10, metadata={"help": "maximum length of prefix sequence"}
    )
    prefix_dim: Optional[int] = field(
        default=512, metadata={"help": "maximum length of a sequence"}
    )
    num_layers: Optional[int] = field(
        default=8, metadata={"help": "MLP layers in projection from clip to GPT2"}
    )
    mapping_type: Optional[str] = field(
        default="mlp", metadata={"help": "What kind of layers to use for projection"}
    )
    only_prefix: Optional[bool] = field(default=False)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="/media/jj_data/data/CLIP",
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default="main",
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    data_dir: Optional[str] = field(
        default=None, metadata={"help": "The data directory containing input files."}
    )
    image_column: Optional[str] = field(
        default="pixel_values",
        metadata={
            "help": "The name of the column in the datasets containing the full image file paths."
        },
    )
    caption_column: Optional[str] = field(
        default="input_ids",
        metadata={
            "help": "The name of the column in the datasets containing the image captions."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a jsonlines file)."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file (a jsonlines file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input testing data file (a jsonlines file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocess_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`validation_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension == "json", "`validation_file` should be a json file."


def save_config(args: argparse.Namespace):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.out_dir, f"{args.prefix}.json")
    with open(out_path, "w") as outfile:
        json.dump(config, outfile)



def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments) # type:ignore
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    dataset = load_dataset(data_args.dataset_name, name=data_args.dataset_config_name)

    gpt2tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # gpt2tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    # Jack Dataset
    # ----------------------------------------------------------------------------------------------------------------------------------#
    image_column = data_args.image_column
    description_column = data_args.caption_column

    def pad_tokens(tokens):
        padding = model_args.max_seq_len - len(tokens)
        if padding > 0:
            tokens = torch.cat(
                (torch.tensor(tokens), torch.zeros(padding, dtype=torch.int64) - 1)
            )
        elif padding < 0:
            tokens = tokens[: model_args.max_seq_len]
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat(
            (torch.ones(model_args.prefix_length), mask), dim=0
        )  # adding prefix mask
        return tokens, mask

    def tokenize_captions(examples):
        captions = list(examples[description_column])
        text_inputs = gpt2tokenizer(
            captions,
            max_length=model_args.max_seq_len,
            truncation=True,
        )
        input_ids, attention_mask = list(
            zip(*[pad_tokens(ids) for ids in text_inputs.input_ids])
        )

        examples["input_ids"] = list(input_ids)
        examples["attention_mask"] = list(attention_mask)
        return examples

    class Transform(torch.nn.Module):
        def __init__(self, image_size, mean, std):
            super().__init__()
            self.transforms = torch.nn.Sequential(
                Resize([image_size], interpolation=InterpolationMode.BICUBIC),
                CenterCrop(image_size),
                ConvertImageDtype(torch.float),
                Normalize(mean, std),
            )

        def forward(self, x) -> torch.Tensor:
            """`x` should be an instance of `PIL.Image.Image`"""
            with torch.no_grad():
                x = self.transforms(x)
            return x

    image_processor = ViTImageProcessor.from_pretrained(
        model_args.image_processor_name_or_path
    )

    image_transformations = Transform(
        224,
        image_processor.image_mean, # type:ignore
        image_processor.image_std, # type:ignore
    )

    def transform_images(examples):
        images = [
            read_image(image_file, mode=ImageReadMode.RGB)
            for image_file in examples[image_column]
        ]
        images = [image_transformations(image) for image in images]

        examples["pixel_values"] = images
        return examples

    def filter_corrupt_images(examples):
        """remove problematic images"""
        valid_images = []
        for image_file in examples[image_column]:
            try:
                Image.open(image_file)
                valid_images.append(True)
            except Exception:
                valid_images.append(False)
        return valid_images
    # column_names = dataset.column_names
    if training_args.do_train:
        train_dataset = dataset["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        train_dataset = train_dataset.filter(
            filter_corrupt_images,
            batched=True,
            num_proc=data_args.preprocess_num_workers,
        )
        train_dataset = train_dataset.map(
            function=tokenize_captions,
            batched=True,
            # remove_columns=[col for col in column_names if col != image_column],
            num_proc=data_args.preprocess_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )
        # train_dataset = train_dataset.map(
        #     function=transform_images,
        #     load_from_cache_file=not data_args.overwrite_cache,
        #     # remove_columns=[image_column],
        #     desc="Running image transforms on train datast"
        # )
        # Transform images on the fly as doing it on the whole dataset takes too much time.
        train_dataset.set_transform(transform_images)

    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a train validation")
        eval_dataset = dataset["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        eval_dataset = eval_dataset.filter(
            filter_corrupt_images,
            batched=True,
            num_proc=data_args.preprocess_num_workers,
        )
        eval_dataset = eval_dataset.map(
            function=tokenize_captions,
            batched=True,
            num_proc=data_args.preprocess_num_workers,
            # remove_columns=[col for col in column_names if col != image_column],
            # load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )
        # eval_dataset = eval_dataset.map(
        #     function=transform_images,
        #     load_from_cache_file=not data_args.overwrite_cache,
        #     # remove_columns=[image_column],
        #     desc="Running image transforms on train datast"
        # )
        # Transform images on the fly as doing it on the whole dataset takes too much time.
        eval_dataset.set_transform(transform_images)

    if training_args.do_predict:
        if "test" not in dataset:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = dataset["test"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(test_dataset), data_args.max_eval_samples)
            test_dataset = test_dataset.select(range(max_eval_samples))

        test_dataset = test_dataset.filter(
            filter_corrupt_images,
            batched=True,
            num_proc=data_args.preprocess_num_workers,
        )
        test_dataset = test_dataset.map(
            function=tokenize_captions,
            batched=True,
            num_proc=data_args.preprocess_num_workers,
            # remove_columns=[col for col in column_names if col != image_column],
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on test dataset",
        )
        # test_dataset = test_dataset.map(
        #     function=transform_images,
        #     load_from_cache_file=not data_args.overwrite_cache,
        #     remove_columns=[image_column],
        #     desc="Running image transforms on train datast"
        # )
        # Transform images on the fly as doing it on the whole dataset takes too much time.
        test_dataset.set_transform(transform_images)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])

        # prefix = torch.stack([example["prefix"] for example in examples])
        input_ids = torch.tensor(
            [example["input_ids"] for example in examples], dtype=torch.long
        )
        attention_mask = torch.tensor(
            [example["attention_mask"] for example in examples], dtype=torch.long
        )
        # return input_ids, attention_mask, prefix
        return {
            "tokens": input_ids,
            "pixel_values": pixel_values,
            "mask": attention_mask,
            "return_loss":True,
        }

    # ----------------------------------------------------------------------------------------------
    model_args.mapping_type = {
        "mlp": MappingType.MLP,
        "transformer": MappingType.Transformer,
    }[model_args.mapping_type]

    clip_model:VisionTextDualEncoderModel = AutoModel.from_pretrained(model_args.pretrained_clip_name_or_path)
    clip_model = clip_model.eval()
    clip_model = torch.compile(clip_model, mode="max-autotune")
    clip_model.to(device="cuda")
    if model_args.only_prefix:
        model = ClipCaptionPrefix(
            clip_model,
            model_args.prefix_length,
            clip_length=model_args.prefix_length_clip,
            prefix_size=model_args.prefix_dim,
            num_layers=model_args.num_layers,
            mapping_type=model_args.mapping_type,
        )
        print("Train only prefix")
    else:

        model = ClipCaptionModel(
            clip_model,
            model_args.prefix_length,
            clip_length=model_args.prefix_length_clip,
            prefix_size=model_args.prefix_dim,
            num_layers=model_args.num_layers,
            mapping_type=model_args.mapping_type,
        )
        print("Train both prefix and GPT")
        sys.stdout.flush()

    trainer = Trainer(
        model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=collate_fn,
    )
    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()
        gpt2tokenizer.save_pretrained(model_args.output_dir)
        image_processor.save_pretrained(model_args.output_dir)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        clip_model.save_pretrained(model_args.output_dir)
    # 10. Evaluation and Testing
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()
