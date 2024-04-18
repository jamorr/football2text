import os
import pathlib
from dataset import NFLVisionTextDataset
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
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

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
import torch
import datasets

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# We use torchvision for faster image pre-processing. The transforms are implemented as nn.Module,
# so we jit it to be faster.
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


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
    attention_mask = torch.tensor([example["attention_mask"] for example in examples], dtype=torch.long)
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "return_loss": True,
    }



def main():
    parser = HfArgumentParser((TrainingArguments,))
    training_args = parser.parse_args_into_dataclasses()[0]

    root_dir = pathlib.Path("/media/jj_data/")
    models_dir = root_dir / "models"
    vit_ver = "1"

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
    preprocessor = VisionTextDualEncoderProcessor(image_processor, tokenizer)
    clip_model = VisionTextDualEncoderModel.from_vision_text_pretrained(
        vit_encoder_dir,
        models_dir / "roberta",  # type: ignore
    )
    # 3. Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

        # 7. Preprocessing the datasets.
    # Initialize torchvision transforms and jit it for faster processing.
    image_transformations = Transform(
        224, image_processor.image_mean, image_processor.image_std
    )
    image_transformations = torch.jit.script(image_transformations)

    dataset = load_dataset(
            "/media/jj_data/data/nfl_image_text_dataset.py",
            'main',
            trust_remote_code=True
        )
    column_names = dataset["train"].column_names
    image_column = "image_frame"
    caption_column = "text"
    train_dataset = dataset["train"]
    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples):
        print(examples)
        captions = list(examples[caption_column])
        print(type(captions[0]))
        text_inputs = tokenizer(captions, max_length=128, padding="max_length", truncation=True)
        examples["input_ids"] = text_inputs.input_ids
        examples["attention_mask"] = text_inputs.attention_mask
        return examples

    def transform_images(examples):
        images = [read_image(image_file, mode=ImageReadMode.RGB) for image_file in examples[image_column]]
        examples["pixel_values"] = [image_transformations(image) for image in images]
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

    train_dataset = train_dataset.filter(
        filter_corrupt_images, batched=True, num_proc=4
    )

    train_dataset = train_dataset.map(
        function=tokenize_captions,
        batched=True,
        remove_columns=[col for col in column_names if col != image_column],
        num_proc=4,
        desc="Running tokenizer on train dataset",
    )
    train_dataset.set_transform(transform_images)
    eval_dataset = dataset["validation"]

    eval_dataset = eval_dataset.filter(
        filter_corrupt_images, batched=True, num_proc=4
    )
    eval_dataset = eval_dataset.map(
        function=tokenize_captions,
        batched=True,
        num_proc=4,
        remove_columns=[col for col in column_names if col != image_column],
        desc="Running tokenizer on validation dataset",
    )

    # Transform images on the fly as doing it on the whole dataset takes too much time.
    eval_dataset.set_transform(transform_images)



    trainer = Trainer(
        clip_model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        data_collator=collate_fn,
        tokenizer=preprocessor
    )
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    image_processor.save_pretrained(training_args.output_dir)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    # 10. Evaluation
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()
