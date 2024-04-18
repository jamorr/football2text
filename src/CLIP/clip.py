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
import torch

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

        # 7. Preprocessing the datasets.
    # Initialize torchvision transforms and jit it for faster processing.
    image_transformations = Transform(
        224, image_processor.image_mean, image_processor.image_std
    )
    image_transformations = torch.jit.script(image_transformations)

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples):
        captions = list(examples[caption_column])
        text_inputs = tokenizer(captions, max_length=data_args.max_seq_length, padding="max_length", truncation=True)
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






    trainer = Trainer(
        clip_model,
        train_dataset=NFLVisionTextDataset(root_dir/"data"/"train"),
        eval_dataset=NFLVisionTextDataset(root_dir/"data"/"val"),
        data_collator=collate_fn,
        tokenizer=preprocessor
    )
    trainer.train()

if __name__ == "__main__":
    main()
