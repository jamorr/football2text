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
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


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
    trainer = Trainer(
        clip_model,
        train_dataset=NFLVisionTextDataset(root_dir/"data"/"train"),
        eval_dataset=NFLVisionTextDataset(root_dir/"data"/"val"),
        tokenizer=preprocessor

    )
    # collate_fn=collate_fn,
    trainer.train()

if __name__ == "__main__":
    main()
