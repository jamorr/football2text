import pathlib

from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    HfArgumentParser,
    RobertaForMaskedLM,
    Trainer,
    TrainingArguments,
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    ViTImageProcessor,
    ViTMAEForPreTraining,
)
def main():
    here = pathlib.Path(__file__).parent
    root_dir = here.parents[1]
    models_dir = root_dir / "models"
    vit_ver = "1"

    vit_pretrained = ViTMAEForPreTraining.from_pretrained(
        models_dir / "ViT" / vit_ver,
        from_tf=False,  # true if from ckpt
    )
    vit_model = vit_pretrained.vit
    vit_encoder_dir = models_dir / "ViT_encoder" / vit_ver
    if not (vit_encoder_dir).exists():
        vit_encoder_dir.mkdir(parents=True)
        vit_model.save_pretrained(vit_encoder_dir)


    clip_model = VisionTextDualEncoderModel.from_vision_text_pretrained(
        vit_encoder_dir, "jkruk/distilroberta-base-ft-nfl"
    )

# tokenizer = AutoTokenizer.from_pretrained("jkruk/distilroberta-base-ft-nfl")
# image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
if __name__ == "__main__":
    main()