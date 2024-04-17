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
here = pathlib.Path(__file__).parent
root_dir = here.parents[1]
models_dir = root_dir / "models"
vit_ver = "1"
vit_pretrained = ViTMAEForPreTraining.from_pretrained(
        models_dir / "ViT" / vit_ver,
    )

