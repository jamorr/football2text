import pathlib
from towhee import pipe,ops,DataCollection
import torchvision
from towhee.trainer.training_config import dump_default_yaml, TrainingConfig
from towhee.trainer.trainer import Trainer
from dataset import NFLDataset
from towhee.models.clip4clip import CLIP4Clip, create_model

# model = create_model()
#Takes video file & metadata
model = CLIP4Clip(**dict(
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
        transformer_layers=12
    ))
# model = ops.video_text_embedding.clip4clip(model_name='clip_vit_b32', modality='video', device='cpu').get_op()
# model = (
#     pipe.input('vid','text')
#         .map('text', 'tvec', ops.video_text_embedding.clip4clip(model_name='clip_vit_b32', modality='text', device='cuda')) \
#         # .map('vid', 'vvec', ) \
#         .output('tvec')
# )


training_config = TrainingConfig(
    output_dir="../models",
    epoch_num=2,
    batch_size=1,
    print_steps=1,

)

if __name__ == "__main__":
    which = "train"
    data_dir = pathlib.Path(__file__).parents[1] / "data"
    train_data = NFLDataset(data_dir/which,True)
    which = "val"
    eval_data = NFLDataset(data_dir/which,True)

    trainer = Trainer(model, training_config, train_dataset=train_data, eval_dataset=eval_data)
    # data_dir = pathlib.Path(__file__).parents[1] / "data" / "val"
    # dset = NFLDataset(data_dir, True)

    trainer.train()
    # for i in range(len(dset))[:10]:
    #     model(dset[i][0])
