import pathlib
from towhee import pipe,ops,DataCollection
import torchvision
from towhee.trainer.training_config import dump_default_yaml, TrainingConfig
from towhee.trainer.trainer import Trainer
from dataset import NFLDataset
#Takes video file & metadata

model = ops.video_text_embedding.clip4clip(model_name='clip_vit_b32', modality='video').get_op()
# model = (
#     pipe.input('vid','text')
#         .map('text', 'tvec', ops.video_text_embedding.clip4clip(model_name='clip_vit_b32', modality='text', device='cuda')) \
#         # .map('vid', 'vvec', ) \
#         .output('tvec')
# )


training_config = TrainingConfig(
    output_dir="../models/train_model",
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
