import argparse
import os
import pathlib
import subprocess
import sys

from lightning.pytorch import LightningDataModule, seed_everything
from src.ViT import write_mp4_to_jpeg
from src.dataset import NFLDataModule
from src.CLIP import create_parquets
from src.ViT import calc_mean_std_images, NFLJPEGDataset, write_mp4_to_jpeg

data_path = pathlib.Path("/data")
models_path = pathlib.Path("/media/jj_data/models")
def main():
    parser = argparse.ArgumentParser(description='CLIP for football')
    #----------------------------------------------Sys Args--------------------------------------------------
    parser.add_argument('--data-path', dest='data_dir', type=str, default=data_path)
    parser.add_argument('--checkpoint-path', dest='check_path',type=str, default=None, help='Location of checkpoint to use or ')
    parser.add_argument('--save-path', dest='save_path',type=str, default=models_path, help='Location to save checkpoints')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--print-model', action='store_true')
    parser.add_argument('--num-labels', dest='num_labels', type=int, default=7, help='Number of classes in target')
    parser.add_argument('--modalities', type=str, default='T+A+V', help="Which modalities to use in model")
    parser.add_argument('--prep-data', action="prepare_data", help='Prepare datasets from starting CSVs')

    #----------------------------------------------Training Args--------------------------------------------------
    parser.add_argument('--num-epochs', dest='num_epochs', type=int, default=8, help='number of epochs')
    parser.add_argument('--num-workers', dest='num_workers', type=int, default=4, help='number workers in dataloader')
    parser.add_argument('--learning-rate',dest='learning_rate', type=float, default=7e-5, help='initial learning rate')
    parser.add_argument('--batch-size-train', dest='batch_size_train', type=int, default=1, help='Training batch size')
    parser.add_argument('--batch-size-val', dest='batch_size_val', type=int, default=18, help='Training batch size')
    parser.add_argument('--grad-accumulation-steps', dest='grad_accumulation_steps', type=int, default=1,
                        help='gradient accumulation for trg task')
    parser.add_argument('--warm-up',dest='warm_up', type=float, default=0.1, help='Proportion of warmup steps to total training steps. Used to calibrate learning rate.')

    parser.add_argument('--weight-decay', dest='weight_decay', type=float, default=0.01, help='Weight decay hyperparameter.')
    parser.add_argument('--patience', type=int, default=100, help='Early stopping patience')
    parser.add_argument('--clip', type=float, default=0.8, help='Gradient clip value')
    parser.add_argument('--log-interval',dest='log_interval', type=int, default=50, help='Number of training steps between logging')
    parser.add_argument('--seed', type=int, default=37, help='random seed')
    parser.add_argument('--load-best', action="store_true", help='load best checkpoint')


    parser.add_argument('--train-vit', action="train_vit", help='Train ViT from scratch')
    parser.add_argument('--train-roberta', action="train_roberta", help='Finetune Football RoBERTa from pretrained')
    parser.add_argument('--train-clip', action="train_clip", help='Finetune CLIP from pretrained')
    parser.add_argument('--train-c4c', action="train_c4c", help='Finetune CLIP4Clip from pretrained CLIP')
    return parser.parse_args()

if __name__ == "__main__":
    args = main()
    seed_everything(args.seed, True)

    if args.prepare_data:
        dm = NFLDataModule(args.data_path)
        dm.prepare_data()
        write_mp4_to_jpeg(args.data_path)
        create_parquets(args.data_path)


    here = pathlib.Path(__file__).parent

    if args.train_vit:
        floc = here / "src" / "ViT"
        floc = str(floc.absolute())
        process = subprocess.Popen(
            f"python {floc}/train_script.sh",
            stdout=subprocess.PIPE,
            text=True,
        )
        out, err = process.communicate()

        result = out.split('\n')
        for lin in result:
            if not lin.startswith('#'):
                print(lin)


    if args.train_roberta:
        floc = here / "src" / "RoBERTa"
        floc = str(floc.absolute())
        process = subprocess.Popen(
            f"python {floc}/train.sh",
            stdout=subprocess.PIPE,
            text=True,
        )

        out, err = process.communicate()

        result = out.split('\n')
        for lin in result:
            if not lin.startswith('#'):
                print(lin)


    if args.train_clip:
        floc = here / "src" / "RoBERTa"
        floc = str(floc.absolute())
        process = subprocess.Popen(
            f"python {floc}/train.sh",
            stdout=subprocess.PIPE,
            text=True,
        )
        out, err = process.communicate()

        result = out.split('\n')
        for lin in result:
            if not lin.startswith('#'):
                print(lin)

