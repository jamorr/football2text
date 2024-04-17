import datetime
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
from torchvision.transforms import Compose, Lambda, Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor
from torchvision.transforms.functional import InterpolationMode



import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from dataset import NFLImageDataset
# make the plt figure larger
plt.rcParams['figure.figsize'] = [20, 5]

def show_image(image, title, ax:Axes):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    ax.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    ax.set_title(title, fontsize=16)
    ax.axis('off')

def visualize(pixel_values, model, save_loc):
    # forward pass
    outputs = model(pixel_values)
    y = model.unpatchify(outputs.logits)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = outputs.mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.config.patch_size**2 *3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    x = torch.einsum('nchw->nhwc', pixel_values)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask


    fig, axes = plt.subplots(1, 4, layout="constrained")

    show_image(x[0], "original", axes[0])
    show_image(im_masked[0], "masked",axes[1])
    show_image(y[0], "reconstruction", axes[2])
    show_image(im_paste[0], "reconstruction + visible", axes[3])
    fig.savefig(save_loc)

def main():
    here = pathlib.Path(__file__).parent
    root_dir = here.parents[1]
    models_dir = root_dir / "models"
    vis_save_dir = root_dir / "assets" / "ViT_examples"
    which = "test"
    vit_ver = "2/checkpoint-5922"
    vit_pretrained = ViTMAEForPreTraining.from_pretrained(models_dir / "ViT" / vit_ver,)
    image_processor = ViTImageProcessor.from_pretrained(models_dir / "ViT" / vit_ver,)
    global imagenet_mean, imagenet_std
    imagenet_mean = np.array(image_processor.image_mean)
    imagenet_std = np.array(image_processor.image_std)
    data_dir = pathlib.Path(__file__).parents[2]/"data"/which


    # image_processor = ViTFeatureExtractor.from_pretrained("facebook/vit-mae-base")
    dset = NFLImageDataset(data_dir)
    timestamp = datetime.datetime.now().strftime(r"%d_%m_%Y__%H_%M_%S")
    # feature_extractor = vit_pretrained.vit
    visualize(image_processor(dset[0], return_tensors="pt").pixel_values, vit_pretrained, save_loc=vis_save_dir/f"test{timestamp}.jpeg")

if __name__ == "__main__":
    main()