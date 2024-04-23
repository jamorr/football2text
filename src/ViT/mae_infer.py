import datetime
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import NFLJPEGDataset
from matplotlib.axes import Axes
from transformers import (
    ViTImageProcessor,
    ViTMAEForPreTraining,
)


def show_image(image, title, ax: Axes):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    ax.imshow(torch.clip((image * image_data_std + image_data_mean) * 255, 0, 255).int())
    ax.set_title(title, fontsize=16)
    ax.axis("off")


def visualize(pixel_values, model, save_loc):
    # forward pass
    outputs = model(pixel_values)
    y = model.unpatchify(outputs.logits)
    y = torch.einsum("nchw->nhwc", y).detach().cpu()

    # visualize the mask
    mask = outputs.mask.detach()
    mask = mask.unsqueeze(-1).repeat(
        1, 1, model.config.patch_size**2 * 3
    )  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum("nchw->nhwc", mask).detach().cpu()

    x = torch.einsum("nchw->nhwc", pixel_values)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    fig, axes = plt.subplots(1, 4, layout="constrained")
    fig.set_size_inches(20, 5)
    show_image(x[0], "original", axes[0])
    show_image(im_masked[0], "masked", axes[1])
    show_image(y[0], "reconstruction", axes[2])
    show_image(im_paste[0], "reconstruction + visible", axes[3])
    fig.savefig(save_loc)


def visualize_inference(version_num, checkpoint_name, root_dir):
    here = pathlib.Path(__file__).parent
    models_dir = root_dir / "models"
    vis_save_dir = here.parents[1] / "assets" / "ViT_examples"
    which = "test"
    model_path = models_dir / "ViT" / version_num
    if checkpoint_name is not None:
        model_path = model_path / checkpoint_name
    else:
        checkpoint_name = 0
    vit_pretrained = ViTMAEForPreTraining.from_pretrained(model_path)
    image_processor = ViTImageProcessor.from_pretrained(model_path)
    global image_data_mean, image_data_std
    image_data_mean = np.array(image_processor.image_mean)
    image_data_std = np.array(image_processor.image_std)
    data_dir = root_dir / "data" / which

    # image_processor = ViTFeatureExtractor.from_pretrained("facebook/vit-mae-base")
    dset = NFLJPEGDataset(data_dir)
    timestamp = datetime.datetime.now().strftime(r"%d_%m_%Y__%H_%M_%S")
    # feature_extractor = vit_pretrained.vit
    idx = np.random.randint(0, len(dset))

    visualize(
        image_processor(dset[idx], return_tensors="pt").pixel_values,
        vit_pretrained,
        save_loc=vis_save_dir
        / f"test{timestamp}-v.{version_num}.{checkpoint_name}.jpeg",
    )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-v", dest="version_num", default="5")
    parser.add_argument("-c", dest="checkpoint_name", default=None)
    parser.add_argument("-d", dest="root_dir", default=pathlib.Path("/media/jj_data"))
    args = parser.parse_args()

    visualize_inference(args.version_num, args.checkpoint_name,  args.root_dir)
