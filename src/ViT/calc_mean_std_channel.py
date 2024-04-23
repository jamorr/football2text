from datasets import load_dataset
import numpy as np
from PIL import Image
import torch

def calc_mean_std_images(dataset, name, split):
    dataset = load_dataset(dataset, name=name, split=split)

    def process_example(example):
        image_path = example["pixel_values"]
        img = Image.open(image_path)
        img_tensor = torch.tensor(np.array(img)) / 255.0
        # Calculate mean and std for each channel
        mean = torch.mean(img_tensor, dim=(0, 1))
        std = torch.std(img_tensor, dim=(0, 1))
        return {"mean": mean.tolist(), "std": std.tolist()}  # Convert tensors to lists for serialization

    # Apply to each example in the dataset
    processed_dataset = dataset.map(process_example, batched=False)

    # calculate overall mean and std for each channel
    mean_sum = np.zeros(3)
    std_sum = np.zeros(3)
    num_samples = len(processed_dataset)
    for example in processed_dataset:
        mean_sum += np.array(example["mean"])
        std_sum += np.array(example["std"])

    overall_mean = mean_sum / num_samples
    overall_std = std_sum / num_samples

    print("Mean for each channel (R, G, B):", overall_mean)
    print("Standard deviation for each channel (R, G, B):", overall_std)
    return overall_mean, overall_std

if __name__ == "__main__":
    calc_mean_std_images("/media/jj_data/data/CLIP", "main", "train")

