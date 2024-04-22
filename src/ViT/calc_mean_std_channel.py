from datasets import load_dataset
import numpy as np
from PIL import Image
import torch

# Load the dataset
dataset = load_dataset("/media/jj_data/data/CLIP", name="main", split="train")  # Replace "dataset_name" with the name of your dataset

# Function to process each example
def process_example(example):
    # Load the image
    image_path = example["pixel_values"]  # Assuming 'pixel_values' is the key for image paths
    img = Image.open(image_path)  # Assuming you have imported Image from PIL
    # Convert the image to a tensor
    img_tensor = torch.tensor(np.array(img)) / 255.0  # Normalize pixel values to [0, 1]
    # Calculate mean and std for each channel
    mean = torch.mean(img_tensor, dim=(0, 1))
    std = torch.std(img_tensor, dim=(0, 1))
    return {"mean": mean.tolist(), "std": std.tolist()}  # Convert tensors to lists for serialization

# Apply the processing function to each example in the dataset
processed_dataset = dataset.map(process_example, batched=False)

# Calculate overall mean and std for each channel
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