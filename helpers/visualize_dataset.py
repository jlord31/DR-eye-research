import matplotlib.pyplot as plt
from collections import Counter
import torch

def visualize_samples(dataset, num_samples=5):
    """
        Visualize a few samples from the dataset.
        Args:
            dataset (torch.utils.data.Dataset): The dataset to visualize.
            num_samples (int): Number of samples to visualize.

        Returns:
            None  
    """
    for i in range(num_samples):
        image, label = dataset[i]
        plt.imshow(image.permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        plt.title(f"Label: {label}")
        plt.axis("off")
        plt.show()


def check_class_distribution(dataset):
    """
        Check the class distribution in the dataset.
        Args:
            dataset (torch.utils.data.Dataset): The dataset to analyze.

        Returns:
            None
    """
    if isinstance(dataset, torch.utils.data.Subset):
        # Access the original dataset and subset indices
        labels = [dataset.dataset.labels[i] for i in dataset.indices]
    else:
        # Standard dataset handling
        labels = [label for _, label in dataset]

    class_counts = Counter(labels)
    print("Class Distribution:", class_counts)
