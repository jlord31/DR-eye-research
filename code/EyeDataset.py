import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_SIZE = 224

class EyeDataset(Dataset):
    def __init__(self, root_dir, transform=None, return_paths=False):
        """
            Custom dataset class that loads images and their labels from a directory.
            The directory structure should have one subfolder per class.
            Args:
                root_dir (str): Path to the dataset directory.
                transform (callable, optional): A function/transform to apply to the images.
                return_paths (bool): If True, returns image paths along with images and labels.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.return_paths = return_paths

        # Get sorted class names and map them to numerical labels
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Collect image file paths and corresponding labels
        self.image_paths = []
        self.labels = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if os.path.isdir(cls_dir):
                for img_name in os.listdir(cls_dir):
                    img_path = os.path.join(cls_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load and preprocess the image
        img_path = self.image_paths[idx]
        original_image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(original_image)
        else:
            image = transforms.ToTensor()(original_image)

        if self.return_paths:
            return image, label, img_path

        else:
            return image, label
