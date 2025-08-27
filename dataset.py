import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import glob
import torchvision.transforms as transforms
import numpy as np

# A helper function to add noise, matching your test script
def add_noise(image, noise_level=25/255.0):
    noise = torch.randn_like(image) * noise_level
    noisy_image = torch.clamp(image + noise, 0., 1.)
    return noisy_image

class DenoisingDataset(Dataset):
    def __init__(self, root_dir, model_type='color'):
        super().__init__()
        self.root_dir = root_dir
        self.model_type = model_type
        
        image_files = glob.glob(os.path.join(root_dir, "**", "*.png"), recursive=True)
        if not image_files:
            print(f"Warning: No images found in {root_dir}")
        self.image_files = image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = self.image_files[index]
        
        image = Image.open(image_path)
        if self.model_type == 'grayscale':
            image = image.convert('L')
        else:
            image = image.convert('RGB')
        
        # Define the complete augmentation pipeline with corrected RandomRotation
        augmentation = transforms.Compose([
            transforms.RandomCrop((50, 50)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomChoice([
                transforms.RandomRotation([0, 0]),
                transforms.RandomRotation([90, 90]),
                transforms.RandomRotation([180, 180]),
                transforms.RandomRotation([270, 270])
            ]),
            transforms.ToTensor()
        ])
        
        clean_patch = augmentation(image)
        noisy_patch = add_noise(clean_patch)
        
        return noisy_patch, clean_patch