# New imports needed for image data handling
import os
import torch
from PIL import Image
import glob
from torch.utils.data import Dataset # Make sure to explicitly import Dataset

class DIV2K_SRDataset(Dataset): # Inheriting from Dataset
    def __init__(self, lr_dir, hr_dir, transform=None):
        # Load file paths for LR and HR images
        self.lr_files = sorted(glob.glob(os.path.join(lr_dir, '*.png')))
        self.hr_files = sorted(glob.glob(os.path.join(hr_dir, '*.png')))
        self.transform = transform
        
        # Ensure files match (e.g., DIV2K_0001.png vs DIV2K_0001x8.png)
        if len(self.lr_files) != len(self.hr_files):
             raise ValueError(f"Mismatch in LR and HR file counts: {len(self.lr_files)} LR files found, but {len(self.hr_files)} HR files found.")

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr_path = self.hr_files[idx]
        lr_path = self.lr_files[idx]
        
        # Load images - RGB to make sure everything is in 3 channels
        hr_img = Image.open(hr_path).convert('RGB')
        lr_img = Image.open(lr_path).convert('RGB')
        
        # Apply the shared transform (which should include ToTensor and Normalize)
        if self.transform:
            hr_img = self.transform(hr_img)
            lr_img = self.transform(lr_img)
        
        # Return the Low-Resolution image (input) and the High-Resolution image (target)
        return lr_img, hr_img