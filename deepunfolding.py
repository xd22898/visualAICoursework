import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataloader import DIV2K_SRDataset

# --- Configuration ---
SCALE_FACTOR = 8 
LR_DIR = f"./DIV2K_train_LR_x8"  
HR_DIR = "./DIV2K_train_HR"


data_transform = transforms.Compose([
    transforms.ToTensor(), # Converts to (3, H, W)
])

# Instantiate Dataset and DataLoader
test_dataset = DIV2K_SRDataset(lr_dir=LR_DIR, hr_dir=HR_DIR, transform=data_transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

print (len(test_dataset))

# Load one batch
lr_batch, hr_batch = next(iter(test_loader))

print("--- DataLoader Verification ---")
print(f"Dataset Size: {len(test_dataset)}")
print(f"LR Batch Shape: {lr_batch.shape}")
print(f"HR Batch Shape: {hr_batch.shape}")
print(f"LR Image Pixel Range (min/max): {lr_batch.min():.4f} / {lr_batch.max():.4f}")

# --- Critical Check for Super-Resolution ---
# Verify the height (H) and width (W) ratio matches the scale factor (e.g., x8)
B, C, H_LR, W_LR = lr_batch.shape
_, _, H_HR, W_HR = hr_batch.shape

print(f"\nLR Size: {H_LR}x{W_LR} (Channels: {C})")
print(f"HR Size: {H_HR}x{W_HR} (Channels: {C})")
print(f"Calculated Scale Factor: {H_HR / H_LR:.1f}")

if C != 3 or (H_HR / H_LR) != SCALE_FACTOR:
    print("❌ ERROR: Channel or Scale factor mismatch. Check paths and image dimensions!")
else:
    print("✅ DataLoader appears correct!")