import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import tifffile
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import rotate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################################
# CONFIG
############################################
PATCH_SIZE = 256           # Size of each patch
PATCHES_PER_IMAGE = 8      # Number of patches per image during training
BATCH_SIZE = 16            # Total patches = BATCH_SIZE * PATCHES_PER_IMAGE
EPOCHS = 50

############################################
# PATHS
############################################
TRAIN_IMG_DIR = "../experimental_dataset/expo_dataset_v3/train/images"
TRAIN_CSV     = "../experimental_dataset/expo_dataset_v3/train/labels.csv"
VAL_IMG_DIR   = "../experimental_dataset/expo_dataset_v3/val/images"
VAL_CSV       = "../experimental_dataset/expo_dataset_v3/val/labels.csv"

############################################
# DATASET
############################################
class HoloDataset(Dataset):
    def __init__(self, img_dir, csv_file, z_mean=None, z_std=None, train=True,
                 patch_size=PATCH_SIZE, patches_per_image=PATCHES_PER_IMAGE,
                 deterministic=False):
        """
        deterministic: if True, selects 5-9 fixed patches per image (center + corners) instead of random
        """
        self.img_dir = img_dir
        self.df = pd.read_csv(csv_file)
        self.img_names = self.df.iloc[:,0].values
        self.z_values  = self.df.iloc[:,1].values.astype(np.float32)

        self.train = train
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.deterministic = deterministic

        # Z normalization
        if z_mean is None:
            self.z_mean = self.z_values.mean()
            self.z_std = self.z_values.std()
        else:
            self.z_mean = z_mean
            self.z_std = z_std

        self.z_norm = (self.z_values - self.z_mean) / (self.z_std + 1e-6)

        # Image normalization params
        self.mean = 0.5
        self.std = 0.25

    def __len__(self):
        return len(self.img_names)

    def get_patch(self, img):
        """Return a single patch, either random or deterministic"""
        h, w = img.shape

        if self.deterministic:
            # Sample center or corners (or random if more patches)
            positions = [
                (0,0),
                (0, w-self.patch_size),
                (h-self.patch_size,0),
                (h-self.patch_size,w-self.patch_size),
                (h//2 - self.patch_size//2, w//2 - self.patch_size//2)
            ]
            idx = np.random.randint(0, len(positions))
            y, x = positions[idx]
        else:
            y = np.random.randint(0, h - self.patch_size)
            x = np.random.randint(0, w - self.patch_size)

        return img[y:y+self.patch_size, x:x+self.patch_size]

    def augment_patch(self, patch):
        """Apply flips, rotation, shift, noise"""
        if np.random.rand() > 0.5:
            patch = np.flip(patch, axis=0)
        if np.random.rand() > 0.5:
            patch = np.flip(patch, axis=1)
        angle = np.random.uniform(-15,15)
        patch = rotate(patch, angle, reshape=False, order=1, mode='reflect')
        shift_x = np.random.randint(-4,5)
        shift_y = np.random.randint(-4,5)
        patch = np.roll(patch, shift=(shift_x, shift_y), axis=(0,1))
        patch = patch * np.random.uniform(0.9,1.1)
        patch = patch + np.random.normal(0,0.01,patch.shape)
        return patch

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = tifffile.imread(img_path).astype(np.float32)
        if img.ndim == 3:
            img = img.mean(axis=-1)

        patches = []
        for _ in range(self.patches_per_image):
            patch = self.get_patch(img)
            if self.train:
                patch = self.augment_patch(patch)
            patch = (patch - self.mean)/self.std
            patch = torch.from_numpy(patch).unsqueeze(0).float()
            patches.append(patch)

        patches = torch.stack(patches)  # [P,1,H,W]
        z = torch.tensor(self.z_norm[idx]).float().repeat(self.patches_per_image)
        return patches, z

############################################
# RESIDUAL BLOCK
############################################
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

############################################
# SPATIAL BRANCH
############################################
class SpatialBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.res_blocks = nn.Sequential(ResidualBlock(32), ResidualBlock(32))
        self.downsample = nn.Sequential(
            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.res_blocks2 = nn.Sequential(ResidualBlock(64), ResidualBlock(64))
        self.final = nn.Sequential(
            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
    def forward(self,x):
        x = self.initial(x)
        x = self.res_blocks(x)
        x = self.downsample(x)
        x = self.res_blocks2(x)
        x = self.final(x)
        return x.view(x.size(0), -1)

############################################
# FFT BRANCH
############################################
class FFTBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(2,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.res_blocks = nn.Sequential(ResidualBlock(32), ResidualBlock(32))
        self.downsample = nn.Sequential(
            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.res_blocks2 = nn.Sequential(ResidualBlock(64), ResidualBlock(64))
        self.final = nn.Sequential(
            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
    def forward(self,x):
        fft = torch.fft.fft2(x)
        fft = torch.fft.fftshift(fft)
        mag = torch.log1p(torch.abs(fft))
        phase = torch.angle(fft)
        mag = (mag - mag.mean(dim=(-2,-1), keepdim=True)) / (mag.std(dim=(-2,-1), keepdim=True)+1e-6)
        phase = (phase - phase.mean(dim=(-2,-1), keepdim=True)) / (phase.std(dim=(-2,-1), keepdim=True)+1e-6)
        fft_input = torch.cat([mag,phase], dim=1)
        x = self.initial(fft_input)
        x = self.res_blocks(x)
        x = self.downsample(x)
        x = self.res_blocks2(x)
        x = self.final(x)
        return x.view(x.size(0), -1)

############################################
# HoloNet
############################################
class HoloNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = SpatialBranch()
        self.fft = FFTBranch()
        self.fc = nn.Sequential(
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128,1)
        )
    def forward(self,x):
        s = self.spatial(x)
        f = self.fft(x)
        feat = torch.cat([s,f], dim=1)
        z = self.fc(feat)
        return z.squeeze()

############################################
# TRAIN / VALIDATION
############################################
def train_epoch(model,loader,optimizer,criterion):
    model.train()
    total_loss = 0
    for imgs,z in loader:
        B,P,C,H,W = imgs.shape
        imgs = imgs.view(B*P,C,H,W).to(device)
        z = z.view(B*P).to(device)
        optimizer.zero_grad()
        pred = model(imgs)
        loss = criterion(pred,z)
        if torch.isnan(loss): continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(loader)

def validate(model,loader,criterion,z_mean,z_std):
    model.eval()
    total_loss = 0
    mae = 0
    with torch.no_grad():
        for imgs,z in loader:
            B,P,C,H,W = imgs.shape
            imgs = imgs.view(B*P,C,H,W).to(device)
            z = z.view(B*P).to(device)
            pred = model(imgs)
            loss = criterion(pred,z)
            total_loss += loss.item()
            pred_um = pred * z_std + z_mean
            z_um = z * z_std + z_mean
            mae += torch.mean(torch.abs(pred_um - z_um)).item()
    return total_loss/len(loader), mae/len(loader)

############################################
# MAIN
############################################
def main():
    # Datasets
    train_dataset = HoloDataset(TRAIN_IMG_DIR, TRAIN_CSV, train=True)
    z_mean = train_dataset.z_mean
    z_std = train_dataset.z_std
    val_dataset = HoloDataset(VAL_IMG_DIR, VAL_CSV, z_mean, z_std, train=False)

    train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4,pin_memory=True)
    val_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=4,pin_memory=True)

    # Model
    model = HoloNet().to(device)
    optimizer = optim.Adam(model.parameters(),lr=3e-4,weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    best_mae = float("inf")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_mae = validate(model, val_loader, criterion, z_mean, z_std)
        print(f"Train Loss: {train_loss:.6f} | Val MAE (µm): {val_mae:.4f}")
        scheduler.step(val_mae)
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), "best_model_fft_final.pth")
            print("✓ Best model saved")

if __name__ == "__main__":
    main()
