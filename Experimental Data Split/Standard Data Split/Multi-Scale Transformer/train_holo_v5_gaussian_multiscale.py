import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import tifffile
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################################
# CONFIG
############################################

BATCH_SIZE = 8
EPOCHS = 100

PATCH_SIZE_BIG = 256
PATCH_SIZE_SMALL = 128

PATCHES_PER_IMAGE_TRAIN = 32
PATCHES_PER_IMAGE_VAL   = 16

ACCUM_STEPS = 2

EARLY_STOPPING_PATIENCE = 20
MIN_DELTA = 0.0005

NUM_BINS = 256

GAUSS_SIGMA_NORM = 0.15

NORM_FILE = "normalization_params.json"

############################################
# DATASET
############################################

class HoloDataset(Dataset):

    def __init__(
        self,
        img_dir,
        csv_file,
        train=True,
        z_mean=None,
        z_std=None,
        z_norm_min=None,
        z_norm_max=None,
        img_mean=None,
        img_std=None
    ):
        self.img_dir = img_dir
        self.df = pd.read_csv(csv_file)

        self.img_names = self.df.iloc[:, 0].values
        self.z_values  = self.df.iloc[:, 1].values.astype(np.float32)

        self.train = train

        assert z_mean is not None and z_std is not None
        assert z_norm_min is not None and z_norm_max is not None
        assert img_mean is not None and img_std is not None

        self.z_mean = float(z_mean)
        self.z_std = float(z_std)
        self.z_norm_min = float(z_norm_min)
        self.z_norm_max = float(z_norm_max)

        self.img_mean = float(img_mean)
        self.img_std = float(img_std)

        self.bin_centers = torch.linspace(self.z_norm_min, self.z_norm_max, NUM_BINS)

    def __len__(self):
        return len(self.img_names)

    ############################################
    # GLOBAL IMAGE NORMALIZATION (FIXED)
    ############################################
    def normalize_image_global(self, img):
        return (img - self.img_mean) / (self.img_std + 1e-6)

    ############################################
    # PATCH NORMALIZATION (LOCAL)
    ############################################
    def normalize_patch(self, x):
        return (x - x.mean()) / (x.std() + 1e-6)

    def gradient_map(self, img):
        gx = torch.gradient(img, dim=0)[0]
        gy = torch.gradient(img, dim=1)[0]
        return torch.sqrt(gx**2 + gy**2)

    def fft_magnitude(self, img):
        # CRITICAL: fftshift for hologram frequency patterns
        fft = torch.fft.fftshift(torch.fft.fft2(img))
        mag = torch.log1p(torch.abs(fft))
        return mag

    def get_patch_train(self, img):
        H, W = img.shape
        y = np.random.randint(0, H - PATCH_SIZE_BIG)
        x = np.random.randint(0, W - PATCH_SIZE_BIG)
        return img[y:y+PATCH_SIZE_BIG, x:x+PATCH_SIZE_BIG]

    def get_patch_val(self, img, i):
        H, W = img.shape
        grid = 4

        step_y = (H - PATCH_SIZE_BIG) // (grid - 1)
        step_x = (W - PATCH_SIZE_BIG) // (grid - 1)

        row = i // grid
        col = i % grid

        y = row * step_y
        x = col * step_x

        return img[y:y+PATCH_SIZE_BIG, x:x+PATCH_SIZE_BIG]

    def center_crop(self, patch, crop_size=128):
        H, W = patch.shape
        y0 = (H - crop_size) // 2
        x0 = (W - crop_size) // 2
        return patch[y0:y0+crop_size, x0:x0+crop_size]

    ############################################
    # Z NORMALIZATION (REVERSIBLE)
    ############################################
    def normalize_z(self, z_value):
        return (z_value - self.z_mean) / (self.z_std + 1e-8)

    def denormalize_z(self, z_norm):
        return z_norm * self.z_std + self.z_mean

    def gaussian_soft_label(self, z_norm_value):
        centers = self.bin_centers.to(z_norm_value.device)
        sigma = GAUSS_SIGMA_NORM

        dist = torch.exp(-0.5 * ((centers - z_norm_value) / sigma) ** 2)
        dist = dist / (dist.sum() + 1e-8)
        return dist

    def __getitem__(self, idx):

        img_path = os.path.join(self.img_dir, self.img_names[idx])
        img = tifffile.imread(img_path).astype(np.float32)

        if img.ndim == 3:
            img = img.mean(axis=-1)

        img = torch.from_numpy(img).float()

        # GLOBAL normalization
        img = self.normalize_image_global(img)

        patches = []
        num_patches = PATCHES_PER_IMAGE_TRAIN if self.train else PATCHES_PER_IMAGE_VAL

        for i in range(num_patches):

            if self.train:
                patch_big = self.get_patch_train(img)
            else:
                patch_big = self.get_patch_val(img, i)

            patch_small = self.center_crop(patch_big, PATCH_SIZE_SMALL)

            # BIG channels
            big_raw = self.normalize_patch(patch_big)
            big_grad = self.normalize_patch(self.gradient_map(patch_big))
            big_fft = self.normalize_patch(self.fft_magnitude(patch_big))

            # SMALL channels
            small_raw = self.normalize_patch(patch_small)
            small_grad = self.normalize_patch(self.gradient_map(patch_small))
            small_fft = self.normalize_patch(self.fft_magnitude(patch_small))

            # resize small channels to 256x256
            small_raw = F.interpolate(
                small_raw.unsqueeze(0).unsqueeze(0),
                size=(PATCH_SIZE_BIG, PATCH_SIZE_BIG),
                mode="bilinear",
                align_corners=False
            ).squeeze()

            small_grad = F.interpolate(
                small_grad.unsqueeze(0).unsqueeze(0),
                size=(PATCH_SIZE_BIG, PATCH_SIZE_BIG),
                mode="bilinear",
                align_corners=False
            ).squeeze()

            small_fft = F.interpolate(
                small_fft.unsqueeze(0).unsqueeze(0),
                size=(PATCH_SIZE_BIG, PATCH_SIZE_BIG),
                mode="bilinear",
                align_corners=False
            ).squeeze()

            stacked = torch.stack([
                big_raw, big_grad, big_fft,
                small_raw, small_grad, small_fft
            ])

            patches.append(stacked)

        patches = torch.stack(patches)

        z_value = torch.tensor(self.z_values[idx]).float()
        z_norm_value = self.normalize_z(z_value)
        soft_target = self.gaussian_soft_label(z_norm_value)

        return patches, soft_target, z_value

############################################
# MODEL
############################################

class PatchEncoder(nn.Module):

    def __init__(self, in_ch=6, embed_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),

            nn.Conv2d(256, embed_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),

            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        x = self.net(x)
        return x.view(x.size(0), -1)


class DepthTransformerGaussian(nn.Module):

    def __init__(self, in_ch=6, embed_dim=256, num_layers=4, nhead=8, num_bins=256):
        super().__init__()

        self.encoder = PatchEncoder(in_ch=in_ch, embed_dim=embed_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_bins)
        )

    def forward(self, x):
        B, P, C, H, W = x.shape

        x = x.view(B * P, C, H, W)
        feats = self.encoder(x)
        feats = feats.view(B, P, -1)

        feats = self.transformer(feats)
        global_feat = feats.mean(dim=1)

        logits = self.classifier(global_feat)
        return logits

############################################
# DECODING
############################################

def decode_expected_z(logits, bin_centers, z_mean, z_std):
    probs = torch.softmax(logits, dim=1)
    expected_norm = torch.sum(probs * bin_centers.unsqueeze(0), dim=1)
    expected_um = expected_norm * z_std + z_mean
    return expected_um

############################################
# TRAIN / VAL
############################################

def train_epoch(model, loader, optimizer, scaler):

    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for step, (imgs, soft_target, z_value) in enumerate(tqdm(loader, desc="Train", leave=False)):

        imgs = imgs.to(device, non_blocking=True)
        soft_target = soft_target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            logits = model(imgs)
            log_probs = F.log_softmax(logits, dim=1)

            loss = F.kl_div(log_probs, soft_target, reduction="batchmean") / ACCUM_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * ACCUM_STEPS

    return total_loss / len(loader)


def validate(model, loader, bin_centers, z_mean, z_std):

    model.eval()
    mae = 0

    with torch.no_grad():
        for imgs, soft_target, z_value in tqdm(loader, desc="Val", leave=False):

            imgs = imgs.to(device, non_blocking=True)
            z_value = z_value.to(device, non_blocking=True)

            logits = model(imgs)
            pred_z = decode_expected_z(logits, bin_centers, z_mean, z_std)

            mae += torch.mean(torch.abs(pred_z - z_value)).item()

    return mae / len(loader)

############################################
# GLOBAL NORMALIZATION COMPUTATION
############################################

def compute_train_stats(train_csv, train_img_dir, max_images=3000):
    """
    Computes:
    - z_mean, z_std
    - z_norm_min, z_norm_max
    - img_mean, img_std  (global image normalization)
    """

    df = pd.read_csv(train_csv)
    z = df.iloc[:, 1].values.astype(np.float32)

    z_mean = float(z.mean())
    z_std = float(z.std())

    z_norm = (z - z_mean) / (z_std + 1e-8)
    z_norm_min = float(z_norm.min())
    z_norm_max = float(z_norm.max())

    # Approximate global image mean/std by sampling subset
    img_means = []
    img_stds = []

    sample_df = df.sample(min(max_images, len(df)), random_state=42)

    for name in tqdm(sample_df.iloc[:, 0].values, desc="Computing global image mean/std"):
        img_path = os.path.join(train_img_dir, name)
        img = tifffile.imread(img_path).astype(np.float32)

        if img.ndim == 3:
            img = img.mean(axis=-1)

        img_means.append(img.mean())
        img_stds.append(img.std())

    img_mean = float(np.mean(img_means))
    img_std = float(np.mean(img_stds))

    return z_mean, z_std, z_norm_min, z_norm_max, img_mean, img_std

############################################
# MAIN
############################################

def main():

    train_img_dir = "../experimental_dataset/expo_dataset_v3/train/images"
    val_img_dir   = "../experimental_dataset/expo_dataset_v3/val/images"

    train_csv = "../experimental_dataset/expo_dataset_v3/train/labels.csv"
    val_csv   = "../experimental_dataset/expo_dataset_v3/val/labels.csv"

    z_mean, z_std, z_norm_min, z_norm_max, img_mean, img_std = compute_train_stats(
        train_csv,
        train_img_dir,
        max_images=3000
    )

    with open(NORM_FILE, "w") as f:
        json.dump({
            "z_mean": z_mean,
            "z_std": z_std,
            "z_norm_min": z_norm_min,
            "z_norm_max": z_norm_max,
            "img_mean": img_mean,
            "img_std": img_std
        }, f, indent=4)

    print("\n[NORMALIZATION PARAMETERS]")
    print(f"z_mean     = {z_mean:.6f}")
    print(f"z_std      = {z_std:.6f}")
    print(f"z_norm_min = {z_norm_min:.6f}")
    print(f"z_norm_max = {z_norm_max:.6f}")
    print(f"img_mean   = {img_mean:.6f}")
    print(f"img_std    = {img_std:.6f}")
    print(f"Saved to {NORM_FILE}\n")

    train_dataset = HoloDataset(
        train_img_dir,
        train_csv,
        train=True,
        z_mean=z_mean,
        z_std=z_std,
        z_norm_min=z_norm_min,
        z_norm_max=z_norm_max,
        img_mean=img_mean,
        img_std=img_std
    )

    val_dataset = HoloDataset(
        val_img_dir,
        val_csv,
        train=False,
        z_mean=z_mean,
        z_std=z_std,
        z_norm_min=z_norm_min,
        z_norm_max=z_norm_max,
        img_mean=img_mean,
        img_std=img_std
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = DepthTransformerGaussian(in_ch=6, num_bins=NUM_BINS).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4, verbose=True
    )

    scaler = torch.cuda.amp.GradScaler()

    bin_centers = torch.linspace(z_norm_min, z_norm_max, NUM_BINS).to(device)

    best_mae = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch+1}")

        train_loss = train_epoch(model, train_loader, optimizer, scaler)
        val_mae = validate(model, val_loader, bin_centers, z_mean, z_std)

        scheduler.step(val_mae)

        print(f"Train Loss: {train_loss:.6f} | Val MAE (µm): {val_mae:.4f}")

        if val_mae < best_mae - MIN_DELTA:
            best_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), "best_depth_multiscale_transformer.pth")
            print("✓ Best model saved")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("\nEarly stopping triggered")
                break


if __name__ == "__main__":
    main()
