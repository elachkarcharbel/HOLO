import os
import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ============================================
# CONFIG
# ============================================

TRAIN_IMG_DIR = "../experimental_dataset/expo_dataset_v3/train/images"
TRAIN_CSV     = "../experimental_dataset/expo_dataset_v3/train/labels.csv"

VAL_IMG_DIR   = "../experimental_dataset/expo_dataset_v3/val/images"
VAL_CSV       = "../experimental_dataset/expo_dataset_v3/val/labels.csv"

IMG_SIZE = 128
BATCH_SIZE = 128
EPOCHS = 400
LR = 5e-4                  # initial LR
EARLY_STOPPING_PATIENCE = 50  # allow longer training

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================
# DATASET
# ============================================

class HoloDataset(Dataset):
    def __init__(self, img_dir, csv_path):
        self.img_dir = img_dir
        self.df = pd.read_csv(csv_path)
        self.image_paths = self.df["image"].values
        self.z = self.df["Z input [um]"].values.astype(np.float32)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.image_paths[idx])
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        return (
            torch.tensor(img[None, :, :], dtype=torch.float32),
            torch.tensor(self.z[idx], dtype=torch.float32)
        )

# ============================================
# CNN MODEL
# ============================================

class ZPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        z = self.regressor(x)
        return z.squeeze(1)  # output in µm directly

# ============================================
# TRAINING
# ============================================

def main():
    print("Loading dataset...")

    train_dataset = HoloDataset(TRAIN_IMG_DIR, TRAIN_CSV)
    val_dataset   = HoloDataset(VAL_IMG_DIR, VAL_CSV)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=8, pin_memory=True
    )

    model = ZPredictor().to(DEVICE)
    criterion = nn.SmoothL1Loss(beta=0.01)  # more stable for small µm differences
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = GradScaler()

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_mae = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(EPOCHS):

        # ================ TRAIN =================
        model.train()
        train_loss = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            with autocast():
                z_pred = model(imgs)
                # optionally clamp to reasonable range
                z_pred = torch.clamp(z_pred, 0.0, 100.0)
                loss = criterion(z_pred, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ================ VALIDATE =================
        model.eval()
        val_mae = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)
                z_pred = model(imgs)
                z_pred = torch.clamp(z_pred, 0.0, 100.0)
                mae = torch.mean(torch.abs(z_pred - labels))
                val_mae += mae.item()
        val_mae /= len(val_loader)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val MAE (µm): {val_mae:.4f}")

        # ================ SAVE BEST =================
        if val_mae < best_mae:
            best_mae = val_mae
            patience_counter = 0
            torch.save({"model_state_dict": model.state_dict()}, "best_baseline_cnn_final.pth")
            print("✓ Best model saved")
        else:
            patience_counter += 1

        # ================ LR SCHEDULER =================
        scheduler.step(val_mae)

        # ================ EARLY STOP =================
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("\nEarly stopping triggered.")
            print(f"Best MAE: {best_mae:.4f} µm")
            break

    print("Training finished.")

# ============================================

if __name__ == "__main__":
    main()
