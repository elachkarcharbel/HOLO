import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import tifffile
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import rotate
import timm
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################################
# CONFIG
############################################

PATCH_SIZE = 384
PATCHES_PER_IMAGE = 4
VAL_PATCHES = 16
BATCH_SIZE = 4
EPOCHS = 100

# 🔥 depth sampling (your request, improved)
DZ_LIST = [0, 10, 20, 40, 60, 80, 100]

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

    def __init__(self,img_dir,csv_file,z_mean=None,z_std=None,train=True):

        self.img_dir = img_dir
        self.df = pd.read_csv(csv_file)

        self.img_names = self.df.iloc[:,0].values
        self.z_values  = self.df.iloc[:,1].values.astype(np.float32)

        self.train = train

        if z_mean is None:
            self.z_mean = self.z_values.mean()
            self.z_std = self.z_values.std()
        else:
            self.z_mean = z_mean
            self.z_std = z_std

        self.z_norm = (self.z_values - self.z_mean)/(self.z_std+1e-6)

        self.mean = 0.5
        self.std = 0.25

    def __len__(self):
        return len(self.img_names)

    def get_patch(self,img):
        h,w = img.shape
        y = np.random.randint(0,h-PATCH_SIZE)
        x = np.random.randint(0,w-PATCH_SIZE)
        return img[y:y+PATCH_SIZE,x:x+PATCH_SIZE]

    def augment_patch(self,patch):

        if np.random.rand()>0.5:
            patch = np.flip(patch,axis=0)

        if np.random.rand()>0.5:
            patch = np.flip(patch,axis=1)

        angle = np.random.uniform(-15,15)
        patch = rotate(patch,angle,reshape=False,order=1,mode='reflect')

        patch = patch * np.random.uniform(0.9,1.1)
        patch = patch + np.random.normal(0,0.01,patch.shape)

        return patch

    def fft_channel(self,patch):
        fft = np.fft.fftshift(np.fft.fft2(patch))
        mag = np.log1p(np.abs(fft))
        return (mag-mag.mean())/(mag.std()+1e-6)

    def propagate(self, patch, dz, wavelength):

        pixel_size = 2.0  # µm

        H, W = patch.shape

        fx = np.fft.fftfreq(W, d=pixel_size)
        fy = np.fft.fftfreq(H, d=pixel_size)

        FX, FY = np.meshgrid(fx, fy)

        phase = np.exp(-1j * np.pi * wavelength * dz * (FX**2 + FY**2))

        F = np.fft.fft2(patch)
        propagated = np.fft.ifft2(F * phase)

        return np.real(propagated)

    def __getitem__(self,idx):

        img_path = os.path.join(self.img_dir,self.img_names[idx])
        img = tifffile.imread(img_path).astype(np.float32)

        if img.ndim==3:
            img = img.mean(axis=-1)

        patches=[]

        num_patches = PATCHES_PER_IMAGE if self.train else VAL_PATCHES

        for _ in range(num_patches):

            patch = self.get_patch(img)

            if self.train:
                patch = self.augment_patch(patch)

            fft_mag = self.fft_channel(patch)

            # wavelengths (µm)
            lambda1 = 0.63
            lambda2 = 0.55

            prop_channels = []

            for dz in DZ_LIST:

                prop1 = self.propagate(patch, dz, lambda1)
                prop2 = self.propagate(patch, dz, lambda2)

                synth = prop1 - prop2

                prop1 = (prop1 - prop1.mean())/(prop1.std()+1e-6)
                prop2 = (prop2 - prop2.mean())/(prop2.std()+1e-6)
                synth = (synth - synth.mean())/(synth.std()+1e-6)

                prop_channels.extend([prop1, prop2, synth])

            patch = (patch-self.mean)/self.std

            stacked = np.stack([patch, fft_mag] + prop_channels)

            patches.append(torch.from_numpy(stacked).float())

        patches = torch.stack(patches)

        z = torch.tensor(self.z_norm[idx]).float()

        return patches,z


############################################
# CNN
############################################

class CNNBranch(nn.Module):

    def __init__(self,in_ch):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_ch,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self,x):
        return self.net(x).view(x.size(0),-1)


############################################
# SWIN
############################################

class SwinBranch(nn.Module):

    def __init__(self,in_ch):
        super().__init__()

        self.backbone = timm.create_model(
            "swin_base_patch4_window12_384",
            pretrained=True,
            num_classes=0
        )

        old = self.backbone.patch_embed.proj

        self.backbone.patch_embed.proj = nn.Conv2d(
            in_ch,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding
        )

        self.out_dim = self.backbone.num_features

    def forward(self,x):
        return self.backbone(x)


############################################
# TRANSFORMER POOLING
############################################

class PatchTransformer(nn.Module):

    def __init__(self, dim, num_heads=8, depth=2):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 64, dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim*4,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):

        B,P,D = x.shape

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        x = x + self.pos_embed[:, :P+1, :]

        x = self.transformer(x)

        return x[:,0]


############################################
# MODEL
############################################

class HoloHybridNet(nn.Module):

    def __init__(self,in_ch):
        super().__init__()

        self.cnn = CNNBranch(in_ch)
        self.swin = SwinBranch(in_ch)

        feat_dim = 128 + self.swin.out_dim

        self.pool = PatchTransformer(feat_dim)

        self.regressor = nn.Sequential(
            nn.Linear(feat_dim,512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512,128),
            nn.GELU(),
            nn.Linear(128,1)
        )

    def forward(self,x):

        B,P,C,H,W = x.shape

        x = x.view(B*P,C,H,W)

        cnn_feat = self.cnn(x)
        swin_feat = self.swin(x)

        feat = torch.cat([cnn_feat,swin_feat],dim=1)
        feat = feat.view(B,P,-1)

        pooled = self.pool(feat)

        return self.regressor(pooled).squeeze()


############################################
# TRAIN
############################################

def train_epoch(model,loader,optimizer,criterion,scaler):

    model.train()
    total_loss=0

    pbar = tqdm(loader, desc="Training", leave=False)

    for imgs,z in pbar:

        imgs = imgs.to(device)
        z = z.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            pred = model(imgs)
            loss = criterion(pred,z)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    return total_loss/len(loader)


############################################
# VALIDATION
############################################

def validate(model,loader,criterion,z_mean,z_std):

    model.eval()
    total_loss=0
    mae=0

    pbar = tqdm(loader, desc="Validation", leave=False)

    with torch.no_grad():
        for imgs,z in pbar:

            imgs = imgs.to(device)
            z = z.to(device)

            pred = model(imgs)
            loss = criterion(pred,z)

            total_loss += loss.item()

            pred_um = pred*z_std + z_mean
            z_um = z*z_std + z_mean

            batch_mae = torch.mean(torch.abs(pred_um-z_um)).item()
            mae += batch_mae

            pbar.set_postfix(mae=batch_mae)

    return total_loss/len(loader), mae/len(loader)


############################################
# MAIN
############################################

def main():

    train_dataset = HoloDataset(TRAIN_IMG_DIR,TRAIN_CSV,train=True)

    z_mean = train_dataset.z_mean
    z_std = train_dataset.z_std

    val_dataset = HoloDataset(VAL_IMG_DIR,VAL_CSV,z_mean,z_std,train=False)

    train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4,pin_memory=True)
    val_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=4,pin_memory=True)

    # 🔥 compute channels automatically
    in_ch = 2 + 3*len(DZ_LIST)

    model = HoloHybridNet(in_ch).to(device)

    optimizer = optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=3)

    scaler = torch.cuda.amp.GradScaler()

    best_mae=float("inf")

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch+1}")

        train_loss = train_epoch(model,train_loader,optimizer,criterion,scaler)
        val_loss,val_mae = validate(model,val_loader,criterion,z_mean,z_std)

        print(f"Train Loss: {train_loss:.6f} | Val MAE (µm): {val_mae:.4f}")

        scheduler.step(val_mae)

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(),"best_physics-cnn-swin.pth")
            print("✓ Best model saved")


if __name__ == "__main__":
    main()
