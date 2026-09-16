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
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################################
# CONFIG
############################################

PATCH_SIZE = 384
PATCHES_PER_IMAGE = 6
VAL_PATCHES = 32   # 👈 more stable validation
BATCH_SIZE = 8
EPOCHS = 100

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
        mag = (mag-mag.mean())/(mag.std()+1e-6)

        return mag

    def focus_channel(self, patch):

        patch = patch.astype(np.float32)
        lap = cv2.Laplacian(patch, cv2.CV_32F)
        lap = (lap - lap.mean()) / (lap.std() + 1e-6)

        return lap

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
            focus = self.focus_channel(patch)

            patch = (patch-self.mean)/self.std

            stacked = np.stack([patch, fft_mag, focus])

            patches.append(torch.from_numpy(stacked).float())

        patches = torch.stack(patches)

        z = torch.tensor(self.z_norm[idx]).float()

        return patches,z


############################################
# CNN BRANCH
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

        x = self.net(x)

        return x.view(x.size(0),-1)


############################################
# SWIN BRANCH
############################################

class SwinBranch(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = timm.create_model(
            "swin_base_patch4_window12_384",
            pretrained=True,
            num_classes=0
        )

        old = self.backbone.patch_embed.proj

        self.backbone.patch_embed.proj = nn.Conv2d(
            3,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding
        )

        self.out_dim = self.backbone.num_features

    def forward(self,x):

        return self.backbone(x)


############################################
# PATCH ATTENTION
############################################

class PatchAttention(nn.Module):

    def __init__(self,dim):

        super().__init__()

        self.attn = nn.Sequential(
            nn.Linear(dim,128),
            nn.GELU(),
            nn.Linear(128,1)
        )

    def forward(self,x):

        w = self.attn(x)
        w = torch.softmax(w,dim=1)

        return (x*w).sum(dim=1)


############################################
# HYBRID MODEL
############################################

class HoloHybridNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.cnn = CNNBranch(3)
        self.swin = SwinBranch()

        feat_dim = 128 + self.swin.out_dim

        self.patch_attention = PatchAttention(feat_dim)

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

        pooled = self.patch_attention(feat)

        z = self.regressor(pooled)

        return z.squeeze()


############################################
# TRAIN
############################################

def train_epoch(model,loader,optimizer,criterion,scaler):

    model.train()

    total_loss=0

    for imgs,z in loader:

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

    return total_loss/len(loader)


############################################
# VALIDATION
############################################

def validate(model,loader,criterion,z_mean,z_std):

    model.eval()

    total_loss=0
    mae=0

    with torch.no_grad():

        for imgs,z in loader:

            imgs = imgs.to(device)
            z = z.to(device)

            pred = model(imgs)

            loss = criterion(pred,z)

            total_loss += loss.item()

            pred_um = pred*z_std + z_mean
            z_um = z*z_std + z_mean

            mae += torch.mean(torch.abs(pred_um-z_um)).item()

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

    model = HoloHybridNet().to(device)

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

            torch.save(model.state_dict(),"best_swin-cnn-final.pth")

            print("✓ Best model saved")

if __name__ == "__main__":
    main()
