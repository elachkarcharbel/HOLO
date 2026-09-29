import os
import time
import numpy as np
import pandas as pd
import tifffile

import torch
import torch.nn as nn
import timm

from scipy.ndimage import rotate
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

PATCH_SIZE = 384
TEST_PATCHES = 16

DZ_LIST = [0, 10, 20, 40, 60, 80, 100]

TEST_IMG_DIR = "../experimental_dataset/expo_dataset_v3/test/images"
TEST_CSV = "../experimental_dataset/expo_dataset_v3/test/labels.csv"

TRAIN_CSV = "../experimental_dataset/expo_dataset_v3/train/labels.csv"

CHECKPOINT = "best_physics-cnn-swin.pth"

OUTPUT_CSV = "test_predictions_physics_cnn_swin.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# ============================================================
# DATASET / PREPROCESSING
# ============================================================

class PhysicsTestDataset:

    def __init__(self, img_dir, csv_file, z_mean, z_std):
        self.img_dir = img_dir
        self.df = pd.read_csv(csv_file)

        self.img_names = self.df.iloc[:, 0].values
        self.z_values = self.df.iloc[:, 1].values.astype(np.float32)

        self.z_mean = z_mean
        self.z_std = z_std

        self.mean = 0.5
        self.std = 0.25

    def __len__(self):
        return len(self.img_names)

    def get_patch(self, img):

        h, w = img.shape

        y = np.random.randint(0, h - PATCH_SIZE)
        x = np.random.randint(0, w - PATCH_SIZE)

        return img[
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE
        ]

    def fft_channel(self, patch):

        fft = np.fft.fftshift(
            np.fft.fft2(patch)
        )

        mag = np.log1p(
            np.abs(fft)
        )

        return (
            mag - mag.mean()
        ) / (
            mag.std() + 1e-6
        )

    def propagate(self, patch, dz, wavelength):

        # Same implementation as training
        pixel_size = 2.0  # micrometers

        H, W = patch.shape

        fx = np.fft.fftfreq(
            W,
            d=pixel_size
        )

        fy = np.fft.fftfreq(
            H,
            d=pixel_size
        )

        FX, FY = np.meshgrid(fx, fy)

        phase = np.exp(
            -1j
            * np.pi
            * wavelength
            * dz
            * (FX ** 2 + FY ** 2)
        )

        F = np.fft.fft2(patch)

        propagated = np.fft.ifft2(
            F * phase
        )

        return np.real(propagated)

    def make_patch(self, patch):

        # ----------------------------------------------------
        # FFT channel
        # ----------------------------------------------------

        fft_mag = self.fft_channel(patch)


        # ----------------------------------------------------
        # Physics propagation channels
        # ----------------------------------------------------

        lambda1 = 0.63  # micrometers
        lambda2 = 0.55  # micrometers

        prop_channels = []

        for dz in DZ_LIST:

            prop1 = self.propagate(
                patch,
                dz,
                lambda1
            )

            prop2 = self.propagate(
                patch,
                dz,
                lambda2
            )

            # Difference / beat representation
            synth = prop1 - prop2


            # Local normalization
            prop1 = (
                prop1 - prop1.mean()
            ) / (
                prop1.std() + 1e-6
            )

            prop2 = (
                prop2 - prop2.mean()
            ) / (
                prop2.std() + 1e-6
            )

            synth = (
                synth - synth.mean()
            ) / (
                synth.std() + 1e-6
            )

            prop_channels.extend([
                prop1,
                prop2,
                synth
            ])


        # ----------------------------------------------------
        # Raw hologram normalization
        # ----------------------------------------------------

        patch = (
            patch - self.mean
        ) / self.std


        # ----------------------------------------------------
        # Final 23-channel representation
        #
        # 1 raw
        # 1 FFT
        # 7 * 3 physics channels
        #
        # = 23 channels
        # ----------------------------------------------------

        stacked = np.stack(
            [patch, fft_mag] + prop_channels
        )

        return torch.from_numpy(
            stacked
        ).float()


    def get_patches(self, idx):

        img_path = os.path.join(
            self.img_dir,
            self.img_names[idx]
        )

        img = tifffile.imread(
            img_path
        ).astype(np.float32)

        if img.ndim == 3:
            img = img.mean(axis=-1)

        patches = []

        for _ in range(TEST_PATCHES):

            # No augmentation at test time
            patch = self.get_patch(img)

            patch = self.make_patch(patch)

            patches.append(patch)

        return torch.stack(patches), self.z_values[idx]


# ============================================================
# CNN BRANCH
# ============================================================

class CNNBranch(nn.Module):

    def __init__(self, in_ch):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(
                in_ch,
                32,
                3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),


            nn.Conv2d(
                32,
                64,
                3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(2),


            nn.Conv2d(
                64,
                128,
                3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):

        return self.net(x).view(
            x.size(0),
            -1
        )


# ============================================================
# SWIN BRANCH
# ============================================================

class SwinBranch(nn.Module):

    def __init__(self, in_ch):

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

    def forward(self, x):

        return self.backbone(x)


# ============================================================
# PATCH TRANSFORMER
# ============================================================

class PatchTransformer(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=8,
        depth=2
    ):

        super().__init__()

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, dim)
        )

        self.pos_embed = nn.Parameter(
            torch.randn(1, 64, dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

    def forward(self, x):

        B, P, D = x.shape

        cls = self.cls_token.expand(
            B,
            -1,
            -1
        )

        x = torch.cat(
            [cls, x],
            dim=1
        )

        x = x + self.pos_embed[:, :P + 1, :]

        x = self.transformer(x)

        return x[:, 0]


# ============================================================
# COMPLETE MODEL
# ============================================================

class HoloHybridNet(nn.Module):

    def __init__(self, in_ch):

        super().__init__()

        self.cnn = CNNBranch(in_ch)

        self.swin = SwinBranch(in_ch)

        feat_dim = (
            128
            + self.swin.out_dim
        )

        self.pool = PatchTransformer(
            feat_dim
        )

        self.regressor = nn.Sequential(

            nn.Linear(
                feat_dim,
                512
            ),

            nn.GELU(),

            nn.Dropout(0.3),

            nn.Linear(
                512,
                128
            ),

            nn.GELU(),

            nn.Linear(
                128,
                1
            )
        )

    def forward(self, x):

        B, P, C, H, W = x.shape

        x = x.view(
            B * P,
            C,
            H,
            W
        )

        cnn_feat = self.cnn(x)

        swin_feat = self.swin(x)

        feat = torch.cat(
            [
                cnn_feat,
                swin_feat
            ],
            dim=1
        )

        feat = feat.view(
            B,
            P,
            -1
        )

        pooled = self.pool(feat)

        return self.regressor(
            pooled
        ).squeeze()


# ============================================================
# METRICS
# ============================================================

def compute_metrics(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    errors = y_pred - y_true

    mae = np.mean(
        np.abs(errors)
    )

    rmse = np.sqrt(
        np.mean(errors ** 2)
    )

    ss_res = np.sum(
        (y_true - y_pred) ** 2
    )

    ss_tot = np.sum(
        (y_true - y_true.mean()) ** 2
    )

    r2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0
        else np.nan
    )

    median_ae = np.median(
        np.abs(errors)
    )

    bias = np.mean(errors)

    return {
        "MAE_um": mae,
        "RMSE_um": rmse,
        "R2": r2,
        "MedianAE_um": median_ae,
        "Bias_um": bias
    }


# ============================================================
# MAIN INFERENCE
# ============================================================

def main():

    # --------------------------------------------------------
    # Training Z statistics
    # --------------------------------------------------------

    train_df = pd.read_csv(
        TRAIN_CSV
    )

    train_z = train_df.iloc[
        :, 1
    ].values.astype(
        np.float32
    )

    z_mean = train_z.mean()
    z_std = train_z.std()

    print()
    print("Training Z statistics")
    print("---------------------")
    print(f"Z mean : {z_mean:.6f} µm")
    print(f"Z std  : {z_std:.6f} µm")


    # --------------------------------------------------------
    # Test dataset
    # --------------------------------------------------------

    test_dataset = PhysicsTestDataset(
        TEST_IMG_DIR,
        TEST_CSV,
        z_mean,
        z_std
    )

    print()
    print(
        "Number of test images:",
        len(test_dataset)
    )

    print(
        "Patches per image:",
        TEST_PATCHES
    )

    print(
        "Channels:",
        2 + 3 * len(DZ_LIST)
    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    IN_CH = 2 + 3 * len(DZ_LIST)

    model = HoloHybridNet(
        IN_CH
    ).to(device)


    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device
    )

    # Training code saves:
    # torch.save(model.state_dict(), ...)
    model.load_state_dict(
        checkpoint
    )

    model.eval()

    print()
    print(
        f"Loaded checkpoint: {CHECKPOINT}"
    )


    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    results = []

    y_true = []
    y_pred = []

    total_inference_time = 0.0


    with torch.no_grad():

        for idx in tqdm(
            range(len(test_dataset)),
            desc="Testing"
        ):

            patches, z_gt = (
                test_dataset.get_patches(idx)
            )

            # Add batch dimension
            patches = patches.unsqueeze(0)

            patches = patches.to(
                device,
                non_blocking=True
            )


            # ------------------------------------------------
            # Accurate CUDA timing
            # ------------------------------------------------

            if device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            pred_norm = model(
                patches
            )

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = (
                time.perf_counter()
                - start_time
            )


            # ------------------------------------------------
            # Convert normalized prediction to µm
            # ------------------------------------------------

            pred_norm = float(
                pred_norm.item()
            )

            z_pred = (
                pred_norm * z_std
                + z_mean
            )

            z_gt = float(z_gt)

            abs_error = abs(
                z_pred - z_gt
            )

            signed_error = (
                z_pred - z_gt
            )


            # ------------------------------------------------
            # Store
            # ------------------------------------------------

            image_name = (
                test_dataset.img_names[idx]
            )

            results.append({
                "image": image_name,
                "z_GT_um": z_gt,
                "z_pred_um": z_pred,
                "abs_error_um": abs_error,
                "signed_error_um": signed_error,
                "inference_time_s": elapsed
            })

            y_true.append(z_gt)
            y_pred.append(z_pred)

            total_inference_time += elapsed


    # ========================================================
    # METRICS
    # ========================================================

    metrics = compute_metrics(
        y_true,
        y_pred
    )

    mean_time = (
        total_inference_time
        / len(test_dataset)
    )

    total_time = total_inference_time


    # ========================================================
    # SAVE CSV
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        OUTPUT_CSV,
        index=False
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print("PHYSICS-AWARE CNN + SWIN TEST RESULTS")
    print("=" * 60)

    print(
        f"Test MAE              : "
        f"{metrics['MAE_um']:.4f} µm"
    )

    print(
        f"Test RMSE             : "
        f"{metrics['RMSE_um']:.4f} µm"
    )

    print(
        f"Test R²               : "
        f"{metrics['R2']:.6f}"
    )

    print(
        f"Test Median AE        : "
        f"{metrics['MedianAE_um']:.4f} µm"
    )

    print(
        f"Test Bias             : "
        f"{metrics['Bias_um']:.4f} µm"
    )

    print(
        f"Mean inference time   : "
        f"{mean_time:.6f} s/image"
    )

    print(
        f"Total inference time  : "
        f"{total_time:.3f} s"
    )

    print(
        f"Mean inference speed  : "
        f"{1.0 / mean_time:.2f} images/s"
    )

    print()
    print(
        f"Predictions saved to: "
        f"{OUTPUT_CSV}"
    )

    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()