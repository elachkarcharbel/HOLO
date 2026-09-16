import os
import time

import torch
import torch.nn as nn
import pandas as pd
import tifffile
import numpy as np

from scipy.ndimage import rotate
import timm
import cv2

from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

PATCH_SIZE = 384
PATCHES_PER_IMAGE = 32

IMG_MEAN = 0.5
IMG_STD = 0.25

MODEL_PATH = "best_swin-cnn-final.pth"

TEST_IMG_DIR = "../experimental_dataset/expo_dataset_v3/test/images"
TEST_CSV = "../experimental_dataset/expo_dataset_v3/test/labels.csv"

OUTPUT_CSV = "test_predictions_swin_cnn_hybrid.csv"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DATASET / PREPROCESSING
# ============================================================

class HoloTestDataset:

    def __init__(
        self,
        img_dir,
        csv_file,
        z_mean,
        z_std
    ):

        self.img_dir = img_dir

        self.df = pd.read_csv(csv_file)

        self.img_names = self.df.iloc[:, 0].values

        self.z_values = (
            self.df.iloc[:, 1]
            .values
            .astype(np.float32)
        )

        self.z_mean = float(z_mean)
        self.z_std = float(z_std)

        self.mean = IMG_MEAN
        self.std = IMG_STD

    def __len__(self):
        return len(self.img_names)

    # --------------------------------------------------------
    # Same patch extraction as training
    # --------------------------------------------------------

    def get_patch(self, img):

        h, w = img.shape

        y = np.random.randint(
            0,
            h - PATCH_SIZE
        )

        x = np.random.randint(
            0,
            w - PATCH_SIZE
        )

        return img[
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE
        ]

    # --------------------------------------------------------
    # Same FFT channel as training
    # --------------------------------------------------------

    def fft_channel(self, patch):

        fft = np.fft.fftshift(
            np.fft.fft2(patch)
        )

        mag = np.log1p(
            np.abs(fft)
        )

        mag = (
            mag - mag.mean()
        ) / (
            mag.std() + 1e-6
        )

        return mag

    # --------------------------------------------------------
    # Same focus/Laplacian channel as training
    # --------------------------------------------------------

    def focus_channel(self, patch):

        patch = patch.astype(
            np.float32
        )

        lap = cv2.Laplacian(
            patch,
            cv2.CV_32F
        )

        lap = (
            lap - lap.mean()
        ) / (
            lap.std() + 1e-6
        )

        return lap

    # --------------------------------------------------------
    # Construct one 3-channel patch
    # --------------------------------------------------------

    def make_patch(self, patch):

        fft_mag = self.fft_channel(
            patch
        )

        focus = self.focus_channel(
            patch
        )

        # Same raw-patch normalization
        patch = (
            patch - self.mean
        ) / self.std

        stacked = np.stack([
            patch,
            fft_mag,
            focus
        ])

        return torch.from_numpy(
            stacked
        ).float()

    # --------------------------------------------------------
    # Generate 32 patches for one image
    # --------------------------------------------------------

    def get_image_patches(self, idx):

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

        for _ in range(PATCHES_PER_IMAGE):

            patch = self.get_patch(img)

            patch = self.make_patch(
                patch
            )

            patches.append(patch)

        return torch.stack(patches)


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

        x = self.net(x)

        return x.view(
            x.size(0),
            -1
        )


# ============================================================
# SWIN BRANCH
# ============================================================

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

        self.out_dim = (
            self.backbone.num_features
        )

    def forward(self, x):

        return self.backbone(x)


# ============================================================
# PATCH ATTENTION
# ============================================================

class PatchAttention(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.attn = nn.Sequential(

            nn.Linear(
                dim,
                128
            ),

            nn.GELU(),

            nn.Linear(
                128,
                1
            )
        )

    def forward(self, x):

        w = self.attn(x)

        w = torch.softmax(
            w,
            dim=1
        )

        return (
            x * w
        ).sum(dim=1)


# ============================================================
# HYBRID MODEL
# ============================================================

class HoloHybridNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.cnn = CNNBranch(3)

        self.swin = SwinBranch()

        feat_dim = (
            128
            + self.swin.out_dim
        )

        self.patch_attention = (
            PatchAttention(feat_dim)
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

        pooled = self.patch_attention(
            feat
        )

        z = self.regressor(
            pooled
        )

        return z.squeeze()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CNN + SWIN HYBRID TEST INFERENCE")
    print("=" * 70)

    print(f"Device              : {DEVICE}")
    print(f"Model               : {MODEL_PATH}")
    print(f"Patch size          : {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"Patches/image       : {PATCHES_PER_IMAGE}")
    print()

    # --------------------------------------------------------
    # Compute Z normalization from TRAINING SET
    # --------------------------------------------------------

    TRAIN_CSV = (
        "../experimental_dataset/"
        "expo_dataset_v3/train/labels.csv"
    )

    train_df = pd.read_csv(
        TRAIN_CSV
    )

    train_z = (
        train_df.iloc[:, 1]
        .values
        .astype(np.float32)
    )

    z_mean = float(
        train_z.mean()
    )

    z_std = float(
        train_z.std()
    )

    print("[TRAINING Z NORMALIZATION]")
    print(f"z_mean = {z_mean:.8f}")
    print(f"z_std  = {z_std:.8f}")
    print()

    # --------------------------------------------------------
    # Test dataset
    # --------------------------------------------------------

    dataset = HoloTestDataset(
        TEST_IMG_DIR,
        TEST_CSV,
        z_mean,
        z_std
    )

    print(
        f"Number of test images: "
        f"{len(dataset)}"
    )
    print()

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = HoloHybridNet().to(
        DEVICE
    )

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            f"Model checkpoint not found: "
            f"{MODEL_PATH}"
        )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # This model was saved directly with:
    #
    # torch.save(
    #     model.state_dict(),
    #     "best_hybrid_model.pth"
    # )
    #
    # Therefore load it directly.

    model.load_state_dict(
        state_dict
    )

    model.eval()

    print(
        "✓ Model loaded successfully"
    )
    print()

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    results = []

    all_gt = []
    all_pred = []

    total_inference_time = 0.0

    print("Running inference...")

    for idx in tqdm(
        range(len(dataset)),
        desc="Test"
    ):

        image_name = (
            dataset.img_names[idx]
        )

        z_gt = float(
            dataset.z_values[idx]
        )

        # ----------------------------------------------------
        # Generate 32 patches
        # ----------------------------------------------------

        patches = (
            dataset.get_image_patches(idx)
        )

        # [P,C,H,W]
        # →
        # [1,P,C,H,W]

        patches = patches.unsqueeze(
            0
        ).to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Accurate GPU timing
        # ----------------------------------------------------

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.no_grad():

            # Model outputs normalized Z
            z_pred_norm = model(
                patches
            )

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        elapsed = (
            time.perf_counter()
            - start_time
        )

        # ----------------------------------------------------
        # Convert normalized Z -> µm
        # ----------------------------------------------------

        z_pred = float(
            z_pred_norm.item()
            * z_std
            + z_mean
        )

        # ----------------------------------------------------
        # Errors
        # ----------------------------------------------------

        signed_error = (
            z_pred - z_gt
        )

        abs_error = abs(
            signed_error
        )

        all_gt.append(z_gt)
        all_pred.append(z_pred)

        total_inference_time += elapsed

        results.append({

            "image": image_name,

            "z_GT_um": z_gt,

            "z_pred_um": z_pred,

            "abs_error_um": abs_error,

            "signed_error_um": signed_error,

            "inference_time_s": elapsed
        })

    # ========================================================
    # METRICS
    # ========================================================

    all_gt = np.asarray(
        all_gt,
        dtype=np.float64
    )

    all_pred = np.asarray(
        all_pred,
        dtype=np.float64
    )

    errors = (
        all_pred - all_gt
    )

    abs_errors = np.abs(
        errors
    )

    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    mae = np.mean(
        abs_errors
    )

    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    rmse = np.sqrt(
        np.mean(
            errors ** 2
        )
    )

    # --------------------------------------------------------
    # R²
    # --------------------------------------------------------

    ss_res = np.sum(
        (all_gt - all_pred) ** 2
    )

    ss_tot = np.sum(
        (all_gt - np.mean(all_gt)) ** 2
    )

    r2 = 1.0 - (
        ss_res / ss_tot
    )

    # --------------------------------------------------------
    # Median Absolute Error
    # --------------------------------------------------------

    median_ae = np.median(
        abs_errors
    )

    # --------------------------------------------------------
    # Bias
    # --------------------------------------------------------

    bias = np.mean(
        errors
    )

    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------

    mean_time = np.mean([
        r["inference_time_s"]
        for r in results
    ])

    total_time = (
        total_inference_time
    )

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
    print("=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    print(
        f"Test MAE                  : "
        f"{mae:.6f} µm"
    )

    print(
        f"Test RMSE                 : "
        f"{rmse:.6f} µm"
    )

    print(
        f"Test R²                   : "
        f"{r2:.6f}"
    )

    print(
        f"Test Median Absolute Error: "
        f"{median_ae:.6f} µm"
    )

    print(
        f"Test Bias                 : "
        f"{bias:.6f} µm"
    )

    print()

    print(
        f"Mean inference time/image : "
        f"{mean_time:.6f} s"
    )

    print(
        f"Total inference time      : "
        f"{total_time:.2f} s"
    )

    print()

    print(
        f"Saved predictions to      : "
        f"{OUTPUT_CSV}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()