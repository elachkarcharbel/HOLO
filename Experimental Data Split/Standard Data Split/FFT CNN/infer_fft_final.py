import os
import time

import torch
import torch.nn as nn
import pandas as pd
import tifffile
import numpy as np


# ============================================================
# CONFIG
# ============================================================

PATCH_SIZE = 256

# Training used 8 patches/image.
# For inference we can use more patches for a more stable
# image-level prediction.
PATCHES_PER_IMAGE = 64

MODEL_PATH = "best_model_fft_final.pth"

TRAIN_IMG_DIR = "../experimental_dataset/expo_dataset_v3/train/images"
TRAIN_CSV = "../experimental_dataset/expo_dataset_v3/train/labels.csv"

TEST_IMG_DIR = "../experimental_dataset/expo_dataset_v3/test/images"
TEST_CSV = "../experimental_dataset/expo_dataset_v3/test/labels.csv"

# Save image-level GT/predictions for later plots
PREDICTIONS_CSV = "test_predictions_fft_final.csv"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# RESIDUAL BLOCK
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):

        identity = x

        out = self.relu(
            self.bn1(
                self.conv1(x)
            )
        )

        out = self.bn2(
            self.conv2(out)
        )

        out += identity

        out = self.relu(out)

        return out


# ============================================================
# SPATIAL BRANCH
# ============================================================

class SpatialBranch(nn.Module):

    def __init__(self):
        super().__init__()

        self.initial = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.res_blocks = nn.Sequential(

            ResidualBlock(32),
            ResidualBlock(32)
        )

        self.downsample = nn.Sequential(

            nn.Conv2d(
                32,
                64,
                3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.res_blocks2 = nn.Sequential(

            ResidualBlock(64),
            ResidualBlock(64)
        )

        self.final = nn.Sequential(

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

        x = self.initial(x)

        x = self.res_blocks(x)

        x = self.downsample(x)

        x = self.res_blocks2(x)

        x = self.final(x)

        return x.view(
            x.size(0),
            -1
        )


# ============================================================
# FFT BRANCH
# ============================================================

class FFTBranch(nn.Module):

    def __init__(self):
        super().__init__()

        self.initial = nn.Sequential(

            nn.Conv2d(
                2,
                32,
                3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.res_blocks = nn.Sequential(

            ResidualBlock(32),
            ResidualBlock(32)
        )

        self.downsample = nn.Sequential(

            nn.Conv2d(
                32,
                64,
                3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.res_blocks2 = nn.Sequential(

            ResidualBlock(64),
            ResidualBlock(64)
        )

        self.final = nn.Sequential(

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

        fft = torch.fft.fft2(x)

        fft = torch.fft.fftshift(fft)

        mag = torch.log1p(
            torch.abs(fft)
        )

        phase = torch.angle(fft)

        # Per-image/patch FFT normalization.
        # This MUST remain exactly as in training.
        mag = (
            mag
            - mag.mean(
                dim=(-2, -1),
                keepdim=True
            )
        ) / (
            mag.std(
                dim=(-2, -1),
                keepdim=True
            )
            + 1e-6
        )

        phase = (
            phase
            - phase.mean(
                dim=(-2, -1),
                keepdim=True
            )
        ) / (
            phase.std(
                dim=(-2, -1),
                keepdim=True
            )
            + 1e-6
        )

        fft_input = torch.cat(
            [mag, phase],
            dim=1
        )

        x = self.initial(fft_input)

        x = self.res_blocks(x)

        x = self.downsample(x)

        x = self.res_blocks2(x)

        x = self.final(x)

        return x.view(
            x.size(0),
            -1
        )


# ============================================================
# HOLONET
# ============================================================

class HoloNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.spatial = SpatialBranch()

        self.fft = FFTBranch()

        self.fc = nn.Sequential(

            nn.Linear(
                256,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(
                128,
                1
            )
        )

    def forward(self, x):

        s = self.spatial(x)

        f = self.fft(x)

        feat = torch.cat(
            [s, f],
            dim=1
        )

        z = self.fc(feat)

        return z.squeeze()


# ============================================================
# PATCH SAMPLING
# ============================================================

def sample_patch(img):

    h, w = img.shape

    if h < PATCH_SIZE or w < PATCH_SIZE:

        raise ValueError(
            f"Image size {img.shape} is smaller than "
            f"PATCH_SIZE={PATCH_SIZE}"
        )

    # +1 because np.random.randint upper bound
    # is exclusive.
    y = np.random.randint(
        0,
        h - PATCH_SIZE + 1
    )

    x = np.random.randint(
        0,
        w - PATCH_SIZE + 1
    )

    return img[
        y:y + PATCH_SIZE,
        x:x + PATCH_SIZE
    ]


# ============================================================
# COMPUTE TRAINING Z NORMALIZATION
# ============================================================

def get_training_z_stats():

    train_df = pd.read_csv(
        TRAIN_CSV
    )

    train_z = train_df.iloc[
        :, 1
    ].values.astype(
        np.float64
    )

    z_mean = train_z.mean()

    z_std = train_z.std()

    return z_mean, z_std


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    y_true = np.asarray(
        y_true,
        dtype=np.float64
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64
    )

    errors = y_pred - y_true

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
    # Median Absolute Error
    # --------------------------------------------------------

    median_ae = np.median(
        abs_errors
    )

    # --------------------------------------------------------
    # R²
    # --------------------------------------------------------

    ss_res = np.sum(
        errors ** 2
    )

    ss_tot = np.sum(
        (
            y_true
            - np.mean(y_true)
        ) ** 2
    )

    if ss_tot > 0:

        r2 = 1.0 - (
            ss_res / ss_tot
        )

    else:

        r2 = float("nan")

    # --------------------------------------------------------
    # Bias
    # --------------------------------------------------------

    bias = np.mean(
        errors
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MedianAE": median_ae,
        "Bias": bias
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 75)
    print("HoloNet TEST INFERENCE")
    print("=" * 75)

    print(
        f"Device:              {device}"
    )

    print(
        f"Model:               {MODEL_PATH}"
    )

    print(
        f"Patch size:          {PATCH_SIZE}"
    )

    print(
        f"Patches/image:      {PATCHES_PER_IMAGE}"
    )

    # ========================================================
    # TRAINING Z NORMALIZATION
    # ========================================================

    z_mean, z_std = get_training_z_stats()

    print()
    print(
        "Training Z normalization:"
    )

    print(
        f"  Z mean = {z_mean:.8f}"
    )

    print(
        f"  Z std  = {z_std:.8f}"
    )

    # ========================================================
    # TEST DATA
    # ========================================================

    test_df = pd.read_csv(
        TEST_CSV
    )

    images = test_df.iloc[
        :, 0
    ].values

    z_gt = test_df.iloc[
        :, 1
    ].values.astype(
        np.float64
    )

    print()
    print(
        f"Number of test images: {len(images)}"
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = HoloNet().to(device)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    print(
        "Model loaded successfully."
    )

    print()
    print(
        "Image | Z_GT | Z_pred | Abs Error | Time(s)"
    )

    print("-" * 80)

    # ========================================================
    # STORAGE
    # ========================================================

    image_names = []

    ground_truths = []

    predictions = []

    inference_times = []

    # ========================================================
    # INFERENCE
    # ========================================================

    for i, img_name in enumerate(images):

        img_path = os.path.join(
            TEST_IMG_DIR,
            img_name
        )

        img = tifffile.imread(
            img_path
        ).astype(
            np.float32
        )

        # Convert multi-channel TIFF to grayscale
        if img.ndim == 3:

            img = img.mean(
                axis=-1
            )

        # ----------------------------------------------------
        # Generate patches
        # ----------------------------------------------------

        patches = []

        for _ in range(
            PATCHES_PER_IMAGE
        ):

            patch = sample_patch(
                img
            )

            # SAME normalization as training
            patch = (
                patch - 0.5
            ) / 0.25

            patch = torch.from_numpy(
                patch
            ).unsqueeze(
                0
            ).unsqueeze(
                0
            )

            patches.append(
                patch
            )

        patches = torch.cat(
            patches,
            dim=0
        ).float().to(device)

        # ----------------------------------------------------
        # Timing
        # ----------------------------------------------------

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.no_grad():

            preds = model(
                patches
            )

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()

        elapsed = end - start

        # ----------------------------------------------------
        # Average patch predictions
        # ----------------------------------------------------

        pred_normalized = (
            preds.mean().item()
        )

        # ----------------------------------------------------
        # Denormalize using TRAIN statistics
        # ----------------------------------------------------

        pred_um = (
            pred_normalized
            * z_std
            + z_mean
        )

        gt_um = z_gt[i]

        abs_error = abs(
            pred_um - gt_um
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        image_names.append(
            img_name
        )

        ground_truths.append(
            gt_um
        )

        predictions.append(
            pred_um
        )

        inference_times.append(
            elapsed
        )

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print(
            f"{img_name} | "
            f"{gt_um:.3f} | "
            f"{pred_um:.3f} | "
            f"{abs_error:.3f} | "
            f"{elapsed:.4f}"
        )

    # ========================================================
    # METRICS
    # ========================================================

    metrics = calculate_metrics(
        ground_truths,
        predictions
    )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    ground_truths_np = np.asarray(
        ground_truths
    )

    predictions_np = np.asarray(
        predictions
    )

    results_df = pd.DataFrame({

        "image":
            image_names,

        "z_GT_um":
            ground_truths_np,

        "z_pred_um":
            predictions_np,

        "abs_error_um":
            np.abs(
                predictions_np
                - ground_truths_np
            ),

        "signed_error_um":
            predictions_np
            - ground_truths_np,

        "inference_time_s":
            inference_times
    })

    results_df.to_csv(
        PREDICTIONS_CSV,
        index=False
    )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print()
    print("=" * 75)
    print("FINAL TEST RESULTS")
    print("=" * 75)

    print(
        f"Test MAE:                 "
        f"{metrics['MAE']:.6f} µm"
    )

    print(
        f"Test RMSE:                "
        f"{metrics['RMSE']:.6f} µm"
    )

    print(
        f"Test R²:                  "
        f"{metrics['R2']:.6f}"
    )

    print(
        f"Test Median AE:           "
        f"{metrics['MedianAE']:.6f} µm"
    )

    print(
        f"Test Bias:                "
        f"{metrics['Bias']:.6f} µm"
    )

    print(
        f"Mean inference time:      "
        f"{np.mean(inference_times):.6f} s/image"
    )

    print(
        f"Median inference time:    "
        f"{np.median(inference_times):.6f} s/image"
    )

    print()
    print(
        f"Prediction file:          "
        f"{PREDICTIONS_CSV}"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()
