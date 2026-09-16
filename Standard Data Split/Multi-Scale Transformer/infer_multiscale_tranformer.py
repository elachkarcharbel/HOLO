import os
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import tifffile

from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATCH_SIZE_BIG = 256
PATCH_SIZE_SMALL = 128

PATCHES_PER_IMAGE = 16
GRID_SIZE = 4

NUM_BINS = 256
GAUSS_SIGMA_NORM = 0.15

MODEL_PATH = "best_depth_multiscale_transformer.pth"
NORM_FILE = "normalization_params.json"

TEST_IMG_DIR = "../experimental_dataset/expo_dataset_v3/test/images"
TEST_CSV = "../experimental_dataset/expo_dataset_v3/test/labels.csv"

OUTPUT_CSV = "test_predictions_multiscale_transformer.csv"


# ============================================================
# DATASET / PREPROCESSING
# ============================================================

class HoloTestDataset:
    def __init__(
        self,
        img_dir,
        csv_file,
        z_mean,
        z_std,
        z_norm_min,
        z_norm_max,
        img_mean,
        img_std,
    ):
        self.img_dir = img_dir
        self.df = pd.read_csv(csv_file)

        self.img_names = self.df.iloc[:, 0].values
        self.z_values = self.df.iloc[:, 1].values.astype(np.float32)

        self.z_mean = float(z_mean)
        self.z_std = float(z_std)

        self.z_norm_min = float(z_norm_min)
        self.z_norm_max = float(z_norm_max)

        self.img_mean = float(img_mean)
        self.img_std = float(img_std)

        self.bin_centers = torch.linspace(
            self.z_norm_min,
            self.z_norm_max,
            NUM_BINS
        )

    def __len__(self):
        return len(self.img_names)

    # --------------------------------------------------------
    # Same global normalization as training
    # --------------------------------------------------------

    def normalize_image_global(self, img):
        return (img - self.img_mean) / (self.img_std + 1e-6)

    # --------------------------------------------------------
    # Same local patch normalization as training
    # --------------------------------------------------------

    def normalize_patch(self, x):
        return (x - x.mean()) / (x.std() + 1e-6)

    # --------------------------------------------------------
    # Same gradient computation as training
    # --------------------------------------------------------

    def gradient_map(self, img):
        gx = torch.gradient(img, dim=0)[0]
        gy = torch.gradient(img, dim=1)[0]

        return torch.sqrt(gx ** 2 + gy ** 2)

    # --------------------------------------------------------
    # Same FFT computation as training
    # --------------------------------------------------------

    def fft_magnitude(self, img):
        fft = torch.fft.fftshift(torch.fft.fft2(img))
        mag = torch.log1p(torch.abs(fft))

        return mag

    # --------------------------------------------------------
    # Same deterministic 4x4 validation grid
    # --------------------------------------------------------

    def get_patch(self, img, i):
        H, W = img.shape

        step_y = (H - PATCH_SIZE_BIG) // (GRID_SIZE - 1)
        step_x = (W - PATCH_SIZE_BIG) // (GRID_SIZE - 1)

        row = i // GRID_SIZE
        col = i % GRID_SIZE

        y = row * step_y
        x = col * step_x

        return img[
            y:y + PATCH_SIZE_BIG,
            x:x + PATCH_SIZE_BIG
        ]

    # --------------------------------------------------------
    # Same center crop as training
    # --------------------------------------------------------

    def center_crop(self, patch, crop_size=128):
        H, W = patch.shape

        y0 = (H - crop_size) // 2
        x0 = (W - crop_size) // 2

        return patch[
            y0:y0 + crop_size,
            x0:x0 + crop_size
        ]

    # --------------------------------------------------------
    # Construct exactly the six training channels
    # --------------------------------------------------------

    def make_patch(self, patch_big):

        patch_small = self.center_crop(
            patch_big,
            PATCH_SIZE_SMALL
        )

        # Big channels
        big_raw = self.normalize_patch(patch_big)

        big_grad = self.normalize_patch(
            self.gradient_map(patch_big)
        )

        big_fft = self.normalize_patch(
            self.fft_magnitude(patch_big)
        )

        # Small channels
        small_raw = self.normalize_patch(patch_small)

        small_grad = self.normalize_patch(
            self.gradient_map(patch_small)
        )

        small_fft = self.normalize_patch(
            self.fft_magnitude(patch_small)
        )

        # Resize small channels 128 -> 256
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

        # Same channel order as training
        stacked = torch.stack([
            big_raw,
            big_grad,
            big_fft,
            small_raw,
            small_grad,
            small_fft
        ])

        return stacked

    # --------------------------------------------------------
    # Get all 16 deterministic patches
    # --------------------------------------------------------

    def get_image_patches(self, idx):

        img_path = os.path.join(
            self.img_dir,
            self.img_names[idx]
        )

        img = tifffile.imread(img_path).astype(np.float32)

        if img.ndim == 3:
            img = img.mean(axis=-1)

        img = torch.from_numpy(img).float()

        # Same global image normalization
        img = self.normalize_image_global(img)

        patches = []

        for i in range(PATCHES_PER_IMAGE):

            patch_big = self.get_patch(img, i)

            stacked = self.make_patch(patch_big)

            patches.append(stacked)

        patches = torch.stack(patches)

        return patches


# ============================================================
# MODEL
# ============================================================

class PatchEncoder(nn.Module):

    def __init__(self, in_ch=6, embed_dim=256):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(
                in_ch, 32,
                3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.GELU(),

            nn.Conv2d(
                32, 64,
                3,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.GELU(),

            nn.Conv2d(
                64, 128,
                3,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.GELU(),

            nn.Conv2d(
                128, 256,
                3,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(256),

            nn.GELU(),

            nn.Conv2d(
                256,
                embed_dim,
                3,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(embed_dim),

            nn.GELU(),

            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):

        x = self.net(x)

        return x.view(x.size(0), -1)


class DepthTransformerGaussian(nn.Module):

    def __init__(
        self,
        in_ch=6,
        embed_dim=256,
        num_layers=4,
        nhead=8,
        num_bins=256
    ):
        super().__init__()

        self.encoder = PatchEncoder(
            in_ch=in_ch,
            embed_dim=embed_dim
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(
            enc_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                embed_dim,
                256
            ),

            nn.GELU(),

            nn.Dropout(0.2),

            nn.Linear(
                256,
                num_bins
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

        feats = self.encoder(x)

        feats = feats.view(
            B,
            P,
            -1
        )

        feats = self.transformer(feats)

        global_feat = feats.mean(dim=1)

        logits = self.classifier(global_feat)

        return logits


# ============================================================
# DECODING
# ============================================================

def decode_expected_z(
    logits,
    bin_centers,
    z_mean,
    z_std
):

    probs = torch.softmax(
        logits,
        dim=1
    )

    expected_norm = torch.sum(
        probs * bin_centers.unsqueeze(0),
        dim=1
    )

    expected_um = (
        expected_norm * z_std
        + z_mean
    )

    return expected_um


# ============================================================
# MAIN INFERENCE
# ============================================================

def main():

    print("=" * 70)
    print("MULTISCALE TRANSFORMER TEST INFERENCE")
    print("=" * 70)

    print(f"Device              : {DEVICE}")
    print(f"Model               : {MODEL_PATH}")
    print(f"Patch size          : {PATCH_SIZE_BIG}x{PATCH_SIZE_BIG}")
    print(f"Small crop          : {PATCH_SIZE_SMALL}x{PATCH_SIZE_SMALL}")
    print(f"Patches/image       : {PATCHES_PER_IMAGE}")
    print(f"Number of bins      : {NUM_BINS}")
    print()

    # --------------------------------------------------------
    # Load normalization parameters from training
    # --------------------------------------------------------

    if not os.path.exists(NORM_FILE):
        raise FileNotFoundError(
            f"Normalization file not found: {NORM_FILE}"
        )

    with open(NORM_FILE, "r") as f:
        norm = json.load(f)

    z_mean = float(norm["z_mean"])
    z_std = float(norm["z_std"])

    z_norm_min = float(norm["z_norm_min"])
    z_norm_max = float(norm["z_norm_max"])

    img_mean = float(norm["img_mean"])
    img_std = float(norm["img_std"])

    print("[TRAINING NORMALIZATION PARAMETERS]")
    print(f"z_mean     = {z_mean:.8f}")
    print(f"z_std      = {z_std:.8f}")
    print(f"z_norm_min = {z_norm_min:.8f}")
    print(f"z_norm_max = {z_norm_max:.8f}")
    print(f"img_mean   = {img_mean:.8f}")
    print(f"img_std    = {img_std:.8f}")
    print()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = HoloTestDataset(
        TEST_IMG_DIR,
        TEST_CSV,
        z_mean=z_mean,
        z_std=z_std,
        z_norm_min=z_norm_min,
        z_norm_max=z_norm_max,
        img_mean=img_mean,
        img_std=img_std
    )

    print(f"Number of test images: {len(dataset)}")
    print()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = DepthTransformerGaussian(
        in_ch=6,
        embed_dim=256,
        num_layers=4,
        nhead=8,
        num_bins=NUM_BINS
    ).to(DEVICE)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(state_dict)

    model.eval()

    bin_centers = torch.linspace(
        z_norm_min,
        z_norm_max,
        NUM_BINS
    ).to(DEVICE)

    print("✓ Model loaded successfully")
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

        image_name = dataset.img_names[idx]
        z_gt = float(dataset.z_values[idx])

        # Build 16 deterministic patches
        patches = dataset.get_image_patches(idx)

        # [P, C, H, W] -> [1, P, C, H, W]
        patches = patches.unsqueeze(0).to(
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
            logits = model(patches)

            z_pred_tensor = decode_expected_z(
                logits,
                bin_centers,
                z_mean,
                z_std
            )

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start_time

        z_pred = float(z_pred_tensor.item())

        abs_error = abs(z_pred - z_gt)
        signed_error = z_pred - z_gt

        total_inference_time += elapsed

        all_gt.append(z_gt)
        all_pred.append(z_pred)

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

    errors = all_pred - all_gt
    abs_errors = np.abs(errors)

    mae = np.mean(abs_errors)

    rmse = np.sqrt(
        np.mean(errors ** 2)
    )

    ss_res = np.sum(
        (all_gt - all_pred) ** 2
    )

    ss_tot = np.sum(
        (all_gt - np.mean(all_gt)) ** 2
    )

    r2 = 1.0 - (ss_res / ss_tot)

    median_ae = np.median(abs_errors)

    bias = np.mean(errors)

    mean_time = np.mean([
        r["inference_time_s"]
        for r in results
    ])

    total_time = total_inference_time

    # ========================================================
    # SAVE CSV
    # ========================================================

    results_df = pd.DataFrame(results)

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

    print(f"Test MAE                  : {mae:.6f} µm")
    print(f"Test RMSE                 : {rmse:.6f} µm")
    print(f"Test R²                   : {r2:.6f}")
    print(f"Test Median Absolute Error: {median_ae:.6f} µm")
    print(f"Test Bias                 : {bias:.6f} µm")

    print()
    print(f"Mean inference time/image : {mean_time:.6f} s")
    print(f"Total inference time      : {total_time:.2f} s")

    print()
    print(f"Saved predictions to      : {OUTPUT_CSV}")

    print("=" * 70)


if __name__ == "__main__":
    main()