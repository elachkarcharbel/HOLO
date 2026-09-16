import os
import time

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

TEST_IMG_DIR = "../experimental_dataset/expo_dataset_v3/test/images"
TEST_CSV = "../experimental_dataset/expo_dataset_v3/test/labels.csv"

MODEL_PATH = "best_baseline_cnn_final.pth"

IMG_SIZE = 128

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

OUTPUT_CSV = "test_predictions_baseline_cnn.csv"


# ============================================================
# DATASET
# ============================================================

class HoloTestDataset:

    def __init__(
        self,
        img_dir,
        csv_path
    ):

        self.img_dir = img_dir

        self.df = pd.read_csv(csv_path)

        self.image_paths = self.df["image"].values

        self.z = self.df[
            "Z input [um]"
        ].values.astype(np.float32)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        path = os.path.join(
            self.img_dir,
            self.image_paths[idx]
        )

        # Same preprocessing as training
        img = cv2.imread(
            path,
            cv2.IMREAD_GRAYSCALE
        )

        if img is None:
            raise RuntimeError(
                f"Could not read image: {path}"
            )

        img = cv2.resize(
            img,
            (IMG_SIZE, IMG_SIZE)
        ).astype(np.float32) / 255.0

        img = torch.tensor(
            img[None, :, :],
            dtype=torch.float32
        )

        return img


# ============================================================
# MODEL
# ============================================================

class ZPredictor(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
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

        self.regressor = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                64,
                1
            )
        )

    def forward(self, x):

        x = self.features(x)

        z = self.regressor(x)

        return z.squeeze(1)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BASELINE CNN TEST INFERENCE")
    print("=" * 70)

    print(f"Device       : {DEVICE}")
    print(f"Model        : {MODEL_PATH}")
    print(f"Image size   : {IMG_SIZE}x{IMG_SIZE}")
    print()

    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    dataset = HoloTestDataset(
        TEST_IMG_DIR,
        TEST_CSV
    )

    print(
        f"Number of test images: {len(dataset)}"
    )
    print()

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = ZPredictor().to(DEVICE)

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # Training saved:
    #
    # torch.save(
    #     {"model_state_dict": model.state_dict()},
    #     "best_cnn_model.pth"
    # )
    #
    # Therefore extract the state dictionary.

    if "model_state_dict" not in checkpoint:

        raise KeyError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    state_dict = checkpoint["model_state_dict"]

    model.load_state_dict(state_dict)

    model.eval()

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

        image_name = dataset.image_paths[idx]

        z_gt = float(
            dataset.z[idx]
        )

        # ----------------------------------------------------
        # Load and preprocess image
        # ----------------------------------------------------

        img = dataset[idx]

        img = img.unsqueeze(0).to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Accurate inference timing
        # ----------------------------------------------------

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.no_grad():

            z_pred_tensor = model(img)

            # Same output restriction used during training
            z_pred_tensor = torch.clamp(
                z_pred_tensor,
                0.0,
                100.0
            )

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        elapsed = (
            time.perf_counter()
            - start_time
        )

        z_pred = float(
            z_pred_tensor.item()
        )

        # ----------------------------------------------------
        # Errors
        # ----------------------------------------------------

        signed_error = z_pred - z_gt

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
    # CONVERT TO ARRAYS
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

    # ========================================================
    # METRICS
    # ========================================================

    # MAE
    mae = np.mean(
        abs_errors
    )

    # RMSE
    rmse = np.sqrt(
        np.mean(
            errors ** 2
        )
    )

    # R²
    ss_res = np.sum(
        (all_gt - all_pred) ** 2
    )

    ss_tot = np.sum(
        (all_gt - np.mean(all_gt)) ** 2
    )

    r2 = 1.0 - (
        ss_res / ss_tot
    )

    # Median Absolute Error
    median_ae = np.median(
        abs_errors
    )

    # Mean signed error
    bias = np.mean(
        errors
    )

    # Timing
    mean_time = np.mean([
        r["inference_time_s"]
        for r in results
    ])

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


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()