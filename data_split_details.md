# Experimental Dataset Details

## Experimental Dataset

**Total number of images:** 76,305
**Z-depth range:** 0.000429–96.464237 µm

### Standard Data Split

| Split     | Number of Images | Proportion | Min. Z Depth (µm) | Max. Z Depth (µm) |
| :-------- | ---------------: | ---------: | ----------------: | ----------------: |
| Train     |           65,000 |     ~85.0% |          0.000429 |         96.464237 |
| Test      |            5,653 |      ~7.4% |          0.004260 |         65.325530 |
| Val       |            5,652 |      ~7.4% |          0.000950 |         65.236260 |
| **Total** |       **76,305** |   **100%** |                 — |                 — |

### Special Data Split

| Split     | Number of Images | Min. Z Depth (µm) | Max. Z Depth (µm) |
| :-------- | ---------------: | ----------------: | ----------------: |
| Train     |           70,463 |          0.000430 |         64.999782 |
| Test      |            2,921 |         65.011137 |         96.464237 |
| Val       |            2,921 |         65.010801 |         96.458196 |
| **Total** |       **76,305** |                 — |                 — |

> **Note:** The special split separates the data based on Z depth, with the training set covering depths up to approximately **65 µm**, while the validation and test sets cover depths above **65 µm**.

---

## UNAL Dataset

> **Note:** Only the standard data split is used for the UNAL dataset.

**Total number of images:** 3,540
**Z-depth range:** 330–4,050 µm

### Standard Data Split

| Split     | Number of Images | Proportion |
| :-------- | ---------------: | ---------: |
| Train     |            3,009 |      85.0% |
| Val       |              265 |       7.5% |
| Test      |              266 |       7.5% |
| **Total** |        **3,540** |   **100%** |

### Z-Depth Range

The Z-depth range is the same across all splits:

* **Minimum Z depth:** 330 µm
* **Maximum Z depth:** 4,050 µm
