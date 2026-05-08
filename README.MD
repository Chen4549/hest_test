# Spatial Transcriptomics Prediction from H&E Images

This repository implements spatial transcriptomics gene expression prediction from H&E whole slide images (WSIs). Built on top of the [HEST-Bench](https://github.com/mahmoodlab/HEST) framework, it leverages various foundation models and spatial neighbor aggregation techniques to accurately map tissue morphology to gene expression.

## Task Overview

The core objective is to predict RNA gene expression at specific spatial spots on a tissue slide using frozen patch-level embeddings from foundation models. 

**Pipeline:**
1. Extract an image patch from the H&E WSI for each spatial spot.
2. Encode the patch using a pre-trained foundation model.
3. Pass the resulting embeddings into a ridge regression probe.
4. Predict the expression values of 50 highly variable genes.

## Experimental Settings

* **Target Variable (RNA Expression):** Raw gene counts are normalized using `log1p` (i.e., `log(count + 1)`). Raw counts are typically zero-inflated and highly right-skewed. The `log1p` transformation compresses this dynamic range, stabilizes variance, and optimizes the target distribution for linear regression.
* **Evaluation Metric:** Pearson correlation coefficient. This is computed per gene between predicted and actual expression values across all test spots, then averaged across the 50 genes. Pearson correlation is preferred over MSE because it is scale-invariant, measuring whether the model correctly ranks spots by expression level regardless of absolute scale. This ensures robustness against normalization differences across slides.
* **Validation Strategy:** K-fold cross-validation is used (folds vary by dataset). Results are reported as the mean and standard deviation of per-fold Pearson correlations to account for train/test split variability and ensure a reliable estimate of generalizability.
* **Dimensionality Reduction:** By default, Principal Component Analysis (PCA) reduces embeddings to 256 dimensions prior to ridge regression, ensuring fair computational comparisons across different models.

## Environment Setup

For full setup documentation, refer to the official [HEST GitHub repository](https://github.com/mahmoodlab/HEST.git).

```bash
conda create -n hest python=3.11
conda activate hest
pip install -e .
pip install -e ".[benchmark]"
```

**GPU Acceleration (Optional but Recommended):**
If a GPU is available, install the following libraries for faster inference:

```bash
pip install \
    --extra-index-url=[https://pypi.nvidia.com](https://pypi.nvidia.com) \
    cudf-cu12==24.6.* dask-cudf-cu12==24.6.* cucim-cu12==24.6.* \
    raft-dask-cu12==24.6.*
```

> **Note:** **DINOv3 weights** must be downloaded separately and stored locally before running the associated scripts.

---

## Usage & Script Launching

### 1. Foundation Models (Baseline)

```bash
python implementations/benchmark.py \
  --model_name uni_v2 \
  --bench_data_root /PATH/TO/DATA \
  --embed_dataroot /PATH/TO/EMBEDDINGS \
  --results_dir /PATH/TO/RESULTS
  --weights /PATH/TO/WEIGHTS #if using DINO
```

### ⚙️ Baseline Arguments

| Argument | Description |
| :--- | :--- |
| **`--model_name`** | Foundation model to use (`uni_v2`, `dinov3_vitl16`, `dinov3_vitb16`). |
| **`--bench_data_root`** | Root directory containing the downloaded HEST-Bench datasets. |
| **`--embed_dataroot`** | Directory where extracted patch embeddings will be saved. |
| **`--results_dir`** | Directory where prediction results will be saved. |
| **`--weights`** | Path to the local DINOv3 pretrained weights file. *(DINOv3 only)* |

*Example using DINOv3:*
```bash
python implementations/benchmark.py \
  --model_name dinov3_vitb16 \
  --weights /PATH/TO/dinov3_vitb16_weights.pth \
  --bench_data_root /PATH/TO/DATA \
  --embed_dataroot /PATH/TO/EMBEDDINGS \
  --results_dir /PATH/TO/RESULTS
```

### 2. Foundation Models + Spatial Neighbor Aggregation

```bash
python implementations/neighbor_agg.py \
  --model_name dinov3_vitb16 \
  --weights /PATH/TO/dinov3_vitb16_weights.pth \
  --spatial_agg \
  --spatial_k 8 \
  --spatial_agg_mode concat \
  --bench_data_root /PATH/TO/DATA \
  --embed_dataroot /PATH/TO/EMBEDDINGS \
  --results_dir /PATH/TO/RESULTS
```

**Additional Aggregation Arguments:**

| Argument | Description |
| :--- | :--- |
| `--spatial_agg` | Flag to enable spatial neighbor aggregation. |
| `--spatial_k` | Number of nearest neighbor spots to aggregate (e.g., `8` = 3×3 window minus center, `24` = 5×5 window minus center). |
| `--spatial_agg_mode` | Aggregation mode: `concat` (concatenate neighbor embeddings, then PCA) or `mean` (average all embeddings, preserves dimensionality). |

---

## Results

Mean Pearson correlation across 9 HEST-Bench datasets (IDC, PRAD, PAAD, SKCM, COAD, READ, CCRCC, LUNG, LYMPH_IDC):

| Method | IDC | PRAD | PAAD | SKCM | COAD | READ | CCRCC | LUNG | LYMPH_IDC | **Avg** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **UNI v2** | 0.5898 | 0.3567 | 0.5000 | 0.6611 | 0.3018 | 0.2226 | 0.2640 | 0.5589 | 0.2728 | **0.4142** |
| **DINOv3-L16** | 0.4911 | 0.3712 | 0.3738 | 0.5474 | 0.2511 | 0.1259 | 0.2172 | 0.5025 | 0.2448 | **0.3472** |
| **DINOv3-B16** | 0.4811 | 0.3127 | 0.3956 | 0.5057 | 0.2130 | 0.1058 | 0.2191 | 0.5002 | 0.2453 | **0.3309** |
| **DINOv3-B16 + Concat (k=8)** | 0.4826 | 0.3529 | 0.3897 | 0.4877 | 0.2648 | 0.0905 | 0.2222 | 0.5049 | 0.2404 | **0.3373** |
| **DINOv3-B16 + Mean (k=8)** | 0.4476 | 0.2230 | 0.3420 | 0.4195 | 0.1491 | 0.0973 | 0.2528 | 0.4008 | 0.2410 | **0.2859** |

---

## Summary & Findings

* **Pathology Pretraining Wins:** UNI v2 achieves the highest average Pearson correlation (0.4142), significantly outperforming both DINOv3 variants. Because UNI v2 is pretrained specifically on histopathology images, its embeddings naturally align better with tissue-level biological features driving gene expression.
* **Spatial Aggregation Nuances:** 
  * **Concat:** Spatial neighbor aggregation via concatenation (k=8) yields a modest, though inconsistent, improvement over the baseline DINOv3-B16 (+0.006 average, improving 5/9 datasets). The largest gains occur in PRAD (+0.040) and COAD (+0.052), suggesting the local tissue microenvironment is a stronger determinant of expression patterns in these specific cancer types.
  * **Mean:** Conversely, mean aggregation consistently degrades performance (drops in 8/9 datasets, −0.045 average). Averaging 9 distinct spots over-smooths the embeddings, ultimately diluting the discriminative signal of the center spot.

## Limitations

* **Inconsistent Aggregation Benefits:** Spatial aggregation via concatenation does not universally improve performance. The dataset-dependent benefits imply that simple k=8 pixel-space nearest neighbors might fail to capture true biological context across all cancer types. While a learnable aggregation head (e.g., a small transformer) could intelligently weight neighbor contributions, it would require substantially more training data to stave off overfitting.
* **Failure of Mean Aggregation:** Averaging neighboring embeddings collapses the distinct identity of a spot into its broader microenvironment. This effectively destroys the localized discriminative signals that the ridge regression model requires to function.
* **The Domain Gap:** The gap between general-purpose and pathology-specific encoders remains large. Despite being powerful large-scale vision transformers, DINOv3 models trail UNI v2 by roughly ~8 points on average. Spatial aggregation cannot close this gap, proving that domain-specific pretraining is the primary driver of success in spatial transcriptomics.
* **Data Constraints:** Each dataset in HEST-Bench contains roughly 3,000–5,000 spatial spots per WSI. This limits the capacity of the regression probe. Attempting to use more expressive spatial aggregation strategies (like Graph Neural Networks or cross-attention) in this low-data regime will likely lead to severe overfitting without stronger regularization or vastly larger WSI cohorts.
