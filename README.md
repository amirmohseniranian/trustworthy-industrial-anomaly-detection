# Trustworthy Industrial Anomaly Detection Under Distribution Shift

**VisA PCB1 — leakage-aware anomaly scoring, calibration, robustness stress testing, and reproducible artifact generation**

This repository contains a Google Colab-oriented research implementation for **industrial visual anomaly detection on the VisA PCB1 category**. The pipeline uses frozen pretrained visual backbones to construct normalized image embeddings, scores test images against a reference bank using cosine-derived $k$-nearest-neighbor distances, calibrates those anomaly scores with logistic and isotonic regression using a disjoint training-derived calibration pool, and evaluates both clean and corrupted test conditions.

The implementation is organized around **data provenance, leakage control, calibration quality, distribution-shift stress testing, and artifact integrity**, rather than end-to-end model training.

> **Implementation basis.** This README documents the accompanying Python source file `trustworthy_industrial_anomaly_detection_visa_compact_fixed_v2_colab.py`. The claims below are restricted to behavior implemented by that source file or to explicitly identified mathematical definitions.

---

## Overview

The experimental pipeline has five central ideas:

1. **Reconstruct a compact VisA PCB1 dataset** from verified Parquet files downloaded from the Hugging Face dataset repository.
2. **Extract frozen deep visual representations** with ResNet-50 and DINOv2 ViT-S/14.
3. **Compute non-parametric anomaly scores** from distances to a reference bank of normal training images.
4. **Calibrate anomaly probabilities** using clean normal examples plus synthetic anomalies drawn only from a disjoint calibration pool.
5. **Evaluate reliability and robustness** on untouched real test defects, including controlled image corruptions that emulate distribution shift.

The main experiment uses:

- **1 VisA category:** `pcb1`
- **2 frozen backbones:** `resnet50`, `dinov2_vits14`
- **3 seeds:** `13`, `42`, `123`
- **6 main trials:** 2 backbones × 1 category × 3 seeds
- **Reference bank size:** up to 250 normal training images
- **Calibration pool size:** up to 150 additional normal training images
- **Main $k$-NN setting:** $k=5$
- **Input resolution:** $224 \times 224$

No backbone fine-tuning or task-specific neural-network training is performed by this script.

---

## Scientific Motivation

Industrial anomaly detection is often evaluated under an implicit assumption that the deployment distribution resembles the development distribution. In practice, visual appearance can change because of illumination, blur, compression, resolution loss, acquisition conditions, or other nuisance factors.

This project therefore separates three related questions:

### Anomaly discrimination

Can a frozen visual representation distinguish normal PCB images from real industrial defects?

### Probability calibration

Can the raw anomaly score be transformed into a probability-like quantity that better reflects empirical defect frequency?

### Robustness under shift

How do anomaly discrimination and calibration behave when test images are subjected to controlled out-of-distribution image corruptions?

The design explicitly keeps the **real VisA test set outside calibrator fitting**, so the calibration models are not fitted on the same real defects on which the final evaluation is reported.

---

## Dataset

### VisA PCB1

The implementation uses the `pcb1` category from the **VisA** industrial visual anomaly dataset. Rather than relying on a pre-existing extracted directory tree, the script reconstructs the required image/mask files from compact Parquet files stored in the Hugging Face dataset repository:

- Repository: `BrachioLab/visa`
- Train file: `data/pcb1.train-00000-of-00001.parquet`
- Test file: `data/pcb1.test-00000-of-00001.parquet`

The source code records an expected SHA-256 digest for both Parquet files and stops when a downloaded file does not match its expected hash.

### Reconstruction procedure

For each Parquet row:

- `label == 0` is mapped to `normal`.
- `label == 1` is mapped to `anomaly`.
- RGB images are written as JPEG with quality 95.
- Anomaly masks, when present, are written as grayscale PNG files.
- A standardized `1cls.csv` index is generated containing:
  - `object`
  - `split`
  - `label`
  - `image`
  - `mask`

The experiment then uses:

- **Training normals** as the source population for the reference bank and calibration pool.
- **Untouched test samples** for the primary evaluation.

No test image is sampled into either the reference bank or the calibration pool.

---

## Data Splitting and Leakage Control

The source implements an explicit two-stage split of the **training-normal population**:

1. Randomly permute the training rows using the trial seed.
2. Allocate the first block to the **reference bank**.
3. Allocate the next disjoint block to the **calibration pool**.

Formally, let the normal training set be $D_{\mathrm{train}}$. For a seed $s$, a random permutation partitions it into

$$
D_{\mathrm{train}}
\rightarrow
B_s \cup C_s,
\qquad
B_s \cap C_s = \varnothing.
$$

Here:

- $B_s$ is the reference bank.
- $C_s$ is the calibration pool.

The implementation caps these sets at:

$$
|B_s| \le 250,
\qquad
|C_s| \le 150.
$$

The calibration pool is used to generate clean normal calibration samples and synthetic anomaly calibration samples. The real VisA test set remains separate until evaluation.

### Why the split matters

The anomaly score for a test image depends on distances to the reference bank. The calibration model is fitted on scores from the calibration pool rather than on the test set. This separation matters because fitting the calibrator on the real evaluation set would make the reported calibration metrics optimistic and leakage-prone.

---

## Preprocessing

All images are converted to RGB and processed with the following ImageNet normalization pipeline:

1. Resize to $224 \times 224$.
2. Convert to tensor.
3. Normalize each channel with the ImageNet mean and standard deviation.

The normalized input is

$$
x'_{c}
=
\frac{x_c-\mu_c}{\sigma_c},
$$

with

$$
\mu=(0.485,\,0.456,\,0.406),
\qquad
\sigma=(0.229,\,0.224,\,0.225).
$$

---

## Feature Extraction

Two pretrained backbones are loaded in evaluation mode with all parameters frozen.

### ResNet-50

The model is initialized with the torchvision default pretrained weights, and the final fully connected classification layer is replaced with `Identity`. The resulting vector is used directly as the image representation.

### DINOv2 ViT-S/14

The script loads `facebookresearch/dinov2` with the `dinov2_vits14` model.

Rather than using only the class token, the implementation forms a multi-scale global representation by concatenating:

- the normalized CLS token;
- the mean of the normalized patch tokens.

The representation is

$$
z
=
\left[
z_{\mathrm{CLS}}
\;\middle\|\;
\frac{1}{N}\sum_{i=1}^{N}z_{\mathrm{patch},i}
\right].
$$

The final embedding is then L2-normalized:

$$
\hat{z}
=
\frac{z}{\lVert z\rVert_2}.
$$

Feature extraction is performed in inference mode, and CUDA automatic mixed precision is used when a GPU is available.

### Feature caching

Clean embeddings for all reconstructed images are cached under `results/` as:

- `features_<backbone>.npz`
- `features_<backbone>_meta.csv`

This avoids recomputing clean embeddings when subsequent stages access the same backbone representations.

---

## Anomaly Scoring

The anomaly score is the mean distance to the $k$ nearest reference embeddings.

Because the embeddings are L2-normalized, dot products are equivalent to cosine similarity:

$$
s_{ij}
=
\hat{z}_i^\top \hat{r}_j,
$$

where $\hat{z}_i$ is the query embedding and $\hat{r}_j$ is a normalized reference embedding.

The corresponding cosine-derived distance is

$$
d_{ij}
=
1-s_{ij}.
$$

For query $i$, let $N_k(i)$ denote the indices of its $k$ nearest reference samples. The anomaly score is

$$
a_i
=
\frac{1}{k}
\sum_{j\in N_k(i)} d_{ij}.
$$

The main experiment uses

$$
k=5.
$$

Higher values of $a_i$ indicate that the query embedding is farther from the normal reference bank and is therefore treated as more anomalous.

### Computational detail

Distance calculations are performed in chunks to limit memory use:

- query chunk size: 512
- matrix operation: query embeddings multiplied by the transpose of reference embeddings
- nearest distances: partial selection with `numpy.partition`

This avoids materializing the full distance matrix for all queries at once.

---

## Synthetic Calibration Anomalies

The calibration pool contains normal training images. To provide positive calibration examples without using real test defects, the implementation generates four synthetic defect types:

- `cutpaste_patch`
- `subtle_noise`
- `subtle_blur`
- `micro_scratch`

These transformations are used **only for calibrator fitting** in the main protocol.

### Cut-and-paste patch

A small local patch is copied within the same image. The source patch may also be rotated by 180 degrees before insertion.

### Subtle noise

Independent Gaussian noise is added in normalized pixel space with a randomly sampled standard deviation of 0.015 to 0.035.

### Subtle blur

A Gaussian blur radius is sampled from 0.4 to 0.9.

### Micro-scratch

A short straight perturbation is added by increasing pixel intensity along a randomly oriented line segment.

For each synthetic calibration family, the generated image receives calibration label $1$. Clean calibration images receive label $0$.

The combined calibrator-training dataset is constructed as

$$
D_{\mathrm{cal}}
=
D_{\mathrm{clean}}
\cup
D_{\mathrm{cutpaste}}
\cup
D_{\mathrm{noise}}
\cup
D_{\mathrm{blur}}
\cup
D_{\mathrm{scratch}}.
$$

The source creates one synthetic version of each calibration-pool image for each defect type, so the positive calibration pool can contain up to four times as many synthetic samples as the clean calibration pool.

> **Interpretation note.** These transformations are synthetic calibration mechanisms. The code does not establish that they are statistically identical to real VisA defects; their role is to provide a controlled anomaly-like score distribution for calibration.

---

## Probability Calibration

The raw anomaly score $a$ is not inherently a calibrated probability. The script fits two post hoc calibrators using only the training-derived calibration pool.

### Logistic calibration

A one-dimensional logistic regression is fitted with:

- `C = 1.0`
- solver = `lbfgs`
- `max_iter = 1000`

The model learns a probability of the form

$$
p_{\mathrm{log}}(y=1\mid a)
=
\sigma(\beta_0+\beta_1 a),
$$

where

$$
\sigma(t)
=
\frac{1}{1+e^{-t}}.
$$

### Isotonic calibration

An isotonic regression maps anomaly scores to a monotonic probability estimate constrained to $[0,1]$. Scores outside the fitted range are clipped using `out_of_bounds='clip'`.

The calibrated outputs are therefore:

- `logistic_probability`
- `isotonic_probability`

These quantities are evaluated on the untouched real VisA test set.

---

## Evaluation Metrics

The implementation reports both **discrimination** and **probability quality** metrics.

### AUROC

The Area Under the Receiver Operating Characteristic curve is computed from the raw anomaly score:

$$
\mathrm{AUROC}
=
\int_0^1
\mathrm{TPR}(u)
\,d\mathrm{FPR}(u).
$$

Higher values indicate better ranking-based separation of normal and anomalous images.

### AUPRC

Average precision is computed from the raw anomaly score to summarize precision-recall performance across thresholds.

Higher values are preferable.

### Adaptive Expected Calibration Error

The calibration error function first builds **equal-frequency (quantile) bins** from the predicted probabilities. Let bin $b$ contain a fraction $w_b$ of the samples, with mean predicted probability $\mathrm{conf}_b$ and empirical anomaly frequency $\mathrm{freq}_b$. The reported metric is

$$
\mathrm{ECE}
=
\sum_b
w_b
\left|
\mathrm{conf}_b-\mathrm{freq}_b
\right|.
$$

The function is called with its default value of **10 bins**.

> **Important naming detail.** The CSV columns are named `logistic_ece15` and `isotonic_ece15`, but the implementation calls `expected_calibration_error_adaptive` with `n_bins=10`. Therefore, those columns contain **adaptive 10-bin ECE values, not ECE-15 values**. The column names are retained by the source code but should not be interpreted as a 15-bin calculation.

If the predicted probabilities have extremely low variance and the quantile boundaries collapse, the function falls back to equal-width bins.

### Brier score

For binary labels $y_i\in\{0,1\}$ and predicted probabilities $p_i$, the Brier score is

$$
\mathrm{Brier}
=
\frac{1}{n}
\sum_{i=1}^{n}
(p_i-y_i)^2.
$$

Lower values are preferable.

### Risk-coverage curve and AURC

Predictions are converted to binary labels using a probability threshold of 0.5:

$$
\hat{y}_i
=
\mathbf{1}[p_i\ge 0.5].
$$

Prediction confidence is defined as

$$
c_i
=
\max(p_i,1-p_i).
$$

Samples are sorted from highest to lowest confidence. At coverage level $q$, empirical risk is the cumulative error rate among the retained highest-confidence samples.

The area under the risk-coverage curve is approximated numerically as

$$
\mathrm{AURC}
=
\int_0^1
R(q)\,dq.
$$

Lower AURC is preferable because it corresponds to lower error among confident predictions.

---

## Main Experimental Protocol

For each backbone and seed, the procedure is:

### Step 1 — Prepare the data

Select:

- normal training images for `pcb1`;
- all real `pcb1` test images.

### Step 2 — Sample the reference and calibration pools

Using the current seed:

- up to 250 normal training images form the reference bank;
- the next up to 150 disjoint normal training images form the calibration pool.

### Step 3 — Score clean calibration images

Clean calibration images are assigned label 0 and scored against the reference bank.

### Step 4 — Generate synthetic calibration anomalies

Each calibration image is transformed with each of the four synthetic defect generators and scored against the same reference bank.

### Step 5 — Fit calibrators

Logistic and isotonic regressions are fitted on the combined calibration scores and labels.

### Step 6 — Evaluate real test images

Real VisA test embeddings are scored against the reference bank. The raw scores are then passed through both calibrators.

### Step 7 — Compute metrics

The script computes:

- AUROC
- AUPRC
- adaptive ECE
- Brier score
- AURC

### Step 8 — Save predictions

Per-sample predictions are written with:

- source path
- category
- ground-truth label
- raw anomaly score
- logistic probability
- isotonic probability
- backbone
- seed

---

## Experimental Matrix

The main experiment is:

| Dimension | Values |
|---|---|
| Category | `pcb1` |
| Backbones | `resnet50`, `dinov2_vits14` |
| Seeds | `13`, `42`, `123` |
| Main reference-bank budget | 250 |
| Calibration-pool budget | 150 |
| $k$ | 5 |
| Input size | $224 \times 224$ |
| Main trials | 6 |

Each trial is independent at the level of its random reference/calibration allocation and synthetic-corruption seeds.

---

## Hyperparameter Sensitivity

A focused sensitivity analysis is performed for **DINOv2 only**.

The reference-bank size is varied over

$$
B\in\{50,150,250\},
$$

and the number of nearest neighbors is varied over

$$
k\in\{1,5,10\}.
$$

All three seeds are evaluated for every $(B,k)$ pair.

This produces

$$
3\times 3\times 3=27
$$

DINOv2 sensitivity trials.

The sensitivity analysis reports:

- AUROC
- AUPRC

and summarizes mean AUROC across the three seeds for each configuration.

---

## Robustness Stress Test

The robustness stage evaluates both backbones under four controlled image corruptions:

- `blur`
- `brightness`
- `jpeg`
- `resolution`

Each corruption is evaluated at severity levels

$$
s\in\{1,2,3\}.
$$

A clean reference condition with severity 0 is also recorded.

### Corruption definitions

| Corruption | Severity behavior |
|---|---|
| Blur | Gaussian blur radius = $0.7s$ |
| Brightness | Multipliers: 1.15, 0.75, 0.50 |
| JPEG | Qualities: 75, 45, 20 |
| Resolution | Downsample factors: 0.65, 0.40, 0.25, then resize back |

The robustness stage uses the seed **13** for reference/calibration allocation and corruption generation.

The calibrators are fitted using the same training-derived clean/synthetic calibration protocol, after which corrupted **real test images** are evaluated without refitting.

For each stress condition the script reports:

- AUROC
- AUPRC
- logistic adaptive ECE
- logistic Brier score
- isotonic adaptive ECE
- isotonic Brier score

> **Interpretation note.** The stress stage is not a multi-seed aggregate. It is a fixed-seed robustness analysis based on seed 13.

---

## Reliability Analysis

For DINOv2 at seed 13, the script generates a reliability diagram for isotonic probabilities on the real VisA test set.

The plot contains:

1. Mean predicted confidence per adaptive probability bin.
2. Empirical anomaly frequency per bin.
3. A diagonal perfect-calibration reference.
4. A lower histogram showing the distribution of predicted probabilities.

This makes systematic under-confidence, over-confidence, or concentration of predictions visually inspectable.

---

## Ablation Study

The source constructs a compact CSV ablation table for each backbone at seed 13:

1. Raw anomaly score.
2. Raw score + logistic calibration.
3. Raw score + isotonic calibration.

The code deliberately leaves **AUROC and AUPRC identical across the three rows** because it copies the raw-score values into the calibrated rows rather than recomputing ranking metrics from the calibrated outputs.

The calibration-specific quantities are taken from the corresponding calibrated predictions.

> **Methodological interpretation.** This table is therefore a **calibration-focused ablation**, not an independent ranking-performance comparison between raw and calibrated outputs. Its AUROC/AUPRC entries should be read as the baseline ranking performance carried through each row, exactly as implemented.

---

## Qualitative Error Diagnosis

The script performs a qualitative inspection for DINOv2 at seed 13.

### False positives

The four highest-scoring normal test images are selected:

$$
\text{label}=0
\quad\text{and highest anomaly score}.
$$

### False negatives

The four lowest-scoring real anomaly test images are selected:

$$
\text{label}=1
\quad\text{and lowest anomaly score}.
$$

A grid is generated for each group, displaying the source image together with:

- raw anomaly score
- isotonic probability

This stage is intended for visual diagnosis of difficult normal examples and missed defects rather than numerical model selection.

---

## Reproducibility and Determinism

The implementation explicitly seeds:

- Python `random`
- NumPy
- PyTorch
- CUDA PyTorch RNGs when CUDA is available

with the selected trial seed.

However, the code sets

```python
torch.backends.cudnn.benchmark = True
```

Therefore, **bit-for-bit deterministic GPU execution is not guaranteed** across environments. The protocol is seeded, but it should not be described as strictly deterministic at the numerical-kernel level.

Other reproducibility mechanisms include:

- verified SHA-256 hashes for the downloaded Parquet files;
- explicit seed values;
- cached clean embeddings;
- explicit configuration constants;
- deterministic seed derivation for synthetic and stress corruptions;
- explicit output files;
- archive packaging.

---

## Environment

The script is designed for Google Colab and recommends a GPU-enabled runtime.

The environment setup contains a deliberate Pillow reset sequence because the implementation requires a clean native-extension state:

1. Remove loaded `PIL` modules from `sys.modules`.
2. Uninstall existing Pillow/PIL packages.
3. Remove matching site-packages artifacts.
4. Install `Pillow==12.3.0`.
5. In Colab, terminate the current Python process so that the new C extensions are loaded by a fresh runtime.

Outside Colab, the script falls back to the normal Pillow import path.

### Core Python dependencies used by the source

The implementation imports or relies on:

- Python standard library
- Pillow
- NumPy
- pandas
- Matplotlib
- seaborn
- PyTorch
- torchvision
- DuckDB
- `huggingface_hub`
- scikit-learn

The source explicitly pins Pillow to version `12.3.0`, but it does **not** fully pin every imported dependency to an exact version.

---

## Installation

The intended execution environment is Google Colab with GPU acceleration enabled.

After the runtime is available, the script itself performs the Pillow-specific environment reset and then imports the remaining scientific stack.

Because the source depends on pretrained model downloads, the runtime also needs network access when the corresponding artifacts are not already cached.

No separate package-lock or fully pinned environment file is included by the source script.

---

## Usage

The primary entry point is the provided Python script:

```bash
python trustworthy_industrial_anomaly_detection_visa_compact_fixed_v2_colab.py
```

In Google Colab, the same logic is organized as executable cells in the notebook-oriented version described by the script header.

The pipeline proceeds from top to bottom:

```text
Environment setup
    ↓
Dataset download and hash verification
    ↓
VisA PCB1 reconstruction
    ↓
Exploratory visualizations
    ↓
Synthetic calibration-corruption definitions
    ↓
Frozen backbone loading
    ↓
Clean feature extraction and caching
    ↓
Main calibration/evaluation trials
    ↓
Hyperparameter sensitivity
    ↓
Robustness stress testing
    ↓
Reliability diagram and ablation
    ↓
Qualitative error diagnosis
    ↓
Artifact packaging
```

---

## Output Directory

The script uses the root directory:

```text
/content/visa_research_project/
```

The main subdirectories are:

```text
visa_research_project/
├── data/
│   ├── compact_parquet/
│   ├── VisA/
│   └── 1cls.csv
├── results/
│   ├── figures/
│   ├── tables/
│   ├── logs/
│   └── predictions/
├── models/
├── checkpoints/
└── final_package/
```

### Important generated tables

| File | Purpose |
|---|---|
| `results/tables/dataset_counts.csv` | Counts by split and label |
| `results/tables/main_experiment_results.csv` | Six primary experiment trials |
| `results/tables/overall_aggregated_results.csv` | Per-backbone mean and standard deviation summaries |
| `results/tables/hyperparameter_results.csv` | DINOv2 reference-budget and $k$ sensitivity |
| `results/tables/ablation_results.csv` | Seed-13 calibration ablation |
| `results/tables/robustness_results.csv` | Clean and corrupted test conditions |
| `results/tables/robustness_summary.csv` | Grouped robustness summary |

### Important prediction output

```text
results/predictions/clean_test_predictions.csv
```

This file contains per-test-image anomaly scores and calibrated probabilities for all main trials.

### Important figures

The script generates, among others:

```text
results/figures/
├── dataset_class_distribution.png
├── representative_normal_samples.png
├── representative_anomaly_samples.png
├── clean_auroc_by_backbone.png
├── hyperparameter_auroc_pcb1.png
├── robustness_curve_resnet50.png
├── robustness_curve_dinov2_vits14.png
├── calibration_stress_resnet50.png
├── calibration_stress_dinov2_vits14.png
├── reliability_curve_dino_pcb1.png
├── false_positives_pcb1.png
└── false_negatives_pcb1.png
```

---

## Feature Cache

For each backbone, clean embeddings are cached as a compressed NumPy archive plus a metadata CSV:

```text
results/features_<backbone>.npz
results/features_<backbone>_meta.csv
```

The cache stores:

- normalized feature vectors;
- numeric labels;
- corresponding image paths.

This cache reuses clean representations across the main experiment, hyperparameter sensitivity analysis, and other downstream stages.

---
