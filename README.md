# Physics-Guided Single-Image Dehazing with Dense Beta Estimation

## Project overview

This project investigates single-image dehazing using a hybrid physics-guided deep-learning framework. Rather than treating haze removal only as an image-to-image translation task, the model explicitly estimates the physical quantities that govern atmospheric scattering:

\[
I(x) = J(x)t(x) + A(1-t(x)), \qquad t(x) = e^{-\beta(x)d(x)}
\]

where `I` is the observed hazy image, `J` is the recovered clean image, `A` is atmospheric light, `d` is depth, `t` is transmission, and `beta` is the spatial atmospheric-scattering coefficient.

The implementation predicts a dense beta map and atmospheric light from a ViT-S/16 feature encoder, combines these predictions with estimated depth for physics-based inversion, and applies a residual restoration branch to refine the recovered image. The project evaluates both the direct physical reconstruction and the refined final output.

## Objectives

1. Estimate spatially varying haze parameters from a single hazy image.
2. Restore clean images using the atmospheric scattering model.
3. Compare direct physics-based restoration with residual-refined restoration.
4. Assess generalization on the NHD, RESIDE_NHD, and reside-6k datasets.
5. Measure image quality and beta-estimation quality using standard quantitative metrics.

## Datasets

| Dataset | Role in the project | Available data |
|---|---|---|
| NHD | Training, validation, and primary evaluation | Paired hazy/clean images, beta maps, depth maps, and transmission maps |
| RESIDE_NHD | External restoration evaluation | 200 paired hazy and clean images |
| reside-6k | External restoration evaluation | Paired evaluation images, subject to local availability |

### NHD experimental split

The NHD data are divided using disjoint clean-image identities so that a scene used for training is not reused in validation or testing.

| Partition | Number of paired images |
|---|---:|
| Training | 555 |
| Validation | 70 |
| Test | 70 |

The selected NHD haze-generation strategies are `uniform`, `fog_bank`, `perlin`, `radial`, and `depth_dep`. Five incomplete training variants were excluded because their corresponding depth or transmission files were unavailable.

### Data access and configuration

The project notebook is `nhdv2.ipynb`. Before execution, set the dataset-root variables in the notebook to the local locations provided by the project owner. The expected datasets are:

- NHD: training, validation, and primary testing.
- RESIDE_NHD: external paired testing.
- reside-6k: external paired testing, if the folder has been populated.

The `reside-6k` evaluation should be run only after confirming that the required files are present locally. Beta-map evaluation is limited to NHD because external datasets do not provide corresponding beta ground truth.

## Software requirements

- Python 3.10 or later
- PyTorch and TorchVision
- CUDA-enabled GPU recommended for training
- JupyterLab or Jupyter Notebook

Install the required packages in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install jupyterlab notebook ipykernel numpy pandas matplotlib pillow opencv-python scikit-image scipy tqdm
pip install torch torchvision
pip install transformers timm lpips
```

For GPU execution, install the PyTorch build that matches the operating system and available CUDA version.

## Execution procedure

### 1. Verify the dataset structure

Confirm that the NHD metadata correctly links each hazy image with its clean reference, beta map, depth map, and transmission map. For external testing, confirm that each hazy image has a matching clean target.

### 2. Configure paths

Open `nhdv2.ipynb` and update the dataset paths if required. Ensure that the NHD, RESIDE_NHD, and reside-6k roots point to the correct local directories.

### 3. Launch the notebook

```bash
jupyter lab nhdv2.ipynb
```

### 4. Execute cells sequentially

1. Import packages and initialise random seeds.
2. Load NHD metadata and remove incomplete records.
3. Create clean-image-disjoint training, validation, and test splits.
4. Build PyTorch datasets and dataloaders.
5. Initialise the ViT-S/16 encoder and depth-estimation component.
6. Train the `EndToEndPhysicsDehazer` model.
7. Evaluate direct physics-based and residual-refined outputs on NHD.
8. Evaluate restoration performance on RESIDE_NHD and reside-6k when available.
9. Save trained checkpoints, restored images, beta maps, qualitative comparisons, and metric summaries.

## Evaluation measures

Image restoration quality is assessed using:

- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index (SSIM)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

For NHD beta-map prediction, the project also computes:

- Beta MAE and beta RMSE
- Pearson correlation
- Gradient correlation

All datasets should be reported separately. Image resizing, normalization, data pairing, random seeds, and checkpoint selection should remain fixed across comparative runs.

## Expected outputs

The execution produces:

- Trained model checkpoints
- Predicted beta maps and atmospheric-light estimates
- Direct physics-based dehazed images
- Residual-refined dehazed images
- Per-image and aggregate evaluation metrics
- Qualitative comparison panels containing hazy input, clean reference, predicted beta map, direct reconstruction, and final output

## Reproducibility notes

- Preserve the random seed and clean-image-disjoint split definition.
- Record Python, PyTorch, CUDA, and GPU versions for each experiment.
- Save the complete training configuration with each model checkpoint.
- Verify all image pairs before computing metrics.
- Avoid combining scores from different datasets into a single value unless the weighting procedure is explicitly stated.
