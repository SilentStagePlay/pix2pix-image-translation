# Histopathology Nuclei Segmentation with Pix2Pix

**PyTorch · Conditional GAN · Semantic segmentation**

An exploration of paired image-to-image translation for segmenting nuclei in H&E histopathology images. A residual fully convolutional generator predicts a single-channel nuclei map from an RGB image, supervised by a conditional discriminator and an L1 reconstruction objective. The experimental comparison uses a supervised U-Net baseline.

**Repository:** [SilentStagePlay/pix2pix-image-translation](https://github.com/SilentStagePlay/pix2pix-image-translation)

**中文简介：** 本项目将 Pix2Pix 条件对抗学习用于病理图像细胞核语义分割，采用残差 FCN 生成器，并与 U-Net 对照。仓库保留自定义实验代码、参考工程、部分数据及 U-Net 输出。

## Project status

This repository is a research archive that needs additional preparation for reproducible training and inference. The custom implementation is in [`mypix/`](mypix/); the other model directories contain separate reference implementations with different interfaces and defaults.

The repository structure and source were reviewed at commit [`a9e93a7`](https://github.com/SilentStagePlay/pix2pix-image-translation/tree/a9e93a7a6246830eb088bd3ed824960c5f91b7af) on **19 October 2026**. The code review was static; no model training, inference, or metric reproduction was performed.

- Available: custom model/training/prediction source, MATLAB preprocessing and evaluation helpers, 30 source JPG images and 30 corresponding mask PNGs in `code/`, and archived U-Net prediction images.
- Missing: the complete processed training/test split with manifests, model checkpoints, a confirmed standalone U-Net baseline training configuration, and a pinned environment for `mypix`.

## Scope of the work

The project covers paired image/mask preparation, overlapping patch extraction, geometric augmentation, a residual generator and conditional discriminator, and a quantitative comparison against U-Net using pixel accuracy and foreground IoU. It demonstrates an experimental workflow spanning Python/PyTorch and MATLAB.

The task is **binary semantic segmentation** of nuclei foreground. Instance separation, cell counting, clinical validation, and deployment are outside the demonstrated scope. This is an educational research project.

## Method

### Generator and discriminator

[`mypix/networks.py`](mypix/networks.py) implements a **residual FCN generator**:

```text
RGB image [3 × 512 × 512]
  → 7×7 convolution, 64 channels
  → 3 downsampling stages: 128 → 256 → 512 channels
  → 9 residual blocks at 64 × 64 spatial resolution
  → 3 transposed-convolution upsampling stages: 256 → 128 → 64 channels
  → 7×7 convolution + Tanh
Generated nuclei map [1 × 512 × 512]
```

The generator uses reflection padding, batch normalization, and ReLU activations. Its Tanh output is a continuous map; a documented conversion and threshold are needed to obtain a binary mask.

The conditional discriminator concatenates the RGB image and the real/generated mask into a four-channel input. Four stride-2, 4×4 convolutions with batch normalization and LeakyReLU are followed by a 1×1 output convolution. A 512×512 input produces a **1×32×32 score map**. Although named `PixelDiscriminator` in the source, it follows a spatial PatchGAN-style design; this implementation should not be described as the standard 70×70 PatchGAN.

### Training objective

[`mypix/train.py`](mypix/train.py) combines a least-squares adversarial objective with L1 reconstruction:

```text
L_G = 0.01 × MSE(D(x, G(x)), 1) + 0.99 × L1(G(x), y)
L_D = 0.5 × [MSE(D(x, y), 1) + MSE(D(x, G(x).detach()), 0)]
```

Here, `x` is the RGB image and `y` is the reference nuclei mask. These are the weights used by this project; a generic Pix2Pix command with different loss weights or a U-Net generator does not reproduce this configuration.

The archived experiment configuration specifies Adam with an initial learning rate of **0.0002** and **200 epochs** for both models. The U-Net comparison uses binary cross-entropy. The current custom training script defaults to batch size **8** and Adam betas **(0.5, 0.999)**; these defaults alone do not establish the exact historical run configuration. Its `--decay_epoch` argument is declared but is not used by a learning-rate scheduler.

## Dataset and preparation

The archived experimental data preparation is summarized below. The complete processed dataset and split manifests are not supplied in the repository.

| Stage | Experimental setup |
| --- | --- |
| Source images | 30 TCGA-derived H&E images, 1000×1000 pixels each |
| Annotations | Expert nuclei masks; approximately 21,000 cells across seven organs and 18 hospitals, as recorded in the experiment documentation |
| Patch extraction | 512×512 sliding windows with stride 122; 25 patches per image, 750 patches in total |
| Augmentation | Paired rotations, flips, skew/shear, distortion, and zoom using Augmentor; expanded to 6,000 image/mask pairs |
| Split | Random 70/30 image-level split after patch extraction and augmentation: 4,200 training and 1,800 test images |

An authoritative annotated dataset release name, download URL, and annotation version still need confirmation. TCGA provenance alone is insufficient to identify the annotated dataset; a particular benchmark name is therefore not assumed here. The repository contains 30 source JPGs in [`code/RGBimages/`](code/RGBimages/) and 30 mask PNGs with matching base names in [`code/binaryMasks/`](code/binaryMasks/). Their presence does not establish the dataset license or reconstruct the historical split.

**Split limitation:** overlapping crops and augmented variants from the same source image may occur in both splits. Patient- or slide-independent separation is not established. The reported comparison should be interpreted within this protocol; it does not demonstrate performance on independent patients, slides, or institutions. A future experiment should split source cases before generating training patches and augmentations, and report its results separately.

For `mypix`, RGB tensors are normalized using mean `(0.666, 0.465, 0.615)` and standard deviation `(0.179, 0.191, 0.154)`. The training loader crops a horizontally concatenated pair into an image on the left and a mask on the right, then converts the mask to binary. [`mypix/data_prepare.m`](mypix/data_prepare.m) contains the concatenation step.

The checked-in augmentation scripts differ from the 6,000-pair experimental setup: `mypix/augment(1).py` currently calls `sample(10)`, while `code/augment.py` calls `sample(20000)` with some different augmentation parameters. Neither script should be assumed to regenerate the archived experiment unchanged.

## Experimental results

The following values summarize the archived experimental results. They have not been recomputed from this repository.

| Model | Split | Pixel accuracy ↑ | Foreground IoU ↑ |
| --- | --- | ---: | ---: |
| Pix2Pix with residual FCN generator | Train | 0.9161 | 0.6877 |
| Pix2Pix with residual FCN generator | Test | **0.9060** | **0.6877** |
| U-Net | Train | 0.8915 | 0.5906 |
| U-Net | Test | 0.8834 | 0.6015 |

On the reported test split, the differences are **+2.26 percentage points in pixel accuracy** and **+8.62 percentage points in IoU**. Both recorded Pix2Pix IoU entries are `0.6877`; these values are retained as recorded. No repeated-run uncertainty or confidence intervals are supplied.

Pixel accuracy counts correct foreground and background pixels. Foreground IoU is `|prediction ∩ target| / |prediction ∪ target|`. The archived evaluation describes averaging image-level measurements, but a reproducible evaluation still needs the exact prediction-to-mask conversion, thresholding, file matching, empty-mask policy, and aggregation procedure.

**Evaluation caveat:** [`code/calculate_acc_dice.m`](code/calculate_acc_dice.m) names its overlap variable `dice`, but computes intersection divided by union: **IoU/Jaccard, not Dice**. It uses MATLAB `imbinarize`, assumes 512×512 images, matches masks and predictions by directory order, and expects PNG predictions. The custom predictor instead saves normalized JPG images under numeric filenames. These steps need alignment before they can reproduce the table; the archived U-Net outputs alone do not verify the reported metrics.

## Repository map

```text
.
├── mypix/                              # Custom residual-FCN cGAN experiment
│   ├── networks.py                     # Generator and conditional discriminator
│   ├── train.py                        # Training entry point
│   ├── predict.py                      # Prediction entry point
│   ├── datasets_train.py               # Horizontally paired image/mask loader
│   ├── datasets_predict.py             # RGB prediction-image loader
│   ├── data_prepare.m                  # Image/mask concatenation
│   └── ...                             # Patch, augmentation, split, normalization helpers
├── code/
│   ├── RGBimages/                      # 30 source JPG images
│   ├── binaryMasks/                    # 30 corresponding mask PNGs
│   └── ...                             # MATLAB/Python preparation and metric helpers
├── NucleiSegmentation-master/
│   └── NucleiSegmentation-master/       # Separate cGAN nuclei-segmentation reference
├── pix2pix-master/                     # Original-style Torch/Lua Pix2Pix reference
├── pytorch-CycleGAN-and-pix2pix-master/
│   └── pytorch-CycleGAN-and-pix2pix-master/  # Separate PyTorch reference framework
├── Unet结果/Unet结果/                    # 6,000 archived PNG outputs
├── mask/                               # Additional mask artifacts
└── 数据增强/                            # Augmentation artifacts
```

The U-Net generators inside the reference frameworks are components of those frameworks. They do not establish the standalone BCE-trained U-Net comparison reported above. No trained checkpoint files were found in the inspected tree.

## Training and inference entry points

| Implementation | Train | Predict/test | Interpretation |
| --- | --- | --- | --- |
| Custom residual FCN implementation | `mypix/train.py` | `mypix/predict.py` | Implements the project model; preparation and fixes required |
| Nuclei segmentation reference | `NucleiSegmentation-master/NucleiSegmentation-master/train.py` | Same directory: `test.py` | Separate defaults, dependencies, and data conventions |
| PyTorch CycleGAN/Pix2Pix reference | `pytorch-CycleGAN-and-pix2pix-master/pytorch-CycleGAN-and-pix2pix-master/train.py` | Same directory: `test.py` | General reference framework with separate configuration |
| Torch/Lua Pix2Pix reference | `pix2pix-master/train.lua` | `pix2pix-master/test.lua` | Requires the Torch/Lua stack; separate from PyTorch |

The recorded experimental environment was **Ubuntu 16.04, Python 3.6, MATLAB R2018b, an NVIDIA GTX TITAN X, and 32 GB RAM**. The PyTorch version is unspecified. The custom Python files import PyTorch, torchvision, NumPy, and Pillow; preprocessing additionally uses Augmentor and MATLAB. Requirements/environment files in the reference directories belong to those implementations and do not pin a tested environment for `mypix`.

After installing a compatible environment, inspect the custom command-line options from its own directory:

```bash
git clone https://github.com/SilentStagePlay/pix2pix-image-translation.git
cd pix2pix-image-translation/mypix
python train.py --help
python predict.py --help
```

These option-inspection commands are not a verified end-to-end run. The custom loader expects the following structure beneath the root passed to `--path` (the complete structure is not currently supplied):

```text
<experiment_root>/
├── Data/Nuclei_Segmentation/
│   ├── train/                  # Paired RGB | mask images, normally 1024×512
│   ├── val/                    # Paired images; historical held-out role needs confirmation
│   └── predict/                # RGB images only, normally 512×512
├── saved_model/Nuclei_Segmentation/
└── predict_results/pred_map/
```

The recorded experiment uses a 70/30 train/test split, while the custom loader calls the held-out directory `val`. The current training script also samples this directory for visualization. Its relationship to the final evaluation must be clarified before claiming an untouched test set.

Before attempting a run:

1. Replace the hard-coded preprocessing paths and supply a verified image/mask pairing and split manifest.
2. Create checkpoint/output directories; the scripts do not create all of them. Fix the resume paths in `train.py`: the leading `/saved_model/...` can discard the supplied root on the historical Linux platform.
3. Supply a compatible checkpoint and make checkpoint selection explicit. `predict.py` currently hard-codes `saved_model/Nuclei_Segmentation/generator_30.pth`; `--mod` does not select it. CPU/GPU loading also needs compatible `DataParallel` keys and device mapping.
4. Check library compatibility and output/mask conventions, then run a small training/inference smoke check. In `networks.py`, the condition `norm_type == 'batch' or 'spectral'` is always true and needs correction if normalization options are used.
5. Align prediction names/formats with the evaluator and document thresholding and metric aggregation. Preserve raw prediction maps for evaluation instead of relying on per-image normalized JPG previews.

Once those prerequisites are addressed, the existing training interface takes the following form:

```bash
# Template only: replace the path with a prepared experiment root.
python train.py --path /absolute/path/to/experiment_root --dataset_name Nuclei_Segmentation --n_epochs 200 --batch_size 8

# Requires a compatible checkpoint and corrected/confirmed checkpoint selection.
python predict.py --path /absolute/path/to/experiment_root --dataset_name Nuclei_Segmentation
```

## What is needed for reproducibility

- **Environment:** exact Python, PyTorch, torchvision, NumPy, Pillow, Augmentor, MATLAB, CUDA/cuDNN, driver, OS, and GPU versions; a tested lock file or container and a complete run command.
- **Data provenance:** annotated dataset name, authoritative source/download URL, release/version, annotation origin, and applicable access/redistribution terms. Confirm the relationship of the included files to that release.
- **Data preparation:** source IDs and checksums, image/mask pairing, patch coordinates, augmentation settings and seeds, exact 4,200/1,800 manifests, and the original split unit. Specify whether `val` was used for visualization, model selection, or final reporting.
- **Weights:** generator and discriminator checkpoints with download location, checksum, architecture, normalization, training configuration, epoch, and any `DataParallel` wrapping. Recover the U-Net implementation/configuration and its corresponding weights as well.
- **Evaluation:** the exact checkpoint used for the reported results, raw predictions or a regeneration path, binary-mask conversion, thresholds, filename mapping, image-level averaging, empty-mask handling, and a script that reports both models on the same split.

Recovering the historical experiment and evaluating a new source-independent split are separate tasks. Results from a corrected or newly split experiment should be reported separately from the archived values above.

## Attribution and usage

This archive includes reference code. The original implementations and their authors are credited below:

- **Pix2Pix:** Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A. Efros, *Image-to-Image Translation with Conditional Adversarial Networks*, CVPR 2017. See the [original project](https://phillipi.github.io/pix2pix/) and [Torch implementation](https://github.com/phillipi/pix2pix). The local `pix2pix-master/` identifies this source.
- **PyTorch CycleGAN and Pix2Pix:** [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), credited in the bundled README to Jun-Yan Zhu and Taesung Park, with support from Tongzhou Wang.
- **Adversarial nuclei segmentation reference:** [faisalml/NucleiSegmentation](https://github.com/faisalml/NucleiSegmentation), credited in its bundled README to Faisal Mahmood and coauthors, with the associated [2018 paper](https://arxiv.org/abs/1810.00236).

The exact upstream revisions and the full derivation history of `mypix` are not recorded in the inspected files. This repository should not be presented as an entirely original implementation of all bundled frameworks.

There is **no repository-wide license file** in the inspected snapshot. License texts are included under [`pix2pix-master/LICENSE`](pix2pix-master/LICENSE) and [`pytorch-CycleGAN-and-pix2pix-master/pytorch-CycleGAN-and-pix2pix-master/LICENSE`](pytorch-CycleGAN-and-pix2pix-master/pytorch-CycleGAN-and-pix2pix-master/LICENSE); preserve the relevant notices and follow their terms. The bundled nuclei-segmentation README states MIT licensing but references a license file that is absent from that local directory. Confirm the applicable upstream notice before redistributing that component. Code-license statements do not establish permission to redistribute the image data or annotations.
