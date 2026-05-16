# README

## 0. Environment Setup

This project uses Python 3.12.

```bash
conda create -n anomalyvfm python=3.12 -y
conda activate anomalyvfm
pip install -r requirement.txt
```

Recommended project layout:

```text
.
├── mvtec_ad_2/
├── synthetic_dataset_flux_filter_dinov3/
├── backbone/
│   └── sam_vit_h_4b8939.pth
├── experiments/
│   └── tipsv2_2500/
│       └── model.pkl
├── results/
├── tipsv2-so400m14/
├── download_dataset.py
├── download_tipsv2.py
├── train_mvtec_ad2.py
├── test_mvtec_ad2.py
├── SAM-Finer.py
└── requirement.txt
```

---

## 1. Preparation

### 1.1 Prepare MVTec AD 2 Dataset

Assume the MVTec AD 2 dataset is placed under the current project directory:

```text
./mvtec_ad_2
```

For example:

```text
./mvtec_ad_2/
├── can/
├── fabric/
├── fruit_jelly/
├── ...
└── walnuts/
```

---



### 1.2 Download TIPSv2-SO400M/14 Model

The model used in this project is:

```text
google/tipsv2-so400m14
```

We provide a download script in the project root:

```text
download_tipsv2.py
```

Run:

```bash
python download_tipsv2.py
```

The model will be saved to:

```text
./tipsv2-so400m14
```

---

### 1.3 Download SAM Weight

The mask refinement stage uses SAM ViT-H:

```text
sam_vit_h_4b8939.pth
```

Create the backbone folder:

```bash
mkdir -p ./backbone
```

Download the SAM ViT-H checkpoint:

```bash
wget -O ./backbone/sam_vit_h_4b8939.pth \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

After downloading, check that the file exists:

```bash
ls -lh ./backbone/sam_vit_h_4b8939.pth
```

Expected path:

```text
./backbone/sam_vit_h_4b8939.pth
```

---


## 2. Quick Start: Use Provided Checkpoint

If you want to run the project quickly without training, please first download the pretrained checkpoint from the following link:

```text
https://drive.google.com/file/d/10f3cgYJp6XBIleOsYJeZXzt_7Qnl56k-/view?usp=sharing
```

After downloading, place the checkpoint at:

```text
./experiments/tipsv2_2500/model.pkl
```


Then jump to:

1. [3.2 Coarse Mask Generation](#32-coarse-mask-generation)
<!-- 2. [3.3 Mask Refinement](#33-mask-refinement)
3. [3.4 Convert Masks to TIFF16](#34-convert-masks-to-tiff16)
4. [3.5 Prepare Submission](#35-prepare-submission) -->

<!-- Quick inference commands:

```bash
python test_mvtec_ad2.py \
    --data-path ./mvtec_ad_2 \
    --model-path ./experiments/tipsv2_2500/model.pkl \
    --out-path ./results/anomaly_images_thresholded

python SAM-Finer.py \
    --bin_savedir ./results \
    --data_path ./mvtec_ad_2

python convert_to_tiff16.py \
    --input_dir ./results/anomaly_images_thresholded \
    --output_dir ./results/anomaly_images

python mvtec_public_code/check_and_prepare_data_for_upload.py \
    --submission_path ./results
```

Note: `./results/anomaly_images_thresholded` should not be renamed, because `SAM-Finer.py` reads coarse masks from this folder. -->

---

## 3. Standard Pipeline


### 3.0 Download AnomalyVFM Synthetic Dataset

This project uses the synthetic anomaly dataset from AnomalyVFM:

```text
synthetic_dataset_flux_filter_dinov3
```

Run the download script directly from the project root:

```bash
python download_dataset.py
```

After downloading, make sure the dataset folder is located under the current project directory:

```text
./synthetic_dataset_flux_filter_dinov3
```

This dataset is required for training. If you only want to run inference with the provided checkpoint, you can skip this step and start from [Quick Start](#2-quick-start-use-provided-checkpoint).

---


### 3.1 Training

Run:

```bash
python train_mvtec_ad2.py \
    --data-path ./synthetic_dataset_flux_filter_dinov3 \
    --out-path ./experiments/tipsv2_2500/
```

Arguments:

```text
--data-path
    Path to the AnomalyVFM synthetic dataset.
    Example:
    ./synthetic_dataset_flux_filter_dinov3

--out-path
    Directory for saving the trained PKL model and related training outputs.
    Example:
    ./experiments/tipsv2_2500/
```

After training, the expected model file is:

```text
./experiments/tipsv2_2500/model.pkl
```

---

### 3.2 Coarse Mask Generation

Run:

```bash
python test_mvtec_ad2.py \
    --data-path ./mvtec_ad_2 \
    --model-path ./experiments/tipsv2_2500/model.pkl \
    --out-path ./results/anomaly_images_thresholded \
    --save-images
```

Arguments:

```text
--data-path
    Path to the MVTec AD 2 dataset.
    Example:

--model-path
    Path to the trained or provided model checkpoint.
    Example:
    ./experiments/tipsv2_2500/model.pkl

--out-path
    Output directory for coarse anomaly masks.
    This folder name should not be changed.
    Example:
    ./results/anomaly_images_thresholded

--save-images
    Whether the generated results are stored in out-path
```

Expected output:

```text
./results/anomaly_images_thresholded
```

---

### 3.3 Mask Refinement

Use SAM-Finer to refine the coarse masks.

Run:

```bash
python SAM-Finer.py \
    --bin_savedir ./results \
    --data_path ./mvtec_ad_2
```

Arguments:

```text
--bin_savedir
    The parent directory of anomaly_images_thresholded.
    Use:
    ./results

--data_path
    Path to the MVTec AD 2 dataset.
    Use:
    ./mvtec_ad_2
```

<!-- Important:

```text
./results/anomaly_images_thresholded
```

is the coarse mask folder generated by `test_mvtec_ad2.py`.

`SAM-Finer.py` will read masks from:

```text
./results/anomaly_images_thresholded
```

and the refined masks will overwrite the original coarse mask results. -->

---

### 3.4 Convert Masks to TIFF16

Convert the refined masks to the required TIFF16 submission format.

Run:

```bash
python convert_to_tiff16.py \
    --input_dir ./results/anomaly_images_thresholded \
    --output_dir ./results/anomaly_images
```

Arguments:

```text
--input_dir
    Input directory containing the refined masks from the previous step.
    Use:
    ./results/anomaly_images_thresholded

--output_dir
    Output directory for the final masks to be submitted.
    Use:
    ./results/anomaly_images
```

Expected output:

```text
./results/anomaly_images
```

---

### 3.5 Prepare Submission

Use the MVTec public code checker to validate and package the submission.

Run:

```bash
python mvtec_public_code/check_and_prepare_data_for_upload.py ./results
```

Arguments:

```text
--submission_path
    The submission directory containing anomaly_images.
    Use:
    ./results
```

The prepared submission archive will be saved as:

```text
./result.tar.gz
```

---

## 4. Full Standard Pipeline Example

```bash
conda create -n anomalyvfm python=3.12 -y
conda activate anomalyvfm
pip install -r requirement.txt

python download_dataset.py
python download_tipsv2.py

mkdir -p ./backbone
wget -O ./backbone/sam_vit_h_4b8939.pth \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

python train_mvtec_ad2.py \
    --data-path ./synthetic_dataset_flux_filter_dinov3 \
    --out-path ./experiments/tipsv2_2500/

python test_mvtec_ad2.py \
    --data-path ./mvtec_ad_2 \
    --model-path ./experiments/tipsv2_2500/model.pkl \
    --out-path ./results/anomaly_images_thresholded \
    --save-images

python SAM-Finer.py \
    --bin_savedir ./results \
    --data_path ./mvtec_ad_2

python convert_to_tiff16.py \
    --input_dir ./results/anomaly_images_thresholded \
    --output_dir ./results/anomaly_images

python mvtec_public_code/check_and_prepare_data_for_upload.py ./results
```

---

## 5. Expected Outputs

After training:

```text
./experiments/tipsv2_2500/model.pkl
```

After coarse mask generation:

```text
./results/anomaly_images_thresholded
```

After SAM refinement:

```text
./results/anomaly_images_thresholded
```

The refined masks overwrite the original coarse masks.

After TIFF16 conversion:

```text
./results/anomaly_images
```

After preparing the submission:

```text
./result.tar.gz
```

[1]: https://github.com/MaticFuc/AnomalyVFM "GitHub - MaticFuc/AnomalyVFM"
[2]: https://huggingface.co/google/tipsv2-so400m14 "google/tipsv2-so400m14 · Hugging Face"
[3]: https://github.com/facebookresearch/segment-anything"segment-anything/README.md at main · facebookresearch/segment-anything · GitHub"

---

## Acknowledgements

We sincerely thank the authors and maintainers of the following open-source projects, models, datasets, and tools:

- [AnomalyVFM](https://github.com/MaticFuc/AnomalyVFM)
- [TIPSv2-SO400M/14](https://huggingface.co/google/tipsv2-so400m14)
- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [MVTec AD 2](https://www.mvtec.com/company/research/datasets/mvtec-ad-2)
- [Hugging Face](https://huggingface.co/)

All third-party components remain under their respective original licenses.