# Dataset Documentation

## Original Dataset

This project uses a manually curated version of the publicly available **Diabetic Foot Ulcer (DFU) Dataset** from Kaggle.

Original dataset source:  
**Diabetic Foot Ulcer (DFU) Dataset — Kaggle**

The original Kaggle dataset was not used directly for model training. The images were manually reviewed and reorganized according to the requirements of this project.

## Dataset Preparation

The following preprocessing and organization steps were performed:

- Reviewed the available images manually.
- Removed unclear or unsuitable samples.
- Removed confusing samples such as callus or mixed-condition images where necessary.
- Organized the images into two classes:
  - `ulcer`
  - `nonulcer`
- Divided the curated images into training, validation, and test subsets.
- Used a separate test set for final model evaluation.
- Images were resized to `224 × 224` during model preprocessing.

## Curated Dataset Structure

```text
dataset4/
├── train/
│   ├── nonulcer/
│   └── ulcer/
├── val/
│   ├── nonulcer/
│   └── ulcer/
└── test/
    ├── nonulcer/
    └── ulcer/
```

## Dataset Distribution

| Split | Non-Ulcer | Ulcer | Total |
|---|---:|---:|---:|
| Training | 593 | 355 | 948 |
| Validation | 194 | 90 | 284 |
| Test | 135 | 71 | 206 |
| **Total** | **922** | **516** | **1,438** |

The archive contains image files in JPG, JPEG, PNG, WEBP, and GIF formats.

## Dataset Availability

The curated image dataset itself is not included in this repository because it was derived from an externally hosted public dataset.

The `dataset_manifest.csv` file documents the images and their assigned split/class in the curated dataset used for this project.

Users who want to reproduce the project should obtain the original DFU dataset from Kaggle and prepare it using the dataset organization described above.

## Note

This dataset organization represents the dataset used during the original EEE 4710 course project. Any future dataset-cleaning, deduplication, or revised train/validation/test splitting should be treated as a separate post-course improvement rather than as part of the original reported experimental results.