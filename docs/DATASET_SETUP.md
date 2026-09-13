# Dataset Setup

Datasets are **strictly external** and are **NOT** committed to this Git repository due to their size and licensing restrictions.

## General Guidelines
1. Do not commit downloaded ZIP files, CSVs with thousands of records, or raw media.
2. The `data/` folder is listed in `.gitignore` to prevent accidental uploads.
3. You must independently download the datasets required for the modality you are training.

## Expected Directory Structure
Locally, your `data/` folder should look similar to this:
```
data/
  raw/
    ntire2026/
    asvspoof2019/
  processed/
    image_ntire/
    audio_asvspoof/
```

## How to Prepare a Dataset
If a team member clones this repository and wants to train a model:

1. **Obtain Data:** Download the dataset from the official source (e.g., ASVspoof 2019 LA is available through the ASVspoof challenge portal, NTIRE datasets via their respective competitions).
2. **Place in Raw:** Extract the data into `data/raw/<dataset_name>`.
3. **Run Preparation Script:** Use the corresponding script in `tools/` or `training/` to parse the metadata and structure it.
   - Example: `python training/prepare_ntire_dataset.py`
4. **Manifests:** The preparation scripts usually generate a `manifest.csv` which maps file paths to labels, avoiding the need to reorganize massive folders physically.

## Important Scientific Directives
- **Do not fabricate labels.** Always use the ground truth provided by the dataset publishers.
- **Do not modify official evaluation data.** The `test` or `eval` splits must remain untouched to ensure scientific integrity and fair comparison with state-of-the-art benchmarks.
