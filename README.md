# Dataset Builder (Image Selection Pipeline)

A scalable and automated tool for building balanced image datasets from large local image repositories. It prepares datasets for machine learning tasks by applying structured filtering, proportional sampling, and size constraints across categories and subfolders.

---

## Overview

This project constructs a clean and balanced dataset from a hierarchical folder structure of images. It ensures uniform representation across categories while preventing overrepresentation from large folders and eliminating extremely small datasets.

It is particularly useful for:

* Deep learning dataset preparation
* Image classification training pipelines
* CLIP-style dataset construction
* Large-scale vision research workflows

---

## Dataset Structure

The expected input format is:

```
BASE_PATH/
    _<category_folder>/
        <image files>
```

Each folder belongs to a semantic category, grouped into the following high-level domains:

* Animals
* Nature & Weather
* Food & Drink
* Sports & Activities
* Urban & Architecture
* People & Portraits
* Objects & Technology
* Art & Culture

---

## Dataset Selection Rules

To ensure dataset quality and balance, the following rules are applied per folder:

* **Less than 2000 images** → Folder is skipped (replacement prompted)
* **2000 to 5000 images** → All images are retained
* **More than 5000 images** → Randomly sampled down to 5000 images

---

## Sampling Strategy

### Zone-Based Sampling

Large folders are split into equal-sized zones. Images are then randomly sampled from each zone to preserve diversity and avoid bias toward dense clusters.

### Proportional Subfolder Allocation

When a category contains multiple subfolders, image selection is distributed proportionally based on folder size using a largest-remainder allocation method. This guarantees fair representation across all sources.

---

## Output Structure

Processed datasets are saved to:

```
OUTPUT_BASE/<category_name>/
```

Each image is copied with duplicate-safe naming to avoid collisions.

---

## Generated Outputs

After execution, the system produces:

### 1. CSV Summary

A structured dataset report containing:

* Folder name
* Category
* Number of images found
* Number of images selected
* Selection percentage
* Processing status

### 2. Visualization Charts

* **Per-folder distribution chart**

  * Shows original vs selected images
  * Highlights min/max thresholds

* **Category summary chart**

  * Total images per category
  * Number of contributing folders

---

## Key Features

✔ Automated dataset construction

✔ Hierarchical folder support

✔ Balanced sampling across subfolders

✔ Proportional allocation strategy

✔ Zone-based diversity sampling

✔ Minimum and maximum dataset enforcement

✔ Interactive replacement for insufficient folders

✔ Safe file copying with duplicate handling

✔ Dataset analytics (CSV + charts)

---

## Configuration

Key parameters in the script:

* `MIN_IMAGES = 2000`
* `MAX_IMAGES = 5000`
* Supported formats:
  `.jpg, .jpeg, .png, .gif, .bmp, .webp, .tiff, .heic`

---

## Example Output Summary

During execution, the script prints:

* Folder processing status
* Image counts per category
* Selected vs skipped folders
* Replacement mappings (if triggered)

---

## Purpose

This tool ensures that large-scale image datasets are:

* Cleanly structured
* Statistically balanced
* Diversity-preserving
* Ready for machine learning training pipelines

---

## License

This project is intended for research and educational use. Modify freely for personal or academic datasets.
