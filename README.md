# Underwater Label Data

This repository contains a sample dataset and tools for handling underwater segmentation tasks. The dataset and utilities are designed for point segmentation of underwater coral images, providing both labeled images and CSV files for annotations.

## Dataset Structure

The repository is organized as follows:

```plaintext
Underwater Label Data/ (GitHub repository)
 ├── Coral Dataset Sample
 │ ├── dataset/ 
 │ ├── images/ 
 │ │ ├── img1.jpg 
 │ │ └── img2.jpg
 │ ├── label_csv/ 
 │ │ ├── img1.csv
 │ │ └── img2.csv
 │ ├── label_img/ 
 │ │ ├── img1.tif
 │ │ └── img2.tif
 │ └── CPCe_code.csv 
 ├── dataset_sample.py 
 ├── label_preparation.ipynb 
 ├── label_generator.py 
 └── README.md 
```
## Dataset Details

### `Coral Dataset Sample/dataset/images/`
- Contains raw underwater images in various resolutions, stored in common formats (e.g., `.jpg`).
- Images capture diverse underwater scenes, serving as the base for segmentation tasks.

### `Coral Dataset Sample/dataset/label_csv/`
- Stores annotations as CSV files. Each file contains:
  - **x, y coordinates**: Specify the position of labeled points.
  - **Label name**: Indicates the type of object at each point (e.g., coral species).
  - **Class name**: Represents the classification of the object, linking it to a broader category.

### `Coral Dataset Sample/dataset/label_img/`
- Contains labeled images saved as `.tif` files with 16-bit depth.
- Pixel values range from:
  - **1 to 65525**: Represent different classes or object types.
  - **0**: Denotes unlabeled or background regions.

### `CPCe_code.csv`
- Maps numerical labels in `.tif` files to corresponding **class names** and **explanations**.
- Serves as a reference for understanding label details.

---

## Utilities Overview

### `label_generator.py`
This script includes utilities for managing and processing labels in both CSV and `.tif` formats:
- **Generate `.tif` Labels**: Converts CSV annotations into `.tif` label images.
- **Expand Label Regions**: Increases the size of labeled points for better visibility.
- **Resize Data**: Scales images and labels to match specific model requirements.
- **Split Dataset**: Divides the dataset into training and validation sets.
- **Overlay Visualization**: Creates images that overlay labels onto the original underwater images.

### `label_preparation.ipynb`
- A Jupyter Notebook that demonstrates:
  - Using `label_generator.py` to preprocess and prepare the dataset.
  - Generating overlay images for data inspection.

### `dataset_sample.py`
- Provides a PyTorch dataset class for handling the underwater dataset.
- Includes additional utility functions for:
  - Loading images and labels.
  - Visualizing data during training and validation.

---

## Citation
If you use this dataset for research or development, please cite it as follows:

Ouyang, B.; Farrington, S.; Kunapinun, A.; "Underwater Labelling Images Dataset for Segmentation and Classification," NOAA, FAU, 2025.

### BibTeX
```bibtex
@dataset{underwater2025,
  author = {Ouyang, B. and Farrington, S. and Kunapinun, A.},
  title = {Underwater Labelling Images Dataset for Segmentation and Classification},
  year = {2025},
  publisher = {NOAA, FAU},
}
```

---

We hope this dataset provides a valuable resource for your underwater image analysis projects!

