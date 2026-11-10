# bangladeshi-fish-detection-yolo11
YOLO11s-based object detection model for identifying 33 Bangladeshi fish classes. Trained on a Roboflow dataset using Python and Ultralytics YOLO. Includes model evaluation, test results, class-wise performance analysis, and prediction visualizations.
# Bangladeshi Fish Detection Using YOLO11s

An object detection project to identify and localize 33 different fish classes found in Bangladesh using the YOLO11s model. The model was trained using a Roboflow dataset and evaluated on a separate test set.

## Project Overview

- **Project Type:** Object Detection
- **Model:** YOLO11s
- **Dataset:** Bangladeshi Delta Fish Dataset
- **Number of Classes:** 33
- **Framework:** Ultralytics YOLO
- **Training Platform:** Kaggle
- **GPU:** NVIDIA Tesla T4
- **Image Size:** 640 × 640
- **Epochs:** 50
- **Batch Size:** 16

## Dataset

The dataset was collected from Roboflow and exported in YOLO format.

It contains 33 fish categories and is divided into three subsets:

- Training set
- Validation set
- Test set

### Fish Classes

Batashi, Bele, Chanda, Chapila, Chela, Foli, Ilish, Kajuli, Kayakanta, Koi, Magur, Mola, Mrigel, Pabda, Pangas, Puti, Rui, Rupchada, Shol, Sorputi, Taki, Tarabaim, Telapia, Tengra, Veda, Vetki, ayer, boyal, chitol, kalbaus, katol, kholshe, shing.

**Note:** The dataset is not included in this repository. Please refer to the original Roboflow project for dataset access and licensing information.

## Model Training

A pretrained YOLO11s model was fine-tuned on the custom fish detection dataset.

```python
from ultralytics import YOLO

model = YOLO("yolo11s.pt")

results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="fish-yolo11s",
    device=0
)
```

- **Total Training Time:** Approximately 8,178 seconds
- **Best Model Weights:** `best.pt`

## Test Set Evaluation

The best-trained model was evaluated on the independent test dataset.

| Evaluation Metric | Result |
|---|---:|
| Precision | 0.8055 |
| Recall | 0.7720 |
| mAP@0.50 | 0.828 |
| mAP@0.50:0.95 | 0.621 |
| Test Images | 455 |
| Test Instances | 601 |

## Class-wise Performance Analysis

### Top 3 Performing Classes

| Fish Class | AP@0.50:0.95 |
|---|---:|
| Mola | 0.8873 |
| Chanda | 0.8782 |
| Chapila | 0.8747 |

### Bottom 3 Performing Classes

| Fish Class | AP@0.50:0.95 |
|---|---:|
| Pabda | 0.1787 |
| Chela | 0.2691 |
| Shing | 0.2871 |

The results show that model performance varies across fish categories. Classes with lower AP may require further investigation, additional data, or improved annotations.

## Visualizations

The project includes evaluation and prediction visualizations such as:

- Prediction examples with bounding boxes
- Confusion matrix
- Normalized confusion matrix
- Precision-Recall analysis
- Training and validation performance curves
- Class-wise performance comparison

## Repository Structure

```text
bangladeshi-fish-detection-yolo11/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── notebooks/
│   └── fish-detection-yolo11s.ipynb
│
└── assets/
    ├── prediction-examples.png
    ├── confusion-matrix.png
    ├── confusion-matrix-normalized.png
    └── train-validation-loss.png
```

## Technologies Used

- Python
- Ultralytics YOLO11
- Roboflow
- OpenCV
- Supervision
- Matplotlib
- Pandas
- Kaggle

## How to Run

1. Clone this repository.
2. Install the required Python libraries.
3. Open the notebook in Kaggle or Jupyter Notebook.
4. Download or connect the dataset from Roboflow.
5. Configure the dataset path and `data.yaml`.
6. Run the notebook cells in sequence to train and evaluate the model.

### Install Dependencies

```bash
pip install ultralytics roboflow opencv-python supervision matplotlib pandas
```

### Run Prediction

```python
from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(
    source="path/to/image.jpg",
    imgsz=640,
    conf=0.25
)
```

## Future Improvements

- Improve detection performance for low-performing fish classes.
- Investigate class imbalance and annotation quality.
- Experiment with data augmentation and model configurations.
- Explore real-time fish detection applications.

## Acknowledgements

- **Dataset:** Roboflow — Bangladeshi Delta Fish Dataset
- **Object Detection Framework:** Ultralytics YOLO

