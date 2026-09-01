# Pulmonary Nodule Detection and Classification using YOLO11n

A research/educational object-detection project for detecting and classifying pulmonary nodules in CT slices using **YOLO11n (Nano)**.

## Classes

The model predicts three classes:

- **benign**
- **equivocal**
- **malignant**

## Project structure

```text
Pulmonary_Nodule_Detection_GitHub/
├── notebooks/
│   └── Pulmonary_Nodule_Detection_Final.ipynb
├── app/
│   └── app.py
├── results/
│   ├── confusion_matrix.png
│   ├── confusion_matrix_normalized.png
│   ├── BoxPR_curve.png
│   ├── BoxF1_curve.png
│   ├── BoxP_curve.png
│   └── BoxR_curve.png
├── examples/
│   ├── dataset_annotations.png
│   ├── test_predictions.png
│   └── single_image_inference.png
├── weights/
│   └── README.md
├── requirements.txt
└── .gitignore
```

## Dataset

The project uses the prepared **LIDC-IDRI CT Lung Detection** object-detection dataset from Roboflow:

https://universe.roboflow.com/lidcidri-detection/lidc-idri-ct-lung-detection-2/dataset/1

The dataset is not included in this repository because it contains thousands of CT images.

The prepared dataset is expected to have the following structure:

```text
LIDC-IDRI-CT-Lung-Detection-2-1/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

The dataset split contains:

- Train: **16,354 images**
- Validation: **2,672 images**
- Test: **1,337 images**

## Model

- Architecture: **YOLO11n (Nano)**
- Task: Object detection
- Classes: 3
- Training epochs: **50**
- Image size: **640 × 640**
- Batch size: **16**

The notebook contains the training configuration, but training is disabled by default:

```python
RUN_TRAINING = False
```

This prevents an accidental 50-epoch training run when the notebook is opened.

## Trained weights

The trained `best.pt` checkpoint is intentionally **not committed to this repository** because model weights are large.

Place the checkpoint here for the standalone application:

```text
weights/best.pt
```

Alternatively, set the `MODEL_PATH` environment variable to another location.

## Evaluation

The final test-set evaluation produced approximately:

| Metric | Result |
|---|---:|
| Precision | **86.64%** |
| Recall | **72.12%** |
| mAP@50 | **81.85%** |
| mAP@50–95 | **51.59%** |

Per-class mAP@50–95:

| Class | mAP@50–95 |
|---|---:|
| Benign | 38.57% |
| Equivocal | 44.89% |
| Malignant | 71.31% |

The repository includes the generated evaluation plots in `results/`.

## Running the notebook

The notebook was developed for Google Colab.

1. Open `notebooks/Pulmonary_Nodule_Detection_Final.ipynb` in Google Colab.
2. Mount Google Drive.
3. Place the prepared dataset at the configured dataset path.
4. Place the trained checkpoint at the configured model path.
5. Run the notebook cells in order.

For a different environment, set:

```text
DATASET_PATH
PROJECT_ROOT
MODEL_PATH
```

as environment variables.

## Running the web application

From the repository root:

```bash
pip install -r requirements.txt
python app/app.py
```

By default, the application looks for:

```text
weights/best.pt
```

To use another checkpoint:

```bash
MODEL_PATH=/path/to/best.pt python app/app.py
```

For a public Gradio share link, set:

```bash
GRADIO_SHARE=true MODEL_PATH=/path/to/best.pt python app/app.py
```

The application allows the user to:

1. Upload a CT image.
2. Select a confidence threshold.
3. Run YOLO11n inference.
4. View detected bounding boxes.
5. View the predicted class and confidence.

## Confidence threshold

The confidence threshold is the minimum confidence required for a detection to be displayed.

For example, with a threshold of `0.50`, detections below 50% confidence are filtered out.

A higher threshold generally produces fewer, more confident detections; a lower threshold allows more detections but may also include weaker predictions.

## Dataset split limitation

The prepared dataset is organized at the **image level**, and the notebook audits patient identifiers across the splits.

The audit found patients appearing in more than one split. This means the test set should not be interpreted as a strictly patient-independent evaluation. The reported metrics describe performance on the provided image-level test split.

## Important limitation

This is a **research/educational prototype**. It is not a medical diagnostic tool and must not be used for clinical decision-making.

