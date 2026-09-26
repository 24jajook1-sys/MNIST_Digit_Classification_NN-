# MNIST Digit Classification using Neural Networks

Classifying handwritten digits (0–9) on the MNIST dataset — progressing from a baseline dense network to a CNN to a data-augmented, batch-normalized CNN, with a full comparison and error analysis.

## Overview

This project builds three increasingly capable models on MNIST and compares them head-to-head:

- Exploratory data analysis (sample grids, class balance, pixel intensity distributions, average digit images)
- Three models: baseline dense network, a CNN, and an enhanced CNN trained with data augmentation
- Full evaluation: confusion matrices, per-model accuracy, misclassification analysis, visual prediction grid
- A saved, reloadable prediction system

## Dataset

**MNIST handwritten digits** (via `tensorflow.keras.datasets.mnist`)
- 60,000 training images, 10,000 test images
- 28×28 grayscale, 10 classes (digits 0–9)

## Project workflow

| Step | Section |
|------|---------|
| 1 | Setup & imports |
| 2 | Load the MNIST dataset |
| 3–7 | EDA (sample grid, class distribution, pixel intensity, average digit images) |
| 8 | Preprocessing (normalization, flat + reshaped versions) |
| 9 | Model 1 — baseline dense network |
| 10 | Model 2 — CNN |
| 11 | Data augmentation setup |
| 12 | Model 3 — enhanced CNN (BatchNorm + augmentation) |
| 13 | Training history visualization |
| 14 | Evaluate all models on the test set |
| 15 | Confusion matrices, model comparison chart, misclassification analysis |
| 16 | Visual prediction grid |
| 17 | Predictive system |
| 18 | Save & reload the best model |

## Models compared

| Model | Architecture | Test accuracy |
|---|---|---|
| Dense NN | Flatten → Dense(128) → Dense(64) → Dense(10) | 97.71% |
| CNN | 2× Conv2D/MaxPooling blocks → Dense(128) → Dense(10) | 99.17% |
| Enhanced CNN | 2 conv blocks with BatchNorm + Dropout, trained with rotation/zoom/shift augmentation | **99.60%** (best) |

## Tech stack

Python · NumPy · TensorFlow / Keras · Matplotlib · Seaborn · scikit-learn (metrics)

## Project structure

```
├── MNIST_Digit_Classification_NN.ipynb   # Main notebook (all 18 steps)
├── mnist_best_model.keras                # Saved enhanced CNN (generated on run)
└── README.md
```

## Getting started

```bash
pip install numpy tensorflow matplotlib seaborn scikit-learn
jupyter notebook MNIST_Digit_Classification_NN.ipynb
```

Run all cells top to bottom — MNIST loads directly through Keras, so no external download is needed.

## Author

Kush Shreeram Jajoo
