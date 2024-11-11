# ASL Recognition with XGBoost

## Overview

This project involves recognizing American Sign Language (ASL) characters using machine learning. We preprocess the dataset, handle class imbalances, and use the XGBoost classifier to train a model that can predict ASL letters based on image data. 

## Features

- **Data Preprocessing**: Cleaned the dataset to handle imbalances, removing classes with fewer than 2 samples.
- **Label Mapping**: Converted categorical labels to continuous integer labels for model compatibility.
- **Model Training**: Trained an XGBoost classifier to predict ASL characters.
- **Evaluation**: Used accuracy to evaluate the model's performance.
- **Model Saving**: Saved the trained model and label mapping for future inference.

## Requirements

- Python 3.7+
- Libraries:
  - `xgboost`
  - `numpy`
  - `sklearn`
  - `pickle`
  - `collections`
