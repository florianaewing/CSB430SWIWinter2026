## Overview

The notebook walks through a complete deep learning workflow:

1. Loading and preprocessing the Sign Language MNIST dataset  
2. Exploratory Data Analysis (EDA) and visualization  
3. Training a baseline neural network  
4. Implementing and evaluating optimized models  
5. Comparing the impact of optimizers and regularization  
6. Generating accuracy curves, confusion matrices, and classification reports  
7. Reflecting on findings and identifying improvement strategies  

This project is intended for learners who want to better understand how different training strategies influence neural network performance.

---

## Dataset

The project uses the **Sign Language MNIST** dataset, downloaded automatically via the KaggleHub API:

- 27,455 training images
- 7,172 test images
- 28×28 grayscale images
- 24 gesture classes (letters A–Y except J and Z)

The dataset is downloaded automatically when running the notebook, so no manual steps are required.

---

## Environment Setup

This project is designed to run in **Google Colab**, and all dependencies are handled within the notebook.

Required libraries include:

- Python 3  
- TensorFlow / Keras  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- scikit-learn  
- kagglehub  

All installations are done directly in the notebook if not already available.

---

---

## Dataset

The project uses the [Sign Language MNIST dataset](https://www.kaggle.com/datamunge/sign-language-mnist). The dataset is automatically downloaded using `kagglehub` in the code.

---

## Environment Setup

### Using Conda

To create a fully reproducible Conda environment:

```bash
conda env create -f environment.yml
conda activate sign_language_mnist
```

Using pip in an existing python environment:

```bash
pip install -r requirements.txt
```

# Sign Language MNIST Classification

This repository contains code for training and evaluating deep learning models (fully connected and convolutional neural networks) on the **Sign Language MNIST** dataset. It includes data preprocessing, visualization, model training with multiple optimizers, and evaluation with confusion matrices and classification reports.

## How to Run the Notebook

Follow these steps to run the project:

### 1. Open the Notebook  
Upload the `.ipynb` file to Google Colab or open it from your Google Drive.

### 2. Enable GPU (recommended)
In Colab:

### 3. Run all cells

### 4. Review outputs  

## Results Summary

A full discussion is provided in the notebook’s reflection section.
