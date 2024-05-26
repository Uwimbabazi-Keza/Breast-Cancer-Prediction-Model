# Breast Cancer Prediction Model using Breast Ultrasound Images Dataset

## Problem Statement and Dataset Description

Breast cancer is a significant cause of mortality among women globally, emphasizing the importance of early detection to reduce mortality rates. This project aims to develop a predictive model for breast cancer using machine learning techniques applied to breast ultrasound images.

### Dataset Description

The dataset used for this project is the Breast Ultrasound Images Dataset, from Kaggle (https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset). 

The dataset contains medical images of breast cancer obtained through ultrasound scans. The dataset contains images classified into three classes: normal, benign, and malignant. The dataset was collected in 2018. It consists of 600 female patients and a total of 780 images. Each image has an average size of 500x500 pixels and is in PNG format. Ground truth images are provided alongside the original images, aiding in classification tasks.

Reference: Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. Data in Brief. 2020 Feb;28:104863. DOI: 10.1016/j.dib.2019.104863.

## Pre-trained Model Selection and Reason

- VGG16: Known for its simplicity and high performance on image classification tasks
- ResNet50: Uses residual learning to manage the vanishing gradient problem, leading to improved training performance for deeper networks.
- InceptionV3: Incorporates efficient computation with a balance between depth and width of the network

## Fine-Tuning Process

1. **Base Model Setup:** We start with a pre-trained model like VGG16, ResNet50, or InceptionV3.
2. **Freeze Pre-trained Weights:** We keep the pre-trained model's knowledge intact by not changing its weights during training.
3. **Input and Conversion:** We define the input shape (size) of our images and convert them from grayscale to color (RGB).
4. **Using Pre-trained Model:** We add the pre-trained model to our network, which helps us extract useful features from our images.
5. **Pooling:** We simplify the extracted features.
6. **Custom Layers:** We add our own layers on top of the pre-trained model to help it learn specific patterns relevant to our task.
7. **Output Layer:** We set up the final layer to give us predictions for the three classes: benign, malignant, and normal.
8. **Model Compilation:** We prepare the model for training by choosing an optimizer and a loss function.

## Evaluation Metrics

To assess the performance of the fine-tuned models, the following evaluation metrics will be employed:

- Accuracy: Measures the overall correctness of the model's predictions.
- Loss: Indicates the error of the model during training.
- Precision: Represents the ratio of true positive predictions to the total number of positive predictions.
- Recall: Measures the ability of the model to identify all relevant instances.
- F1 Score: Mean of precision and recall, providing a balanced measure between the two metrics.

## Experiment Findings

- VGG16 achieved the highest accuracy (78.21%) and performed well across all metrics, making it the most balanced model.
- InceptionV3 also performed well, with a high accuracy (76.92%) and strong precision and recall.
- ResNet50 had the lowest accuracy (67.95%) and lower precision, recall, and F1 score. It may be less effective for this project.
- Overall, VGG16 and InceptionV3 performed well.

## Fine-tuned Models Evaluation in a table

| Model       | Accuracy | Loss    | Precision | Recall | F1 Score |
|-------------|----------|---------|-----------|--------|----------|
| VGG16       | 0.7821   | 0.5407  | 0.83      | 0.78   | 0.76     |
| ResNet50    | 0.6795   | 0.8224  | 0.68      | 0.68   | 0.64     |
| InceptionV3 | 0.7692   | 0.5692  | 0.80      | 0.77   | 0.76     |
