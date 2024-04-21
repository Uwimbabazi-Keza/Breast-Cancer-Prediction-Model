# Breast Cancer Detector

## Project Description
Mammo is a machine learning model designed to predict the likelihood of breast cancer based on various features extracted from breast biopsy samples. The model is a deep learning neural network trained on a dataset containing clinical measurements of cell nuclei for both benign and malignant samples.

Mammo aims to provide an early and accurate diagnosis tool for breast cancer, potentially aiding healthcare professionals in making informed decisions and improving patient outcomes.

### Model Architecture
Mammo uses a model that runs as a feed-forward neural network using TensorFlow's Keras API. It is defined in a Sequential model and consists of the following layers:

- Input layer with input shape of 30 (number of features)
- Dense layer with 16 units, ReLU activation
- Dense layer with 8 units, ReLU activation
- Output Dense layer with sigmoid activation

### Installation
