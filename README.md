# Breast Cancer Detector

## Project Description
Mammo is a machine learning model designed to predict the likelihood of breast cancer based on various features extracted from breast biopsy samples. The model is a deep learning neural network trained on the [Breast Cancer Wisconsin Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/code) containing clinical measurements of cell nuclei for both benign and malignant samples.

Mammo aims to provide an early and accurate diagnosis tool for breast cancer, potentially aiding healthcare professionals in making informed decisions and improving patient outcomes.

## Model Architecture
Mammo uses a model that runs as a feed-forward neural network using TensorFlow's Keras API. It is defined in a Sequential model and consists of the following layers:

- Input layer with input shape of 30 (number of features)
- Dense layer with 16 units, ReLU activation
- Dense layer with 8 units, ReLU activation
- Output Dense layer with sigmoid activation

## Installation

Run:
```
>> git clone https://github.com/Uwimbabazi-Keza/Breast-Cancer-Prediction-Model
>> cd Breast-Cancer-Prediction-Model/src
>> pip install -r requirements.txt
>> python app.py
```

There are 30 input fields. For testing purposes, enter the sample data from  [Breast Cancer Wisconsin Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/code). Your output should read benign or malignant.

## Future Work
- **Mobile-Friendly Design:** Plan to develop a responsive design that will make the application accessible and user-friendly on mobile devices.
- **Additional Features:** Incorporate more features and functionalities, such as personalized recommendations, visualization of prediction results, and integration with other health-related services.
- **Business Strategy:** Conducting market research to target healthcare professionals, patients, and caregivers. Collaborating with medical entities to validate and integrate the app. Forming partnerships with health insurance firms for added-value services.

## Author
Uwimbabazi Keza

### Additional Info
Disclaimer: This application is for educational and informational purposes only. Always consult with a healthcare professional for accurate diagnosis and treatment.