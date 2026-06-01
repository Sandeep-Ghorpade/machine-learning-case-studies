# Titanic Survival Predictor

## Project Overview

Titanic Survival Predictor is a Machine Learning project that predicts whether a passenger survived the Titanic disaster using Logistic Regression.

The project includes data preprocessing, feature engineering, model training, model evaluation, and model persistence using Joblib.

## Features

* Data Loading and Exploration
* Data Preprocessing
* Missing Value Handling
* Feature Encoding
* Train-Test Split
* Logistic Regression Model Training
* Model Evaluation
* Model Saving and Loading using Joblib

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib

## Project Structure

```text
Titanic-Survival-Predictor/
│
├── dataset/
│   └── Titanic.csv
│
├── src/
│   └── titanic_survival_predictor.py
│
├── README.md
└── requirements.txt
```

## Workflow

1. Load the Titanic dataset
2. Explore and analyze the data
3. Handle missing values
4. Remove unnecessary columns
5. Encode categorical features
6. Split data into training and testing sets
7. Train a Logistic Regression model
8. Save the trained model
9. Load the saved model
10. Evaluate model performance

## Model Performance

* Algorithm: Logistic Regression
* Accuracy: 76.72%

The model was evaluated using:

* Accuracy Score
* Confusion Matrix