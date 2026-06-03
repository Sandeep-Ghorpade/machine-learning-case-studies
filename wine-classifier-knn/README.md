# Wine Classifier using K-Nearest Neighbors (KNN)

## Overview

The Wine Classifier is a machine learning project that classifies wines into different categories based on their chemical properties using the K-Nearest Neighbors (KNN) algorithm. The project demonstrates the complete machine learning workflow, including data preprocessing, feature scaling, hyperparameter tuning, model training, evaluation, and visualization.

## Features

* Loads and preprocesses the wine dataset
* Handles missing values
* Separates independent and dependent variables
* Splits data into training and testing sets
* Applies feature scaling using StandardScaler
* Performs K-value tuning to find the optimal number of neighbors
* Visualizes K Values vs Accuracy using Matplotlib
* Trains the final KNN model using the best K value
* Evaluates model performance using:

  * Accuracy Score
  * Confusion Matrix
  * Classification Report

## Project Structure

```text
wine-classifier-kNN/
│
├── src/
│   └── wine_classifier_knn.py
│
├── dataset/
│   └── WinePredictor.csv
│
├── screenshots/
│   └── k_values_vs_accuracy.png
│
├── requirements.txt
└── README.md
```

## Technologies Used

* Python
* Pandas
* Matplotlib
* Scikit-learn

## Visualization

### K Values vs Accuracy

This graph helps identify the optimal K value that provides the best classification accuracy for the model.
