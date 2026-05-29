# California Housing Price Prediction

Machine Learning project for predicting California housing prices using different regression algorithms.

---

##  Project Overview

This project uses multiple Machine Learning regression algorithms to predict housing prices based on various housing-related features from the California Housing dataset.

The project demonstrates:

* Data preprocessing
* Regression techniques
* Model training
* Model evaluation
* Ensemble learning concepts

---

##  Technologies Used

* Python
* Pandas
* Scikit-learn

---

##  Project Structure

```text
california-housing-price-prediction/
│
├── dataset/
│   └── california_housing.csv
│
├── src/
│   ├── california_housing_decision_tree.py
│   ├── california_housing_bagging_regression.py
│   └── california_housing_boosting_regression.py
│
├── README.md
└── requirements.txt
```

---

##  Machine Learning Algorithms Used

### 1. Decision Tree Regressor

* Tree-based regression algorithm
* Predicts housing prices using decision rules

### 2. Bagging Regressor

* Ensemble learning technique
* Combines multiple decision trees to improve prediction performance

### 3. Gradient Boosting Regressor

* Boosting algorithm
* Builds models sequentially to reduce prediction errors

---

##  Workflow

1. Load Dataset
2. Separate Features and Target Variable
3. Split Dataset into Training and Testing Data
4. Train Regression Model
5. Predict Housing Prices
6. Evaluate Model Performance

---

##  Evaluation Metrics

The models are evaluated using:

* Mean Squared Error (MSE)
* R² Score


