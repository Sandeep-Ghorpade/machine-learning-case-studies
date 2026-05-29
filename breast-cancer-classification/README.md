# Breast Cancer Classification

Machine Learning project for predicting the presence of breast cancer using multiple classification and ensemble learning algorithms.

---

##  Project Overview

This project uses various Machine Learning classification algorithms to predict whether breast cancer is present or not using the Breast Cancer dataset.

The project demonstrates:

* Data preprocessing
* Classification techniques
* Model training
* Model evaluation
* Ensemble learning methods
* Voting-based classification

---

##  Technologies Used

* Python
* Pandas
* Scikit-learn

---

##  Project Structure

```text
breast-cancer-classification/
│
├── dataset/
│   └── breast_cancer.csv
│
├── src/
│   ├── breast_cancer_decision_tree.py
│   ├── breast_cancer_random_forest.py
│   ├── breast_cancer_boosting_classifier.py
│   ├── breast_cancer_individual_models.py
│   ├── breast_cancer_hard_voting.py
│   └── breast_cancer_soft_voting.py
│
├── README.md
└── requirements.txt
```

---

##  Machine Learning Algorithms Used

### Individual Classification Models

#### 1. Decision Tree Classifier

* Tree-based classification algorithm
* Used for prediction and decision analysis

#### 2. Logistic Regression

* Statistical classification algorithm
* Suitable for binary classification problems

#### 3. K-Nearest Neighbors (KNN)

* Instance-based learning algorithm
* Classifies samples based on nearest neighbors

#### 4. Random Forest Classifier

* Ensemble learning technique
* Uses multiple decision trees for improved accuracy and robustness

#### 5. AdaBoost Classifier

* Boosting algorithm
* Improves model performance by learning from previous prediction errors

---

##  Voting Ensemble Methods

### Hard Voting Classifier

* Combines predictions from multiple classification models
* Final prediction is based on majority voting

### Soft Voting Classifier

* Combines probability predictions from multiple classification models
* Final prediction is based on average probabilities

---

##  Workflow

1. Load Dataset
2. Separate Features and Labels
3. Split Dataset into Training and Testing Data
4. Train Individual Models
5. Generate Predictions
6. Apply Ensemble Learning Techniques
7. Evaluate Model Performance
8. Compare Classification Results

---

##  Evaluation Metrics

The models are evaluated using:

* Accuracy Score
* Confusion Matrix
* Classification Report

---

##  Learning Outcomes

This project demonstrates:

* Classification Algorithms
* Ensemble Learning
* Bagging Techniques
* Boosting Techniques
* Voting Classifiers
* Model Performance Comparison
