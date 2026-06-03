"""
Wine Classifier using K-Nearest Neighbors (KNN)

Description:
This project predicts the class of wine based on its chemical properties
using the K-Nearest Neighbors (KNN) machine learning algorithm.
The model performs data preprocessing, feature scaling, hyperparameter
tuning, training, evaluation, and accuracy visualization.

Technologies Used:
- Python
- Pandas
- Matplotlib
- Scikit-learn
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

def Classifier(Datapath):
    Border = "-"*50
   
    # Step 1 : Load the dataset from CSV file
    print(Border)
    print("Step 1 : Load the dataset from CSV file")
    print(Border)

    df = pd.read_csv(Datapath) 

    print(Border)
    print("Some entries from dataset")
    print(df.head()) 
    print(Border)

    # Step 2 : Clean the dataset by removing empty rows
    print(Border)
    print("Step 2 : Clean the dataset by removing empty rows")
    print(Border)

    df.dropna(inplace = True)

    print("Total records : ",df.shape[0])
    print("Total columns : ",df.shape[1])
    print(Border)

    # Step 3 :  seperate independent and dependent variable

    print(Border)
    print("Step 3 : Seperate independent and dependent variable")
    print(Border)

    X = df.drop(columns=['Class']) 
    Y = df['Class']

    print("Shape of x : ",X.shape)
    print("Shape of Y : ",Y.shape)

    print(Border)
    print("Input columns : ",X.columns.tolist()) 
    print("Output columns : Class")

    # Step 4 : Split the dataset for training and testing

    print(Border)
    print("Step 4 : Split the dataset for training and testing")
    print(Border)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y) 

    print(Border)
    print("Information of training and testing data")
    print("X_train shape : ",X_train.shape)
    print("X_test shape : ",X_test.shape)
    print("Y_train shape : ",Y_train.shape)
    print("Y_test shape : ",Y_test.shape)
    print(Border)

    # Step 5 : Feature Scaling

    print(Border)
    print("Step 5 : Feature Scaling")
    print(Border)

    Scaler = StandardScaler() 
    
    X_train_scaled = Scaler.fit_transform(X_train)
    X_test_scaled = Scaler.transform(X_test)

    print("Features scaling is done")

    # Step 6 : Explore multiple values of k

    accuracy_scores = []
    k_values = range(1, 21)

    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_scaled, Y_train)
        Y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(Y_test, Y_pred)
        accuracy_scores.append(accuracy)

        print(Border)
        print("Accuracy report of all k values from 1 to 20")
        for value in accuracy_scores:
            print(value)

        print(Border)

    # Step 7 : Plot graph of k vs accuracy

    print(Border)
    print("Step 7 : Plot graph of k vs accuracy")
    print(Border)

    plt.figure(figsize = (8, 5))
    plt.plot(k_values, accuracy_scores, marker = 'o')
    plt.title("K Values vs Accuracy")
    plt.xlabel("Value of K")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.xticks(list(k_values))
    plt.show()

    # Step 8 : Find best values of k

    print(Border)
    print("Step 8 : Find best values of k")
    print(Border)

    best_k = list(k_values)[accuracy_scores.index(max(accuracy_scores))]

    print("Best value of k is : ",best_k)

    # Step 9 : Build final model using best value of k

    print(Border)
    print("# Step 9 : Build final model using best value of k")
    print(Border)

    final_model = KNeighborsClassifier(n_neighbors= best_k)

    final_model.fit(X_train_scaled,Y_train)
    Y_pred = final_model.predict(X_test_scaled)

    # Step 10 : Calculate final accuracy

    print(Border)
    print("Step 10 : Calculate final accuracy")
    print(Border)

    accuracy = accuracy_score(Y_test,Y_pred)
    print("Accuracy of the model is : ",accuracy * 100)

    # Step 11 : Display confusion matrix
    print(Border)
    print("Step 11 : Display confusion matrix")
    print(Border)

    cm = confusion_matrix(Y_test,Y_pred)

    print(cm)

    # Step 12 : Display classification report

    print(Border)
    print("Step 12 : Display classification report")
    print(Border)

    print(classification_report(Y_test,Y_pred))

    

def main():
    Border = "-"*50
    print(Border)
    print("Wine Classifier using KNN")
    print(Border)

    Classifier("WinePredictor.csv")

if __name__ == "__main__":
    main()