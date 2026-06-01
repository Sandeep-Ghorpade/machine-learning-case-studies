"""
Customer Segmentation System

Description:
This project uses the K-Means Clustering algorithm to segment customers
based on Annual Income and Spending Score. The Elbow Method is used to
determine the optimal number of clusters.

"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def main():
    #-------------------------------------------------
    # Step 1 : Load Dataset
    #-------------------------------------------------
    print("Step 1 : Load the dataset")
    df = pd.read_csv("Mall_Customers.csv")

    print("First few records : ")
    print(df.head())

    print("Shape of dataset : ")
    print(df.shape)

    print("Missing values : ")
    print(df.isnull().sum())

    #-------------------------------------------------
    # Step 2 : Select Features(Independent)
    #-------------------------------------------------
    print("Step 2 : Select Features(Independent)")

    X = df[["AnnualIncome", "SpendingScore"]]
    print("Selected Features : ")
    print(X.head())

    print("Shape of selected features : ")
    print(X.shape)

    #-------------------------------------------------
    # Step 3 : Scale the data
    #-------------------------------------------------
    print("Step 3 : Scale the data")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) 

    print("Data after scalling : ")
    print(X_scaled[:5])

    #-------------------------------------------------
    # Step 3 : Use elbow method
    #-------------------------------------------------

    WCSS = []

    for i in range(1, 11):
        model = KMeans(n_clusters=i, random_state=42, n_init=10)
        model.fit(X_scaled)
        WCSS.append(model.inertia_)

    plt.figure(figsize=(8,5))
    plt.plot(range(1, 11), WCSS, marker = 'o')
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS")
    plt.title("Elbow method")
    plt.grid(True)
    plt.show()

    #-------------------------------------------------
    # Step 5 : Train the model
    #-------------------------------------------------

    model = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = model.fit_predict(X_scaled)

    df["Clusters"] = clusters

    print("Dataset with clusters")
    print(df.head())

if __name__ == "__main__":
    main()