import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

#-----------------------------------------------------------------
# Step 1 : Load the dataset
#-----------------------------------------------------------------

df = pd.read_csv("california_housing.csv")
print("Shape of dataset : ",df.shape)
print("First 5 records : ",df.head())

#-----------------------------------------------------------------
# Step 2 : Seperate features and labels
#-----------------------------------------------------------------

X = df.drop("target",axis = 1) # vertical cut
Y = df["target"]

#-----------------------------------------------------------------
# Step 3 : Split dataset for training and testing
#-----------------------------------------------------------------

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#-----------------------------------------------------------------
# Step 4 : Create base model
#-----------------------------------------------------------------

base_model = DecisionTreeRegressor(random_state=42)

#-----------------------------------------------------------------
# Step 5 : Create bagging model
#-----------------------------------------------------------------

bagging_model = BaggingRegressor(
    estimator=base_model, # estimator made multiple decision tree classifier
    n_estimators= 10, # made 10 decision tree classifier
    random_state=42
)

#-----------------------------------------------------------------
# Step 6 : Train the model
#-----------------------------------------------------------------

bagging_model.fit(X_train, Y_train)

#-----------------------------------------------------------------
# Step 7 : Test the model
#-----------------------------------------------------------------

Y_pred = bagging_model.predict(X_test)

#-----------------------------------------------------------------
# Step 8 : Evaluate bagging model
#-----------------------------------------------------------------

print("MeanSquaredError : ",mean_squared_error(Y_test, Y_pred))

print("R Square : ",r2_score(Y_test, Y_pred))