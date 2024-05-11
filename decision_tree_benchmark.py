# documentation referenced
# https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
import time
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# read in csv
data = pd.read_csv('datasets/spiral_10000000.csv')
x = data.drop('y', axis=1)
y = data['y']

# split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# make the regression tree
start_time = time.time()
tree = DecisionTreeRegressor(max_depth=10)
tree.fit(x_train, y_train)

# inference
y_predict = tree.predict(x_test)

# find mse
mse = mean_squared_error(y_test, y_predict)
print(mse)

# compute performance time
end_time = time.time()
print(f"Training and Prediction Time = {end_time - start_time:.6f}")


