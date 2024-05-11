# documentation referenced
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# read in csv
data = pd.read_csv('datasets/spiral_10000.csv')
x = data.drop('y', axis=1)
y = data['y']

# split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# make the random forest
start_time = time.time()
rf = RandomForestClassifier(n_estimators=1000, max_depth=10, n_jobs=1)
rf.fit(x_train, y_train)

# inference
y_predict = rf.predict(x_test)

# find mse
mse = mean_squared_error(y_test, y_predict)
print(mse)

# compute performance time
end_time = time.time()
print(f"Training and Prediction Time = {end_time - start_time:.6f}")


