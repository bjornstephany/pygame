import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Loading the data from CSV files
Answer_Column = "Class"
train_data = pd.read_csv("dry_bean_train.csv")
test_data = pd.read_csv("dry_bean_test.csv")

attributes_columns = [column for column in train_data.columns if column != Answer_Column]
attributes_train_full = train_data[attributes_columns].values
bean_type_train_full = train_data[Answer_Column].values
attributes_test_full = test_data[attributes_columns].values
