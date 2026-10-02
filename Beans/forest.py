# Cross validation accuracy = 0.9222
 
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
 
ANSWER_COLUMN = "Class"
NUMBER_OF_TREES = 50

# each tree only gets to see 80% of the bean rows (cross-validation) and 60% of the measurement columns
SAMPLE_FRACTION = 0.8     
FEATURE_FRACTION = 0.6    
 
# Loading the csv data into tables
train_data = pd.read_csv("dry_bean_train.csv")
test_data = pd.read_csv("dry_bean_test.csv")
feature_columns = [col for col in train_data.columns if col != ANSWER_COLUMN]
 
attributes_train_full = train_data[feature_columns].values
bean_type_train_full = train_data[ANSWER_COLUMN].values
attributes_test_full = test_data[feature_columns].values
 
 
# Cross-validation function that we use for the single tree and a forest by 
# taking in some training beans and some test beans and then returning a guess for 
# the type of every test beans. 
def cross_validate(attributes, bean_type, k, make_predictions):
    row_numbers = list(range(len(attributes)))
    np.random.shuffle(row_numbers)
    folds = np.array_split(row_numbers, k)
 
    accuracies = []
    for i in range(k):
        test_rows = folds[i]
        train_rows = np.concatenate([folds[j] for j in range(k) if j != i])
 
        predictions = make_predictions(
            attributes[train_rows], bean_type[train_rows], attributes[test_rows]
        )
        correct = np.sum(predictions == bean_type[test_rows])
        accuracies.append(correct / len(test_rows))
 
    return sum(accuracies) / len(accuracies)
 
 
# "make_predictions" function but only for a single tree
def tree_predictions(attributes_train, bean_type_train, attributes_test):
    tree = DecisionTreeClassifier(max_depth=10)
    tree.fit(attributes_train, bean_type_train)
    return tree.predict(attributes_test)
 
 
# Building the forest; total 50 times as once per tree
def train_forest(attributes, bean_type):
    number_of_rows = len(attributes)
    number_of_columns = attributes.shape[1]
    forest = []
 
    for i in range(NUMBER_OF_TREES):
        rows_to_use = np.random.choice(number_of_rows, int(number_of_rows * SAMPLE_FRACTION), replace=True)
        columns_to_use = np.random.choice(number_of_columns, int(number_of_columns * FEATURE_FRACTION), replace=False)
 
        tree = DecisionTreeClassifier()
        tree.fit(attributes[rows_to_use][:, columns_to_use], bean_type[rows_to_use])
        forest.append((tree, columns_to_use))
 
    return forest
 
 
# Ask every tree in the forest for its prediction. Then we decide whichever answer got the most votes.
def predict_with_forest(forest, attributes):
    votes = np.array([tree.predict(attributes[:, columns]) for tree, columns in forest])
 
    predictions = []
    for bean_index in range(attributes.shape[0]):
        values, counts = np.unique(votes[:, bean_index], return_counts=True)
        predictions.append(values[np.argmax(counts)])
 
    return np.array(predictions)
 
 # "make_predictions" function but only for an entire forest
def forest_predictions(attributes_train, bean_type_train, attributes_test):
    forest = train_forest(attributes_train, bean_type_train)
    return predict_with_forest(forest, attributes_test)
 
 
tree_score = cross_validate(attributes_train_full, bean_type_train_full, k=5, make_predictions=tree_predictions)
forest_score = cross_validate(attributes_train_full, bean_type_train_full, k=5, make_predictions=forest_predictions)
print("Single tree accuracy:", round(tree_score, 4))
print("Forest accuracy:", round(forest_score, 4))
 
# This trains one real and final forest on all the training beans and uses that to guess the type of the unknown test beans
final_forest = train_forest(attributes_train_full, bean_type_train_full)
test_predictions = predict_with_forest(final_forest, attributes_test_full)
 
output = test_data.copy()
output["Target"] = test_predictions
output.to_csv("forest.csv", index=False)
print("Saved forest.csv with", len(output), "rows")
 