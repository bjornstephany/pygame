# Cross validation Balance accuracy = 93.85%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import balanced_accuracy_score
 
LABEL_COLUMN = "Class"
HIDDEN_SIZE = 64
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUMBER_OF_EPOCHS = 15
DROPOUT_RATE = 0.3        # our regularization technique
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
train_data = pd.read_csv("dry_bean_train.csv")
test_data = pd.read_csv("dry_bean_test.csv")
feature_columns = [col for col in train_data.columns if col != LABEL_COLUMN]
 
# We turn each of the bean names into a number because networks only understand numbers
class_names = sorted(train_data[LABEL_COLUMN].unique())
name_to_number = {name: number for number, name in enumerate(class_names)}
 
attributes_train_full = train_data[feature_columns].values.astype(np.float32)
bean_type_train_full = train_data[LABEL_COLUMN].map(name_to_number).values.astype(np.int64)
attributes_test_full = test_data[feature_columns].values.astype(np.float32)
 
NUMBER_OF_FEATURES = attributes_train_full.shape[1]
NUMBER_OF_CLASSES = len(class_names)
 
 
# Standardise every column to mean = 0 and spread = 1 as networks train better this way
def standardise(attributes, mean, std):
    return (attributes - mean) / std
 
 
# wraps our beans so DataLoader can serve them as shuffled mini-batches
class BeanDataset(Dataset):
    def __init__(self, attributes, bean_type):
        self.attributes = torch.tensor(attributes, dtype=torch.float32)
        self.bean_type = torch.tensor(bean_type, dtype=torch.long)
 
    def __len__(self):
        return len(self.attributes)
 
    def __getitem__(self, index):
        return self.attributes[index], self.bean_type[index]
 
 
# Our network that takes 16 measurements as input and a prediction of which of the 7 types
# it is as the output 
class BeanNetwork(nn.Module):
    def __init__(self, use_dropout):
        super().__init__()
        self.hidden_layer = nn.Linear(NUMBER_OF_FEATURES, HIDDEN_SIZE)
        self.batch_norm = nn.BatchNorm1d(HIDDEN_SIZE)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT_RATE if use_dropout else 0.0)
        self.output_layer = nn.Linear(HIDDEN_SIZE, NUMBER_OF_CLASSES)
 
    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.output_layer(x)
 
 
# Loss function which are averaged over the entire batch
def cross_entropy_loss(outputs, labels):
    probabilities = torch.softmax(outputs, dim=1)
    total_loss = 0.0
    for i in range(len(labels)):
        correct_probability = probabilities[i, labels[i]]
        total_loss += -torch.log(correct_probability)
    return total_loss / len(labels)
 
 
# Trains one network, returns the model, its loss per epoch, and the mean/std used to scale the data
def train_network(attributes_train, bean_type_train, use_dropout):
    mean = attributes_train.mean(axis=0)
    std = attributes_train.std(axis=0)
    attributes_train = standardise(attributes_train, mean, std)
 
    loader = DataLoader(
        BeanDataset(attributes_train, bean_type_train),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
 
    model = BeanNetwork(use_dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
 
    losses = []
    for epoch in range(NUMBER_OF_EPOCHS):
        model.train()
        total_loss = 0.0
        total_rows = 0
 
        for batch_attributes, batch_labels in loader:
            batch_attributes = batch_attributes.to(device)
            batch_labels = batch_labels.to(device)
 
            outputs = model(batch_attributes)
            loss = cross_entropy_loss(outputs, batch_labels)
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
 
            total_loss += loss.item() * len(batch_labels)
            total_rows += len(batch_labels)
 
        losses.append(total_loss / total_rows)
 
    return model, losses, mean, std
 
 
# Trains a network and then predicts attributes_test ("make_predictions")
def network_predictions(attributes_train, bean_type_train, attributes_test, use_dropout):
    model, losses, mean, std = train_network(attributes_train, bean_type_train, use_dropout)
 
    model.eval()
    test_attributes = standardise(attributes_test, mean, std)
    test_tensor = torch.tensor(test_attributes, dtype=torch.float32).to(device)
 
    with torch.no_grad():
        outputs = model(test_tensor)
        predictions = outputs.argmax(dim=1).cpu().numpy()
 
    return predictions
 
 
def baseline_predictions(attributes_train, bean_type_train, attributes_test):
    return network_predictions(attributes_train, bean_type_train, attributes_test, use_dropout=False)
 
 
def regularized_predictions(attributes_train, bean_type_train, attributes_test):
    return network_predictions(attributes_train, bean_type_train, attributes_test, use_dropout=True)
 
 
# Cross-validation (no sklearn CV functions) similar to method used in forest.py
def cross_validate(attributes, bean_type, k, make_predictions):
    row_numbers = list(range(len(attributes)))
    np.random.shuffle(row_numbers)
    folds = np.array_split(row_numbers, k)
 
    scores = []
    for i in range(k):
        test_rows = folds[i]
        train_rows = np.concatenate([folds[j] for j in range(k) if j != i])
 
        predictions = make_predictions(
            attributes[train_rows], bean_type[train_rows], attributes[test_rows]
        )
        scores.append(balanced_accuracy_score(bean_type[test_rows], predictions))
 
    return sum(scores) / len(scores)
 
 
np.random.seed(0)
torch.manual_seed(0)
 
# We do a hyperparameter check by trying two different learning rates: 0.01 and 0.001
print("learning rate 0.01:")
LEARNING_RATE = 0.01
print("  balanced accuracy =", round(cross_validate(
    attributes_train_full, bean_type_train_full, k=3, make_predictions=regularized_predictions), 4))
 
print("learning rate 0.001:")
LEARNING_RATE = 0.001
print("  balanced accuracy =", round(cross_validate(
    attributes_train_full, bean_type_train_full, k=3, make_predictions=regularized_predictions), 4))
# We stick to 0.001 learning rate from here on
 
# Check regularization (with and without dropout)
print("baseline (no dropout):")
print("  balanced accuracy =", round(cross_validate(
    attributes_train_full, bean_type_train_full, k=3, make_predictions=baseline_predictions), 4))
 
print("regularized (with dropout):")
print("  balanced accuracy =", round(cross_validate(
    attributes_train_full, bean_type_train_full, k=3, make_predictions=regularized_predictions), 4))
 
# Use all the training data to train the final network
final_model, final_losses, mean, std = train_network(attributes_train_full, bean_type_train_full, use_dropout=True)
 
# Reload + save the weights 
torch.save(final_model.state_dict(), "network_weights.pt")
loaded_model = BeanNetwork(use_dropout=True).to(device)
loaded_model.load_state_dict(torch.load("network_weights.pt"))
loaded_model.eval()
 
# Visualization of the training loss going down whilst training
plt.plot(final_losses)
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.savefig("training_loss.png")
print("Saved training_loss.png")
 
# Use the training to predict the mystery beans from the test data
test_attributes = standardise(attributes_test_full, mean, std)
test_tensor = torch.tensor(test_attributes, dtype=torch.float32).to(device)
 
with torch.no_grad():
    test_outputs = loaded_model(test_tensor)
    test_predictions = test_outputs.argmax(dim=1).cpu().numpy()
 
predicted_names = [class_names[number] for number in test_predictions]
 
output = test_data.copy()
output["Target"] = predicted_names
output.to_csv("network.csv", index=False)
print("Saved network.csv with", len(output), "rows")