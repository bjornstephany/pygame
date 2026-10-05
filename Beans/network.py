# Cross validation Balance Accuracy = FILL_IN_AFTER_RUNNING %
# NOTE: this needs PyTorch, which isn't available where this was written,
# so it hasn't been run yet. Run it yourself, then put the real number
# that gets printed for "Regularized" on the line above.
 
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
 
# ---- Load the data ----
train_data = pd.read_csv("dry_bean_train.csv")
test_data = pd.read_csv("dry_bean_test.csv")
feature_columns = [col for col in train_data.columns if col != LABEL_COLUMN]
 
# A network only understands numbers, so turn each bean-type NAME into a
# number (0, 1, 2, ...). class_names turns a predicted number back into
# a real name again at the end.
class_names = sorted(train_data[LABEL_COLUMN].unique())
name_to_number = {name: number for number, name in enumerate(class_names)}
 
attributes_train_full = train_data[feature_columns].values.astype(np.float32)
bean_type_train_full = train_data[LABEL_COLUMN].map(name_to_number).values.astype(np.int64)
attributes_test_full = test_data[feature_columns].values.astype(np.float32)
 
NUMBER_OF_FEATURES = attributes_train_full.shape[1]
NUMBER_OF_CLASSES = len(class_names)
 
 
# ---- Put every measurement column on the same scale (mean 0, spread 1)
# - networks train much better this way. ----
def standardise(attributes, mean, std):
    return (attributes - mean) / std
 
 
# A Dataset just says how many rows it has, and hands back one
# (features, label) pair for a row number. DataLoader wraps it to serve
# shuffled MINI-BATCHES during training.
class BeanDataset(Dataset):
    def __init__(self, attributes, bean_type):
        self.attributes = torch.tensor(attributes, dtype=torch.float32)
        self.bean_type = torch.tensor(bean_type, dtype=torch.long)
 
    def __len__(self):
        return len(self.attributes)
 
    def __getitem__(self, index):
        return self.attributes[index], self.bean_type[index]
 
 
# ---- Our network: input -> hidden layer -> output.
# BatchNorm keeps the numbers flowing through training steady.
# Dropout randomly switches off some hidden neurons DURING TRAINING ONLY
# so the network can't just memorise the beans it has seen - that's our
# regularization (use_dropout=False gives the plain baseline). ----
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
 
 
# ---- Our own loss function, written by hand instead of using
# nn.CrossEntropyLoss. For every bean: turn its scores into probabilities
# with softmax, look up the probability it gave the CORRECT answer, and
# take -log of that (a confident correct guess gives a small loss, a
# confident wrong guess gives a huge one). Then average over the whole
# batch, so the loss doesn't just look bigger because the batch is bigger. ----
def cross_entropy_loss(outputs, labels):
    probabilities = torch.softmax(outputs, dim=1)
    total_loss = 0.0
    for i in range(len(labels)):
        correct_probability = probabilities[i, labels[i]]
        total_loss += -torch.log(correct_probability)
    return total_loss / len(labels)
 
 
# ---- Trains one network on the given beans. Returns the trained model,
# the loss for every epoch (for the loss plot), and the mean/std used to
# standardise the data (so new beans can be standardised the same way). ----
def train_network(attributes_train, bean_type_train, use_dropout):
    mean = attributes_train.mean(axis=0)
    std = attributes_train.std(axis=0)
    attributes_train = standardise(attributes_train, mean, std)
 
    # drop_last=True skips a tiny leftover batch at the end of an epoch,
    # which can crash Batch Normalization if it only has 1 bean in it
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
 
            # add up the loss from every mini-batch, so we can log one
            # loss number for the whole epoch
            total_loss += loss.item() * len(batch_labels)
            total_rows += len(batch_labels)
 
        losses.append(total_loss / total_rows)
 
    return model, losses, mean, std
 
 
# ---- Trains a network and uses it to guess the test beans. This is the
# "make_predictions" style function cross_validate (below) can call. ----
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
 
 
# ---- Our own cross-validation, same idea as forest.py's: split the
# beans into k parts, train on k-1, test on the part left out, repeat
# for every part, then average the balanced accuracy. ----
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
 
# ---- Hyperparameter check: try two learning rates, reusing the SAME
# cross_validate function - same trick as comparing the tree and forest. ----
print("learning rate 0.01:")
LEARNING_RATE = 0.01
print("  balanced accuracy =", round(cross_validate(
    attributes_train_full, bean_type_train_full, k=3, make_predictions=regularized_predictions), 4))
 
print("learning rate 0.001:")
LEARNING_RATE = 0.001
print("  balanced accuracy =", round(cross_validate(
    attributes_train_full, bean_type_train_full, k=3, make_predictions=regularized_predictions), 4))
# Keeping LEARNING_RATE = 0.001 from here on.
 
# ---- Regularization check: baseline (no dropout) vs our regularized
# network (with dropout) ----
print("baseline (no dropout):")
print("  balanced accuracy =", round(cross_validate(
    attributes_train_full, bean_type_train_full, k=3, make_predictions=baseline_predictions), 4))
 
print("regularized (with dropout):")
print("  balanced accuracy =", round(cross_validate(
    attributes_train_full, bean_type_train_full, k=3, make_predictions=regularized_predictions), 4))
 
# ---- Train the FINAL network on all the training data, using dropout ----
final_model, final_losses, mean, std = train_network(attributes_train_full, bean_type_train_full, use_dropout=True)
 
# Save the trained weights, then load them into a brand new model, to
# prove saving/loading with state_dict works.
torch.save(final_model.state_dict(), "network_weights.pt")
loaded_model = BeanNetwork(use_dropout=True).to(device)
loaded_model.load_state_dict(torch.load("network_weights.pt"))
loaded_model.eval()
 
# ---- Visualise the loss going down while it trained ----
plt.plot(final_losses)
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.savefig("training_loss.png")
print("Saved training_loss.png")
 
# ---- Predict the mystery test beans and save the submission file ----
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