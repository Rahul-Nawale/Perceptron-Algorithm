"""
******************************************************************************************
** COMP572 Data Mining & Visualization                                                  **
** NAME: RAHUL NAWALE                                                                   **
** STUDENT ID - 201669264                                                               **
** TASK - CA Assignment 1 - Data Classification - Implementing Perceptron algorithm     **
**                                                                                      **
******************************************************************************************
       %%%%       %%%%%     %%%        %%%  %%% %%%%     %%%%%%%%%  %%%%%%%%  %%%%%%
     %%%  %%%   %%%    %%%  %%% %%  %% %%%  %%%     %%%  %%             %%%%       %%%
     %%%        %%%    %%%  %%%   %%   %%%  %%%     %%%  %%%%%          %%%        %%%
     %%%        %%%    %%%  %%%        %%%  %%% %%%%          %%%     %%%      %%%
     %%%  %%%   %%%    %%%  %%%        %%%  %%%               %%%    %%%    %%
      %%%%        %%%%%     %%%        %%%  %%%          %%%%%      %%%     %%%%%%%%%%
===========================================================================================
"""






# Question 2
"""perceptron_train(data, max_iter) takes two arguments:
1. the training data - which is a list of tuples (X, y)
where X is a list of features and y is the target label (either 1 or -1),
2. max_iter = maximum number of iterations to run the algorithm for."""
def perceptron_train(data, max_iter):
    # initialize weights and bias to zero
    weights = [0] * len(data[0][0])
    bias = 0

    # runs a loop for max_iter iterations.
    for i in range(max_iter):
        # In each iteration, the algorithm loops over all the training examples (X, y) in data.
        for X, y in data:
            # calculate activation function a = (dot product of X and weights) + bias
            a = sum([weights[j] * X[j] for j in range(len(X))]) + bias

            # update weights and bias if misclassified
            if y * a <= 0:
                for j in range(len(X)):
                    weights[j] += y * X[j]
                bias += y
    # after all iterations are complete, the algorithm returns the final bias and weights.
    return bias, weights



# The perceptron_test(bias, weights, X) function takes the bias and weights obtained from perceptron_train
# and a new input vector X
def perceptron_test(bias, weights, X):
    # calculate activation function
    a = sum([weights[j] * X[j] for j in range(len(X))]) + bias

    # return sign of activation function
    return 1 if a > 0 else -1


"""Explaination of the algorithm step by step:

perceptron_train(data, max_iter) takes two arguments: the training data data, 
which is a list of tuples (X, y) where X is a list of features and y is the target label (either 1 or -1), 
and max_iter which is the maximum number of iterations to run the algorithm for.

First, the algorithm initializes the weights and bias to zero.
Then, it runs a loop for max_iter iterations.
In each iteration, the algorithm loops over all the training examples (X, y) in data.
For each training example, 
it calculates the activation function "a" as the dot product of weights and X, plus the bias.
If the training example is misclassified (i.e., y * a <= 0), 
then the algorithm updates the weights and bias according to the perceptron update rule: 
wi = wi + y*xi for all i, and bias = bias + y.

Finally, after all iterations are complete, the algorithm returns the final bias and weights.
The perceptron_test(bias, weights, X) function takes the bias and weights 
obtained from perceptron_train and a new input vector X, and returns the predicted label for X. 

It calculates the activation function a for X using the same dot product formula as in the training algorithm, 
and then returns the sign of a, which is either 1 or -1 depending on whether the prediction is positive or negative."""




# Question 3
# Class 1 vs Class 2
import random

# Set random seed for reproducibility
random.seed(8969826)

# Function to read data from a file and separate features and labels
def read_data_1_2(file_name):
    X_1_2 = []
    y_1_2 = []
    # opening and reading the file
    with open(file_name, 'r') as f:
        for line in f:
            # reading data by line and splitting the data by comma
            data = line.strip().split(',')
            # list to hold X (all features)
            X_1_2.append([float(x) for x in data[:-1]])
            # list to hold Y (all labels)
            y_1_2.append(1 if data[-1] == 'class-1' else -1 if data[-1] == 'class-2' else 3)

        # removing unwanted class from data (both X and Y)
        indices = [k for k, x in enumerate(y_1_2) if x == 3]
        for j in reversed(indices):
            del X_1_2[j]
            del y_1_2[j]
    return X_1_2, y_1_2

# Function to train a perceptron classifier
def train_perceptron_1_2(X_train_1_2, y_train_1_2, num_iterations):
    # Initialize weights to zeros
    w_1_2 = [0.0] * len(X_train_1_2[0])
    b_1_2 = 0.0
    # Iterate for the given number of iterations
    for epoch in range(num_iterations):
        # Shuffle the data
        data = list(zip(X_train_1_2, y_train_1_2))
        random.shuffle(data)
        X_train_1_2, y_train_1_2 = zip(*data)
        # Iterate over each training example
        for i in range(len(X_train_1_2)):
            # Compute the predicted label
            y_pred = 1 if sum([w_1_2[j] * X_train_1_2[i][j] for j in range(len(w_1_2))]) + b_1_2 > 0 else -1
            # Update the weights if the predicted label is incorrect
            if y_pred != y_train_1_2[i]:
                w_1_2 = [w_1_2[j] + y_train_1_2[i] * X_train_1_2[i][j] for j in range(len(w_1_2))]
                b_1_2 += y_train_1_2[i]
    return w_1_2, b_1_2

# Function to predict labels using a trained perceptron classifier
def predict(X_test_1_2, w_1_2, b_1_2):
    y_pred = []
    for x in X_test_1_2:
        y_pred.append(1 if sum([w_1_2[j] * x[j] for j in range(len(w_1_2))]) + b_1_2 > 0 else -1)
    return y_pred

# Read the training and test data
X_train_1_2, y_train_1_2 = read_data_1_2('train.data')
X_test_1_2, y_test_1_2 = read_data_1_2('test.data')

# Train the perceptron classifier
w_1_2, b_1_2 = train_perceptron_1_2(X_train_1_2, y_train_1_2, 20)

# Predict labels for the training and test data
y_train_pred_1_2 = predict(X_train_1_2, w_1_2, b_1_2)
y_test_pred_1_2 = predict(X_test_1_2, w_1_2, b_1_2)

# Compute the classification accuracies
train_acc_1_2 = sum([1 if y_train_pred_1_2[i] == y_train_1_2[i] else 0 for i in range(len(y_train_1_2))]) / len(y_train_1_2)
test_acc_1_2 = sum([1 if y_test_pred_1_2[i] == y_test_1_2[i] else 0 for i in range(len(y_test_1_2))]) / len(y_test_1_2)

# Print the classification accuracies
print('\nClass 1 vs Class 2:')
print('Train accuracy: {:.2f}%'.format(train_acc_1_2 * 100))
print('Test accuracy: {:.2f}%'.format(test_acc_1_2 * 100))






# Question 3
# Class 2 vs Class 3
def read_data_2_3(file_name):
    X_2_3 = []
    y_2_3 = []
    with open(file_name, 'r') as f:
        for line in f:
            data = line.strip().split(',')
            X_2_3.append([float(x) for x in data[:-1]])
            y_2_3.append(1 if data[-1] == 'class-2' else -1 if data[-1] == 'class-3' else 3)
        indices = [k for k, x in enumerate(y_2_3) if x == 3]
        for j in reversed(indices):
            del X_2_3[j]
            del y_2_3[j]
    return X_2_3, y_2_3

# Function to train a perceptron classifier
def train_perceptron_2_3(X_train_2_3, y_train_2_3, num_iterations):
    # Initialize weights to zeros
    w_2_3 = [0.0] * len(X_train_2_3[0])
    b_2_3 = 0.0
    # Iterate for the given number of iterations
    for epoch in range(num_iterations):
        # Shuffle the data
        data = list(zip(X_train_2_3, y_train_2_3))
        random.shuffle(data)
        X_train_2_3, y_train_2_3 = zip(*data)
        # Iterate over each training example
        for i in range(len(X_train_2_3)):
            # Compute the predicted label
            y_pred = 1 if sum([w_2_3[j] * X_train_2_3[i][j] for j in range(len(w_2_3))]) + b_2_3 > 0 else -1
            # Update the weights if the predicted label is incorrect
            if y_pred != y_train_2_3[i]:
                w_2_3 = [w_2_3[j] + y_train_2_3[i] * X_train_2_3[i][j] for j in range(len(w_2_3))]
                b_2_3 += y_train_2_3[i]
    return w_2_3, b_2_3

# Function to predict labels using a trained perceptron classifier
def predict(X_test_2_3, w_2_3, b_2_3):
    y_pred = []
    for x in X_test_2_3:
        y_pred.append(1 if sum([w_2_3[j] * x[j] for j in range(len(w_2_3))]) + b_2_3 > 0 else -1)
    return y_pred

# Read the training and test data
X_train_2_3, y_train_2_3 = read_data_2_3('train.data')
X_test_2_3, y_test_2_3 = read_data_2_3('test.data')

# Train the perceptron classifier
w_2_3, b_2_3 = train_perceptron_2_3(X_train_2_3, y_train_2_3, 20)

# Predict labels for the training and test data
y_train_pred_2_3 = predict(X_train_2_3, w_2_3, b_2_3)
y_test_pred_2_3 = predict(X_test_2_3, w_2_3, b_2_3)

# Compute the classification accuracies
train_acc_2_3 = sum([1 if y_train_pred_2_3[i] == y_train_2_3[i] else 0 for i in range(len(y_train_2_3))]) / len(y_train_2_3)
test_acc_2_3 = sum([1 if y_test_pred_2_3[i] == y_test_2_3[i] else 0 for i in range(len(y_test_2_3))]) / len(y_test_2_3)

# Print the classification accuracies
print('\nClass 2 vs Class 3:')
print('Train accuracy: {:.2f}%'.format(train_acc_2_3 * 100))
print('Test accuracy: {:.2f}%'.format(test_acc_2_3 * 100))





# Question 3
# Class 1 vs Class 3
def read_data_1_3(file_name):
    X_1_3 = []
    y_1_3 = []
    with open(file_name, 'r') as f:
        for line in f:
            data = line.strip().split(',')
            X_1_3.append([float(x) for x in data[:-1]])
            y_1_3.append(1 if data[-1] == 'class-1' else -1 if data[-1] == 'class-3' else 3)
        indices = [k for k, x in enumerate(y_1_3) if x == 3]
        for j in reversed(indices):
            del X_1_3[j]
            del y_1_3[j]
    return X_1_3, y_1_3

# Function to train a perceptron classifier
def train_perceptron_1_3(X_train_1_3, y_train_1_3, num_iterations):
    # Initialize weights to zeros
    w_1_3 = [0.0] * len(X_train_1_3[0])
    b_1_3 = 0.0
    # Iterate for the given number of iterations
    for epoch in range(num_iterations):
        # Shuffle the data
        data = list(zip(X_train_1_3, y_train_1_3))
        random.shuffle(data)
        X_train_1_3, y_train_1_3 = zip(*data)
        # Iterate over each training example
        for i in range(len(X_train_1_3)):
            # Compute the predicted label
            y_pred = 1 if sum([w_1_3[j] * X_train_1_3[i][j] for j in range(len(w_1_3))]) + b_1_3 > 0 else -1
            # Update the weights if the predicted label is incorrect
            if y_pred != y_train_1_3[i]:
                w_1_3 = [w_1_3[j] + y_train_1_3[i] * X_train_1_3[i][j] for j in range(len(w_1_3))]
                b_1_3 += y_train_1_3[i]
    return w_1_3, b_1_3

# Function to predict labels using a trained perceptron classifier
def predict(X_test_1_3, w_1_3, b_1_3):
    y_pred = []
    for x in X_test_1_3:
        y_pred.append(1 if sum([w_1_3[j] * x[j] for j in range(len(w_1_3))]) + b_1_3 > 0 else -1)
    return y_pred

# Read the training and test data
X_train_1_3, y_train_1_3 = read_data_1_3('train.data')
X_test_1_3, y_test_1_3 = read_data_1_3('test.data')

# Train the perceptron classifier
w_1_3, b_1_3 = train_perceptron_1_3(X_train_1_3, y_train_1_3, 20)

# Predict labels for the training and test data
y_train_pred_1_3 = predict(X_train_1_3, w_1_3, b_1_3)
y_test_pred_1_3 = predict(X_test_1_3, w_1_3, b_1_3)

# Compute the classification accuracies
train_acc_1_3 = sum([1 if y_train_pred_1_3[i] == y_train_1_3[i] else 0 for i in range(len(y_train_1_3))]) / len(y_train_1_3)
test_acc_1_3 = sum([1 if y_test_pred_1_3[i] == y_test_1_3[i] else 0 for i in range(len(y_test_1_3))]) / len(y_test_1_3)

# Print the classification accuracies
print('\nClass 1 vs Class 3:')
print('Train accuracy: {:.2f}%'.format(train_acc_1_3 * 100))
print('Test accuracy: {:.2f}%'.format(test_acc_1_3 * 100))





# Question 4

# Reading the training and testing data
with open('train.data', 'r') as f:
    train_data = f.read().splitlines()

with open('test.data', 'r') as f:
    test_data = f.read().splitlines()

# Initializing the weights and bias
weights = [[0.0, 0.0, 0.0, 0.0] for i in range(3)]
bias = [0.0 for i in range(3)]

# Defining the activation function
def activation(x, w, b):
    net = sum([x[i]*w[i] for i in range(4)]) + b
    return 1 if net >= 0 else -1

# Training the perceptron for 20 iterations
learning_rate = 1
for iteration in range(20):
    for instance in train_data:
        # Splitting the instance into its features and label
        instance = instance.split(',')
        x = [float(feature) for feature in instance[:4]]
        y = instance[4]

        # Updating the weights and bias for each class
        for i in range(3):
            if y == f"class-{i+1}":
                if activation(x, weights[i], bias[i]) == -1:
                    for j in range(4):
                        weights[i][j] += learning_rate * x[j]
                    bias[i] += learning_rate
            else:
                if activation(x, weights[i], bias[i]) == 1:
                    for j in range(4):
                        weights[i][j] += (-1) * x[j]
                    bias[i] += -1


# Testing the accuracy of the classifier on the training data
train_correct = 0
for instance in train_data:
    instance = instance.split(',')
    x = [float(feature) for feature in instance[:4]]
    y = instance[4]

    # Predicting the label for each class and selecting the class with the highest score
    scores = [sum([x[j]*weights[i][j] for j in range(4)]) + bias[i] for i in range(3)]
    predicted_label = f"class-{scores.index(max(scores))+1}"

    if predicted_label == y:
        train_correct += 1

train_accuracy = train_correct*100 / len(train_data)

# Testing the accuracy of the classifier on the testing data
test_correct = 0
for instance in test_data:
    instance = instance.split(',')
    x = [float(feature) for feature in instance[:4]]
    y = instance[4]

    # Predicting the label for each class and selecting the class with the highest score
    scores = [sum([x[j]*weights[i][j] for j in range(4)]) + bias[i] for i in range(3)]
    predicted_label = f"class-{scores.index(max(scores))+1}"

    if predicted_label == y:
        test_correct += 1

test_accuracy = test_correct*100 / len(test_data)

# Printing the accuracies
print("\nMulti-class")
print("Training accuracy:", train_accuracy)
print("Testing accuracy:", test_accuracy)




# Question 5

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = f.read().splitlines()
    return data

def activation(x, w, b):
    net = sum([x[i]*w[i] for i in range(4)]) + b
    return 1 if net >= 0 else -1

def train_with_l2(train_data, l2):
    weights = [[0.0, 0.0, 0.0, 0.0] for i in range(3)]
    bias = [0.0 for i in range(3)]
    for iteration in range(20):
        for instance in train_data:
            # Splitting the instance into its features and label
            instance = instance.split(',')
            x = [float(feature) for feature in instance[:4]]
            y = instance[4]

            # Updating the weights and bias for each class
            for i in range(3):
                if y == f"class-{i + 1}":
                    if activation(x, weights[i], bias[i]) == -1:
                        for j in range(4):
                            weights[i][j] = ((1-(2*l2))*weights[i][j])+ x[j]
                        bias[i] += 1
                    else:
                        weights[i][j] = ((1-(2*l2))*weights[i][j])
                else:
                    if activation(x, weights[i], bias[i]) == 1:
                        for j in range(4):
                            weights[i][j] = ((1-(2*l2))*weights[i][j])-x[j]
                        bias[i] += -1
                    else:
                        weights[i][j] = ((1-(2*l2))*weights[i][j])
    return bias, weights

def train_acc(train_data, bias, weights):
    train_correct = 0
    for instance in train_data:
        instance = instance.split(',')
        x = [float(feature) for feature in instance[:4]]
        y = instance[4]

        # Predicting the label for each class and selecting the class with the highest score
        scores = [sum([x[j] * weights[i][j] for j in range(4)]) + bias[i] for i in range(3)]
        predicted_label = f"class-{scores.index(max(scores)) + 1}"

        if predicted_label == y:
            train_correct += 1

    train_accuracy = train_correct * 100 / len(train_data)
    return train_accuracy

def test_acc(test_data, bias, weights):
    test_correct = 0
    for instance in test_data:
        instance = instance.split(',')
        x = [float(feature) for feature in instance[:4]]
        y = instance[4]

        # Predicting the label for each class and selecting the class with the highest score
        scores = [sum([x[j] * weights[i][j] for j in range(4)]) + bias[i] for i in range(3)]
        predicted_label = f"class-{scores.index(max(scores)) + 1}"

        if predicted_label == y:
            test_correct += 1

    test_accuracy = test_correct * 100 / len(test_data)
    return test_accuracy

train_data = load_data('train.data')
test_data = load_data('test.data')

for l2 in [0.01, 0.1, 1.0, 10.0, 100.0]:
    bias, weights = train_with_l2(train_data, l2)

    trainAccuracy = train_acc(train_data, bias, weights)
    testAccuracy = test_acc(test_data, bias,weights)

    print(f"\nRegularization coefficient: {l2}")
    print('Train accuracy:', trainAccuracy)
    print('Test accuracy:', testAccuracy)