

"""
SECTION 1 : Load and setup data for training
"""

import csv
import random
import math
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split #para 'splitar' o dataset

import csv
import random
import math
from sklearn import datasets
from sklearn.model_selection import train_test_split #para 'splitar' o dataset

iris = datasets.load_iris()

X = iris.data #pegar somente setosa e versicolor
y = iris.target
target_names = iris.target_names
features = iris.feature_names
#z = (x - u) / s normazalizar
media_colunas = np.mean(X, axis=0)
std_colunas = np.std(X,  axis=0)
x = (X - media_colunas)/std_colunas
print(x)
print(y)
print(target_names)

train_X , test_X , train_y, test_y = train_test_split(x,y, test_size = 0.25)

print(train_X)
print(test_y)
"""
SECTION 2 : Build and Train Model
Multilayer perceptron model, with one hidden layer.
input layer : 4 neuron, represents the feature of Iris
hidden layer : 3 neuron, activation using sigmoid
output layer : 3 neuron, represents the class of Iris
optimizer = gradient descent
loss function = Square ROot Error
learning rate = 0.005
epoch = 400
best result = 96.67%
"""

def matrix_mul_bias(A, B, bias): # Matrix multiplication (for Testing)
    C = [[0 for i in range(len(B[0]))] for i in range(len(A))]    
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[i][j] += A[i][k] * B[k][j]
            C[i][j] = C[i][j] + bias[j]
    return C

def vec_mat_bias(A, B, bias): # Vector (A) x matrix (B) multiplication
    C = [0 for i in range(len(B[0]))]
    for j in range(len(B[0])):
        for k in range(len(B)):
            C[j] += (A[k] * B[k][j]) + bias[j]
            
    return C


def mat_vec(A, B): # Matrix (A) x vector (B) multipilicatoin (for backprop)
    C = [0 for i in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B)):
            C[i] += A[i][j] * B[j]
    return C

def sigmoid(A, deriv=False):
    if deriv: # derivation of sigmoid (for backprop)
        for i in range(len(A)):
            A[i] = A[i] * (1 - A[i])
    else:
        for i in range(len(A)):
            A[i] = 1 / (1 + math.exp(-A[i]))
    return A

# Define parameter
alfa = 0.0008
epoch = 5000
neuron = [4, 5, 3] # number of neuron each layer

# Initiate weight and bias with 0 value
weight = [[0 for j in range(neuron[1])] for i in range(neuron[0])]
weight_2 = [[0 for j in range(neuron[2])] for i in range(neuron[1])]
bias = [0 for i in range(neuron[1])]
bias_2 = [0 for i in range(neuron[2])]

# Initiate weight with random between -1.0 ... 1.0
for i in range(neuron[0]):
    for j in range(neuron[1]):
        weight[i][j] = 2 * random.random() - 1

for i in range(neuron[1]):
    for j in range(neuron[2]):
        weight_2[i][j] = 2 * random.random() - 1


for e in range(epoch):
    cost_total = 0
    for idx, x in enumerate(train_X): # Update for each data; SGD
        
        # Forward propagation
        h_1 = vec_mat_bias(x, weight, bias)
        X_1 = sigmoid(h_1)
        h_2 = vec_mat_bias(X_1, weight_2, bias_2)
        X_2 = sigmoid(h_2)
        print(X_2)
        
        # Convert to One-hot target
        target = [0, 0, 0]
        target[int(train_y[idx])] = 1

        # Cost function, Square Root Eror
        eror = 0
        for i in range(neuron[2]):
            eror +=  (target[i] - X_2[i]) ** 2 
        cost_total += eror * 1 / neuron[2]

        # Backward propagation
        # Update weight_2 and bias_2 (layer 2)
        delta_2 = []
        for j in range(neuron[2]):
            delta_2.append(-1 * 2. / neuron[2] * (target[j]-X_2[j]) * X_2[j] * (1-X_2[j]))

        for i in range(neuron[1]):
            for j in range(neuron[2]):
                weight_2[i][j] -= alfa * (delta_2[j] * X_1[i])
                bias_2[j] -= alfa * delta_2[j]
        
        # Update weight and bias (layer 1)
        delta_1 = mat_vec(weight_2, delta_2)
        for j in range(neuron[1]):
            delta_1[j] = delta_1[j] * (X_1[j] * (1-X_1[j]))
        
        for i in range(neuron[0]):
            for j in range(neuron[1]):
                weight[i][j] -=  alfa * (delta_1[j] * x[i])
                bias[j] -= alfa * delta_1[j]
    
    cost_total /= len(train_X)
    if(e % 100 == 0):
        print(cost_total)

"""
SECTION 3 : Testing
"""

res = matrix_mul_bias(test_X, weight, bias)
res_2 = matrix_mul_bias(res, weight_2, bias)

print(res_2)
# Get prediction
preds = []
for r in res_2:
    preds.append(max(enumerate(r), key=lambda x:x[1])[0])

# Print prediction
print(preds)

# Calculate accuration
acc = 0.0
for i in range(len(preds)):
    if preds[i] == int(test_y[i]):
        acc += 1
print(acc / len(preds) * 100, "%")