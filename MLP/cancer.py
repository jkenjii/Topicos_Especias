
import numpy as np
import csv
import random
import math
from sklearn import datasets
from sklearn.model_selection import train_test_split #para 'splitar' o dataset

data = np.genfromtxt('cancer1.csv', delimiter = ',', dtype = int)
print(data)

X = []
y = []
for i in range(len(data)):
    X.append(data[i][:-1])
    if data[i][-1:] == 2:
        y.append(0)
    else:
        y.append(1)
    
X = np.array(X).tolist()
y = np.array(y).tolist()
print(X)

X = np.array(X)
y = np.array(y)
print(X)
print(y)

#z = (x - u) / s normazalizar
media_colunas = np.mean(X, axis=0)
std_colunas = np.std(X,  axis=0)
x = (X - media_colunas)/std_colunas

train_X , test_X , train_y, test_y = train_test_split(x,y, test_size = 0.25)

def sigmoide(A, derivada=False):# função da sigmoide com a sua derivada, caso requisitada
    if derivada: 
        for i in range(len(A)):
            A[i] = A[i] * (1 - A[i])
    else:
        for i in range(len(A)):
            A[i] = 1 / (1 + math.exp(-A[i]))
    return A

# Define parameter
learning_rate = 0.0008
epochs = 100
neuron = [10, 2, 2] #arquiterura da rede

#criar os pesos com o valor 0
W_1 = [[0 for j in range(neuron[1])] for i in range(neuron[0])]
W_2 = [[0 for j in range(neuron[2])] for i in range(neuron[1])]
B_1 = [0 for i in range(neuron[1])]
B_2 = [0 for i in range(neuron[2])]

#colocar valores aleatorios nos pesos(-1,1)
for i in range(neuron[0]):
    for j in range(neuron[1]):
        W_1[i][j] = 2 * random.random() - 1

for i in range(neuron[1]):
    for j in range(neuron[2]):
        W_2[i][j] = 2 * random.random() - 1


for abrobrinha in range(epochs):
    print('Epochs:', abrobrinha)    
    for item, x in enumerate(train_X):         
        
        h_1 = np.dot(x, W_1) + B_1
        X_1 = sigmoide(h_1)
        h_2 = np.dot(X_1, W_2) +  B_2
        X_2 = sigmoide(h_2)
        #print(X_2)
        
        #gambia
        target = [0, 0]
        target[int(train_y[item])] = 1
        #print(target)      
        
        #att camada 2
        delta_2 = []
        for j in range(neuron[2]):
            delta_2.append(-1 * 2. / neuron[2] * (target[j]-X_2[j]) * X_2[j] * (1-X_2[j]))

        for i in range(neuron[1]):
            for j in range(neuron[2]):
                W_2[i][j] -= learning_rate * (delta_2[j] * X_1[i])
                B_2[j] -= learning_rate * delta_2[j]
        
        #att camada 1
        delta_1 = np.dot(W_2, delta_2)
        for j in range(neuron[1]):
            delta_1[j] = delta_1[j] * (X_1[j] * (1-X_1[j]))
        
        for i in range(neuron[0]):
            for j in range(neuron[1]):
                W_1[i][j] -=  learning_rate * (delta_1[j] * x[i])
                B_1[j] -= learning_rate * delta_1[j]
    

out = np.dot(test_X, W_1) + B_1
out_1 = np.dot(out, W_2) + B_1

print(out_1)

preds = []
for r in out_1:
    preds.append(max(enumerate(r), key=lambda x:x[1])[0]) #pega o valor do indice de maior valor

print(preds)
print(test_y)

#acuracia
acc = 0.0
for i in range(len(preds)):
    if preds[i] == int(test_y[i]):
        acc += 1
print('Acuracia: ', acc / len(preds) * 100, "%")

#apresentou uma acuracia de 96%
#alterando para 10 na segunda camada foi para 87%
