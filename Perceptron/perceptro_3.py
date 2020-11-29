import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split #para 'splitar' o dataset

iris = datasets.load_iris()

X_1 = iris.data[:100]
y_1 = iris.target[:100]
print(y_1)
X_2 = iris.data[50:]
y_2 = iris.target[50:] # 2 para 0
for i in range(len(y_2)):
    if y_2[i] == 2:
        y_2[i] = 0

print((y_2))
iris = datasets.load_iris()
X_3_1 = iris.data[:50] 
X_3_2 = iris.data[100:]
y_3_1 = iris.target[:50] #2 para 1
y_3_2 = iris.target[100:]

y_3 = np.concatenate((y_3_1, y_3_2), axis=None)
X_3 = np.concatenate((X_3_1, X_3_2),axis = 0)


for i in range(len(y_3)):
    if y_3[i] == 2:
        y_3[i] = 1
print(y_3)

iris = datasets.load_iris()

target_names = iris.target_names
features = iris.feature_names
#z = (x - u) / s normazalizar
def normalizar(X):
    media_colunas = np.mean(X, axis=0)
    std_colunas = np.std(X,  axis=0)
    x = (X - media_colunas)/std_colunas
    return x

x_1 = normalizar(X_1)
x_2 = normalizar(X_2)
x_3 = normalizar(X_3)
print(x_1)
print(target_names)

def transform_dataset(x_trans): #transformar em array cada linha
	dataset = []
	for row in x_trans:
		dataset.append(np.array(row))
	return dataset

x_1 = transform_dataset(x_1)
x_2 = transform_dataset(x_2)
x_3 = transform_dataset(x_3)
print(x_3)

labels_1 = np.array(y_1)
labels_2 = np.array(y_2)
labels_3 = np.array(y_3)
print(labels_3)
print(labels_2)

class Perceptron(object):

    def __init__(self, atributos, epochs=1000, learning_rate=0.0001):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = np.zeros(atributos)
        self.bias = np.zeros(1)
           
    def predict(self, inputs):
        funcao = np.dot(inputs, self.weights) + self.bias
        if funcao > 0:
          binario = 1
        else:
          binario = 0            
        return binario

    def train(self, train_inputs, saidas):
        for abrobrinha in range(self.epochs):
            for entrada, saida in zip(train_inputs, saidas):
                prediction = self.predict(entrada)                
                self.weights = self.weights + (self.learning_rate * entrada * (saida - prediction))
                self.bias  = self.bias + (self.learning_rate * (saida - prediction))
        print(self.weights)
        print(self.bias)

Perceptron_1 = Perceptron(4)
Perceptron_2 = Perceptron(4)
Perceptron_3 = Perceptron(4)

Perceptron_1.train(x_1, labels_1)
Perceptron_2.train(x_2, labels_2)
Perceptron_3.train(x_3, labels_3)

iris = datasets.load_iris()
data_test = iris.data

X = normalizar(data_test)
X = transform_dataset(X)

print(X)

saida_predict_1 = []
for row in X:
	saida_predict_1.append(Perceptron_1.predict(row))

saida_predict_2 = []
for row in X:
    saida_predict_2.append(Perceptron_2.predict(row))

saida_predict_3 = []
for row in X:
    saida_predict_3.append(Perceptron_3.predict(row))

predicao = []
for i in range(len(saida_predict_1)):
    if (saida_predict_1[i] == 0 and saida_predict_2[i] == 0) and saida_predict_3[i] == 0:
        predicao.append('Setosa')
    elif (saida_predict_1[i] == 0 and saida_predict_2[i] == 1) and saida_predict_3[i] == 0:
        predicao.append('Setosa')

    elif (saida_predict_1[i] == 1 and saida_predict_2[i] == 1) and saida_predict_3[i] == 0:
        predicao.append('Versicolor')
    elif (saida_predict_1[i] == 1 and saida_predict_2[i] == 1) and saida_predict_3[i] == 1:
        predicao.append('Versicolor')

    elif (saida_predict_1[i] == 0 and saida_predict_2[i] == 0) and saida_predict_3[i] == 1:
        predicao.append('Virginica')
    elif (saida_predict_1[i] == 1 and saida_predict_2[i] == 0) and saida_predict_3[i] == 1:
        predicao.append('Virginica')
    else:
        predicao.append('Indeciso')
print(saida_predict_1)
print(saida_predict_2)
print(saida_predict_3)
print(predicao)
print(len(predicao))

		
