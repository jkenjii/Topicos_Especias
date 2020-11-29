import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split #para 'splitar' o dataset

iris = datasets.load_iris()

X = iris.data[:100] #pegar somente setosa e versicolor
y = iris.target[:100]
target_names = iris.target_names
features = iris.feature_names
#z = (x - u) / s normazalizar
media_colunas = np.mean(X, axis=0)
std_colunas = np.std(X,  axis=0)
x = (X - media_colunas)/std_colunas
print(x)
print(y)
print(target_names)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25)

def transform_dataset(x_trans): #transformar em array cada linha
	dataset = []
	for row in x_trans:
		dataset.append(np.array(row))
	return dataset

train_x = transform_dataset(x_train)
print(train_x)

labels = np.array(y_train)

class Perceptron(object):

    def __init__(self, atributos, epochs=1000, learning_rate=0.0001): #declaração das variaveis
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

Perceptron = Perceptron(4)
Perceptron.train(train_x, labels)

x_test = transform_dataset(x_test)
print(x_test)

saida_predict = []
for row in x_test:
	saida_predict.append(Perceptron.predict(row))

print(saida_predict)
print(y_test)

for i in range(len(saida_predict)):
	if y_test[i] != saida_predict[i]:		
		print('diferente')
	else:
		if y_test[i] == 0:
			print('setosa')
		else:
			print('versicolor')
		
