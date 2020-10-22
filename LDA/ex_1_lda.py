import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data
y_1 = iris.target

atributos = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
iris_df = pd.DataFrame(data= iris.data, columns= atributos) #df apenas com os dados
target_df = pd.DataFrame(data= iris.target, columns= ['Class']) #df so com os valores das classes
irisdata = pd.concat([iris_df, target_df], axis = 1) #df de tudo junto
print (irisdata)
media_atributos = irisdata.groupby('Class').mean() 	
media_a = np.array(media_atributos.values) #media de atributo por classe
print(list(enumerate(media_a)))

media = np.mean(X, axis = 0) #media geral de cada atributo
print(media)

S_B = np.zeros((X.shape[1], X.shape[1])) #between-class scatter matrix
for i, media_a in enumerate(media_a):
	n = X[y_1==i].shape[0]	
	media_a = media_a.reshape(1,X.shape[1]) #para linha
	m = media_a - media
	S_B = S_B + (n * np.matmul(m.T,m))
print(S_B)

x0 = X[y_1==0] #valores apenas da classe 0
x1 = X[y_1==1] #valores apenas da classe 1
x2 = X[y_1==2] #valores apenas da classe 2

conv0 = np.cov(x0.T) #matriz de conv da classe 0
conv1 = np.cov(x1.T) #matriz de conv da classe 1
conv2 = np.cov(x2.T) #matriz de conv da classe 2

S_W = conv0 + conv1 + conv2 #within-class scatter matrix(somatoriA das matrizes de conv de cada classe)
print(S_W)

eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B)) #achar os autovetores e valores

idx = eig_vals.argsort()[::-1]  #indice maior para menor
eig_vecs = eig_vecs[:, idx]
eig_vals = eig_vals[idx]
print(eig_vals)
print(eig_vecs)
W = eig_vecs[:, :2] #escolhe apenas 2 dimensões
print(W)
transformada = X.dot(W) 

target_names = iris.target_names
for i, target_name in zip([0, 1, 2], target_names):
    plt.scatter(transformada[y_1 == i, 0], transformada[y_1 == i, 1],label=target_name)
plt.legend()
plt.title('LDA')
plt.show()