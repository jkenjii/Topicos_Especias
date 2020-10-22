#4DIM -> PCA -> 1DIM -> LDA

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()

M = iris.data
y = iris.target

print(M)
media_colunas = np.mean(M, axis=0)
print(media_colunas)

DataAdjust = M - media_colunas
print(DataAdjust)

matriz_cova = np.cov(DataAdjust, rowvar=False)
print(matriz_cova)

auto_VALORES, auto_VETORES = np.linalg.eig(matriz_cova)

idx = auto_VALORES.argsort()[::-1]  #indice maior para menor
auto_VETORES = auto_VETORES[:, idx]
W = auto_VETORES[:, :2] #DIMENSÕES REDUZIDAS DO PCA 
reduzida = M.dot(W) 
print(reduzida)

print("Vetores: ", auto_VETORES)
print("Valores: ",auto_VALORES)

plt.scatter(reduzida[:, 0],reduzida[:, 0], c=y)
plt.title('PCA')
plt.show()

X = reduzida

Classes = np.unique(y)
 
media_a = []
 
for i in Classes:
	media_a.append(np.mean(X[y == i], axis=0))
print(media_a)

media = np.mean(X, axis = 0) #media geral de cada atributo
print(media)

S_B = np.zeros((X.shape[1], X.shape[1])) #between-class scatter matrix
for i, media_a in enumerate(media_a):
	n = X[y==i].shape[0]	
	media_a = media_a.reshape(1,X.shape[1]) #para linha
	m = media_a - media
	S_B = S_B + (n * np.matmul(m.T,m))
print(S_B)

x0 = X[y==0] #valores apenas da classe 0
x1 = X[y==1] #valores apenas da classe 1
x2 = X[y==2] #valores apenas da classe 2

conv0 = np.cov(x0.T) #matriz de conv da classe 0
conv1 = np.cov(x1.T) #matriz de conv da classe 1
conv2 = np.cov(x2.T) #matriz de conv da classe 2

S_W = conv0 + conv1 + conv2 #within-class scatter matrix(somatoriA das matrizes de conv de cada classe)
print(S_W)

eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B)) #achar os autovetores e valores

idx = eig_vals.argsort()[::-1]  #indice maior para menor
eig_vecs = eig_vecs[:, idx]
print(eig_vecs)
W = eig_vecs[:, :1] #DIMENSÕES REDUZIDAS DO LDA
print(W)
transformed = X.dot(W)

plt.scatter(transformed[:, 0],transformed[:, 0], c=y)
plt.title('LDA')
plt.show()