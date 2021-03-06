#4DIM -> PCA -> 2DIM -> LDA

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

#plt.scatter(reduzida[:, 0],reduzida[:, 1], c=y)
target_names = iris.target_names
for i, target_name in zip([0, 1, 2], target_names):
    plt.scatter(reduzida[y == i, 0], reduzida[y == i, 1],label=target_name)
plt.title('PCA')
plt.legend()
plt.show()

X = reduzida

Classes = np.unique(y)
media_a = [] #media por classe 
for i in Classes:
	media_a.append(np.mean(X[y == i], axis=0))
print("media_a",media_a)

media = np.mean(X, axis = 0) #media geral de cada atributo
print("media",media)

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

VALORES, VETORES = np.linalg.eig(np.linalg.inv(S_W).dot(S_B)) #achar os autovetores e valores

idx = VALORES.argsort()[::-1]  #indice maior para menor
VETORES = VETORES[:, idx]
print("vetores_lds",VETORES)
W = VETORES[:, :2] #DIMENSÕES REDUZIDAS DO LDA
print(W)
transformada = X.dot(W)

target_names = iris.target_names
for i, target_name in zip([0, 1, 2], target_names):
    plt.scatter(transformada[y == i, 0], transformada[y == i, 1],label=target_name)
plt.legend()
plt.plot(np.linspace(-2.5,3.5),(np.linspace(-2.5,3.5)*VETORES[1][0]/VETORES[0][0]))
plt.title('LDA - 2DIM')
plt.show()