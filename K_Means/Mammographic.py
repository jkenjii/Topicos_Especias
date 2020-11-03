#http://archive.ics.uci.edu/ml/datasets/Mammographic+Mass

import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('massa.csv', delimiter = ',', dtype = int)
print(data)

M = np.array(data)

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
W = auto_VETORES[:, :2] #DIMENSÕES REDUZIDAS DO PCA (2 dimensões no caso)
reduzida = M.dot(W) 
print(reduzida)

print("Vetores: ", auto_VETORES)
print("Valores: ",auto_VALORES)

X = reduzida
#z = (x - u) / s normazalizar
media_colunas = np.mean(X, axis=0)
std_colunas = np.std(X,  axis=0)
x = (X - media_colunas)/std_colunas
print(x)

def dist(a,b): #distancia euclidiana
    eu = []
    for i in range(len(a)):
        aux =[0,0] #Alterar de acordo com o numero de clusters
        for j in range(len(b)):                        
            aux[j] = ((((a[i][0]-b[j][0])**2)+((a[i][1]-b[j][1])**2))**0.5) #formula distancia euclidiana
        eu.append(aux)
    return np.array(eu)
	
def kmeans(X,k=2,max_iterations=1000):
    
    ind = np.random.choice(len(X), k, replace=False) #3 (no caso linhas do dataset) pontos aleatorios    
    centroides = X[ind, :] #o valor das 3 linhas aleatorias    
    distan = dist(X, centroides) #distancia dos pontos de X com os centroides aleatorios    
    P = np.argmin(distan,axis=1) #menor numero de cada linha, tendo como saida o seu indice (0,1,2)
    for iterations in range(max_iterations):
        centroides = np.vstack([X[P==i,:].mean(axis=0) for i in range(k)])
        tmp = np.argmin(dist(X, centroides),axis=1)
        if np.array_equal(P,tmp): #se P nao mudar mais ou atingido o max de iterações, é finalizado a execução
            break
        P = tmp
    return P, centroides

P, centroides_ = kmeans(x)
print(centroides_)

plt.figure()
plt.scatter(x[:,0],x[:,1],c=P)
plt.scatter(centroides_[:,0],centroides_[:,1],color="black", label="Centroides")
plt.title('Mammographic Mass Data Set ~ PCA + K-means')
plt.legend()
plt.show()