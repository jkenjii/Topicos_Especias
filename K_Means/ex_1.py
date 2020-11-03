#ex 1

import matplotlib.pyplot as plt
import numpy as np

X = np.array([
	[1.9,7.3],
	[3.4,7.5],
	[2.5,6.8],
	[1.5,6.5],
	[3.5,6.4],
	[2.2,5.8],
	[3.4,5.2],
	[3.6,4.0],
	[5.0,3.2],
	[4.5,2.4],
	[6.0,2.6],
	[1.9,3.0],
	[1.0,2.7],
	[1.9,2.4],
	[0.8,2.0],
	[1.6,1.8],
	[1.0,1.0]])

print(X)

media_colunas = np.mean(X, axis=0)
std_colunas = np.std(X,  axis=0)
x = (X - media_colunas)/std_colunas
print(x)

def dist(a,b): #distancia euclidiana
    eu = []
    for i in range(len(a)):
        aux =[0,0,0] #Alterar de acordo com o numero de clusters
        for j in range(len(b)):                        
            aux[j] = ((((a[i][0]-b[j][0])**2)+((a[i][1]-b[j][1])**2))**0.5) #formula distancia euclidiana
        eu.append(aux)
    return np.array(eu)
	
def kmeans(X,k=3,max_iterations=1000):
    
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
plt.legend()
plt.show()
