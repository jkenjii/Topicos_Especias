#Least Squares Method usa
import matplotlib.pyplot as plt
import numpy as np


M = np.array([[1900,75.9950],  
    [1910,91.9720],    
    [1920,105.7110],    
    [1930,123.2030],
    [1940,131.6690],    
    [1950,150.6970],    
    [1960,179.3230],    
    [1970,203.2120],    
    [1980,226.5050],    
    [1990,249.6330],
    [2000,281.4220]])

#media de cada coluna
print(M)
media_colunas = np.mean(M, axis=0)
print(media_colunas)

#retirar media dos valores
DataAdjust = M - media_colunas
print('Data Ajuste: ',DataAdjust)

#por padrão a função considera as variavei as linhas, por isso a necessidade de transpo
matriz_cova = np.cov(DataAdjust.T)
print("Cov por .cov:",matriz_cova)

#cov dos itens
def cov(x, y):
    x_media, y_media = x.mean(), y.mean()
    return np.sum((x - x_media)*(y - y_media))/(len(x) - 1)

#Calcula a matriz de convolução manualmente, item por item
def cov_mat(X):
    return np.array([[cov(X[0], X[0]), cov(X[0], X[1])], 
                     [cov(X[1], X[0]), cov(X[1], X[1])]])

byhand = cov_mat(DataAdjust.T)
print("Cov manual:", byhand)

#biblioteca do numpy,aplicado com a matriz gerada por .cov
auto_VALORES, auto_VETORES = np.linalg.eig(matriz_cova)
print("Auto Vetores: ", auto_VETORES)
print("Auto Valores: ",auto_VALORES)

#Os novos valores pos PCA(nesse caso considera todas as componentes)
p = np.matmul(auto_VETORES.T,DataAdjust.T).T
print("Novos Valores: ",p)

#para gerar as entradas para a regressão
X = []
for i in range(len(DataAdjust)):
    X.append([1,DataAdjust[i][0]])

y = []
for i in range(len(DataAdjust)):
    y.append([DataAdjust[i][1]])


#Realizar a matrix transposta
def Matriz_transposta(m):
    xt = list(map(list,zip(*m)))
    return xt

#multiplicação matrizes XT(2,17) x X (17,2) tem matriz final (2,2)
def Prod_Matriz(matrizA, matrizB):    
    matrizR = []
    # Multiplica
    for i in range(len(matrizA)):
        matrizR.append([])
        for j in range(len(matrizB[0])):
            val = 0
            for k in range(len(matrizA[0])):
                    val = val + matrizA[i][k]*matrizB[k][j]
            matrizR[i].append(val)
    return matrizR

#determinante matrix 2x2
def Determinante(m):    
    if len(m) == 2:
        det = m[0][0]*m[1][1]-m[0][1]*m[1][0]
        return det

#matriz 2x2 inversa
def Matriz_Inversa(m):
    det = Determinante(m)    
    if len(m) == 2:
        inv = [[m[1][1]/det, -1*m[0][1]/det],[-1*m[1][0]/det, m[0][0]/det]]
        return inv

def ypred(x,coefi):
    y = []
    for i in range(len(x)):                 
        y_aux = coefi[0][0] + x[i]*coefi[1][0]
        y.append(y_aux)
    return y

def estrairX(X):
    x1 = []
    for i in range(len(X)):
        x1.append(float(X[i][1]))
    return x1

#Método dos minimos quadrados para a regressão
XT = Matriz_transposta(X) 
XTX = Prod_Matriz(XT,X)
XTY = Prod_Matriz(XT,y)
inversa = Matriz_Inversa(XTX)
coef = Prod_Matriz(inversa,XTY)
x1 = estrairX(X)
ypred = ypred(x1,coef)

#plot dos gráficos
plt.figure()
plt.scatter(DataAdjust[:,0], DataAdjust[:,1],edgecolors='k')
x=np.linspace(-60,60)
#Reta da Componente Principal(foi utilizado a que possui o maior auto valor)
plt.plot(x, (auto_VETORES[0][0]/auto_VETORES[1][0])*x, color='k',label='PCA') 
#Reta da regressão
plt.plot(x, coef[0][0] + x*coef[1][0], color='y', linewidth=1,linestyle='--',label='LMS')
plt.title('USA Census')
plt.legend()
plt.yticks(())
plt.yscale('linear')
plt.show()

