
import numpy as np

M = np.array([
[0,   9,   45],
[1,   15 , 57],
[0 ,  10 , 45],
[2 ,  16 , 51],
[4 ,  10 , 65],
[4 ,  20 , 88],
[1 ,  11 , 44],
[4 ,  20 , 87],
[3 ,  15 , 89],
[0 ,  15 , 59],
[2 ,  8  , 66],
[1 ,  13 , 65],
[4 ,  18 , 56],
[1 ,  10 , 47],
[0 ,  8  , 66],
[1 ,  10 , 41],
[3 ,  16 , 56],
[0 ,  11 , 37],
[1 ,  19 , 45],
[4 ,  12 , 58],
[4 ,  11 , 47],
[0 ,  19 , 64],
[2 ,  15 , 97],
[3 ,  15 , 55],
[1 ,  20 , 51],
[0 ,  6  , 61],
[3 ,  15 , 69],
[3 ,  19 , 79],
[2 ,  14 , 71],
[2 ,  13 , 62],
[3 ,  17 , 87],
[2 ,  20 , 54],
[2 ,  11 , 43],
[3 ,  20 , 92],
[4 ,  20 , 83],
[4 ,  20 , 94],
[3 ,  9   ,60],
[1 ,  8   ,56],
[2 ,  16 , 88],
[0 ,  10,  62]])

#media de cada coluna
print(M)
media_colunas = np.mean(M, axis=0)
print(media_colunas)

#retirar media dos valores
DataAdjust = M - media_colunas
print(DataAdjust)

#retirar media dos valores
matriz_cova = np.cov(DataAdjust, rowvar=False)
print(matriz_cova)

#biblioteca do numpy
auto_VALORES, auto_VETORES = np.linalg.eig(matriz_cova)
print("Auto Vetores: ", auto_VETORES)
print("Auto Valores: ",auto_VALORES)

#Os novos valores pos PCA(nesse caso considera todas as componentes)
p = np.matmul(auto_VETORES.T,DataAdjust.T).T
print("Novos Valores: ",p)


