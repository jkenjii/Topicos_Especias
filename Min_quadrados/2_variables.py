import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

import mpl_toolkits.mplot3d 

XX = [[1, 0, 9],
	[1,1,15],
	[1,0,10],
	[1,2,16],
	[1,4,10],
	[1,4,20],
	[1,1,11],
	[1,4,20],
	[1,3,15],
	[1,0,15],
	[1,2,8],
	[1,1,13],
	[1,4,18],
	[1,1,10],
	[1,0,8],
	[1,1,10],
	[1,3,16],
	[1,0,11],
	[1,1,19],
	[1,4,12],
	[1,4,11],
	[1,0,19],
	[1,2,15],
	[1,3,15],
	[1,1,20],
	[1,0,6],
	[1,3,15],
	[1,3,19],
	[1,2,14],
	[1,2,13],
	[1,3,17],
	[1,2,20],
	[1,2,11],
	[1,3,20],
	[1,4,20],
	[1,4,20],
	[1,3,9],
	[1,1,8],
	[1,2,16],
	[1,0,10]]

x1=[]
for i in range(len(XX)):
	x1.append(XX[i][1])
print(x1)

x2=[]
for i in range(len(XX)):
	x2.append(XX[i][2])
print(x2)

y = [[45],
	[57],
	[45],
	[51],
	[65],
	[88],
	[44],
	[87],
	[89],
	[59],
	[66],
	[65],
	[56],
	[47],
	[66],
	[41],
	[56],
	[37],
	[45],
	[58],
	[47],
	[64],
	[97],
	[55],
	[51],
	[61],
	[69],
	[79],
	[71],
	[62],
	[87],
	[54],
	[43],
	[92],
	[83],
	[94],
	[60],
	[56],
	[88],
	[62]]

def Matriz_transposta(m):
    return list(map(list,zip(*m)))

#multiplicação matrizes XT(2,17) x X (17,2) tem matriz final (2,2)
def Prod_Matriz(matrizA, matrizB):

    matrizR = []
    
    for i in range(len(matrizA)):
        matrizR.append([])
        for j in range(len(matrizB[0])):
            val = 0
            for k in range(len(matrizA[0])):
                    val += matrizA[i][k]*matrizB[k][j]
            matrizR[i].append(val)
    return matrizR

def subMatrix(m,i,j):
    return [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]

 #https://mathworld.wolfram.com/Determinant.html
def Determinante(m):
    if len(m) == 2:
        return m[0][0]*m[1][1]-m[0][1]*m[1][0]    
    det = 0
    for c in range(len(m)):
        det += ((-1)**c)*m[0][c]*Determinante(subMatrix(m,0,c))
    return det
    
#https://www.infoescola.com/matematica/matriz-inversa-inversao-por-matriz-adjunta/
def getMatrixInverse(m):
    det = Determinante(m)
    coft = []
    for r in range(len(m)):
            coft_1 = []
            for c in range (len(m)):
                minor = subMatrix(m,r,c)
                coft_1.append(((-1)**(r+c))*Determinante(minor))
            coft.append(coft_1)
    coft = Matriz_transposta(coft)
    for r in range(len(coft)):
        for c in range(len(coft)):
            coft[r][c] = coft[r][c]/det
    return coft

def y_pred(x1,x2,coefi):
    y = []
    for i in range(len(x1)):                 
        y.append(coefi[0][0] + x1[i]*coefi[1][0]+x2[i]*coefi[2][0])
    return y

XT = Matriz_transposta(XX) 
print(XT)
XTX = Prod_Matriz(XT,XX)
print('esse qui',XTX)
XTY = Prod_Matriz(XT,y)
print(XTY)
print(Determinante(XTX))
inversa = getMatrixInverse(XTX)
print(inversa)
coef = Prod_Matriz(inversa,XTY)
print(coef)

ypred = y_pred(x1,x2,coef)
print(ypred)

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_trisurf(x1,x2,ypred)
plt.show()