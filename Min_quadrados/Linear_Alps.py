#Least Squares Method ALPS WATER
import matplotlib.pyplot as plt

X = [[1,194.5],
    [1,194.3],
    [1,197.9],
    [1,198.4],
    [1,199.4],
    [1,199.9],
    [1,200.9],
    [1,201.1],
    [1,201.4],
    [1,201.3],
    [1,203.6],
    [1,204.6],
    [1,209.5],
    [1,208.6],
    [1,210.7],
    [1,211.9],
    [1,212.2]]


y = [[20.79],    
    [20.79],  
    [22.40],   
    [22.67],    
    [23.15],    
    [23.35],    
    [23.89],    
    [23.99],    
    [24.02],    
    [24.01],    
    [25.14],    
    [26.57],    
    [28.49],    
    [27.76],    
    [29.04],    
    [29.88],    
    [30.06]]    

print(X)
print(len(X))
print(len(y))

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

XT = Matriz_transposta(X) 
XTX = Prod_Matriz(XT,X)
XTY = Prod_Matriz(XT,y)
inversa = Matriz_Inversa(XTX)
coef = Prod_Matriz(inversa,XTY)
print(coef)

x1 = estrairX(X)
print(x1)

ypred = ypred(x1,coef)
print(ypred)

y200 = coef[0][0] + 200*coef[1][0]
print('Valor de pressao em temp 200F:',y200)
plt.figure()
plt.scatter(x1, y,color='black')
plt.plot(x1, ypred, color='blue', linewidth=1)
plt.yticks(())
plt.yscale('linear')
plt.show()

