#Least Squares Method ALPS WATER

import matplotlib
import matplotlib.pyplot as plt

X = [[1,1900],  
    [1,1910],   
    [1,1920],   
    [1,1930],
    [1,1940],   
    [1,1950],   
    [1,1960],   
    [1,1970],   
    [1,1980],   
    [1,1990],
    [1,2000]]


y = [[75.9950],
    [91.9720],
    [105.7110],
    [123.2030],
    [131.6690],
    [150.6970],
    [179.3230],
    [203.2120],
    [226.5050],
    [249.6330],
    [281.4220]]

def Matriz_trasposta(m):
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

def Determinante(m):
 
    if len(m) == 2:
        return m[0][0]*m[1][1]-m[0][1]*m[1][0]

#matriz 2x2
def Matriz_Inversa(m):
    det = Determinante(m)
    
    if len(m) == 2:
        return [[m[1][1]/det, -1*m[0][1]/det],
                [-1*m[1][0]/det, m[0][0]/det]]

def ypred(x,coefi):
    y = []
    for i in range(len(x)):                 
        y.append(coefi[0][0] + x[i]*coefi[1][0])
    return y

def extrairX(X):
    x1 = []
    for i in range(len(X)):
        x1.append(float(X[i][1]))
    return x1

def obterW(y,y_pred):
    w = []
    for i in range(len(y_pred)):      
        w.append([abs(1/(y[i][0]-y_pred[i]))])
    return w

#fazer com que w tenha mesma dimensão que X
def ajusteW(w):
    W  = []
    for i in range(len(w)):
        W1 = [0,0]
        W1[0] = w[i][0]
        W1[1] = w[i][0]
        W.append(W1)
    return W

#matrizes de mesmas dimensões
def prodEscalarMatriz(matrizA,matrizB):
    produto = []
    for i in range(len(matrizA)):
        produto_aux = []
        produto.append(produto_aux)
        for j in range(len(matrizA[0])):
            valor = matrizA[i][j]*matrizB[i][j]
            produto_aux.append(valor)
    return produto

def robusto(x,w,w1,y):
    xt = Matriz_trasposta(x)
    print('xt:',xt)
    print(Prod_Matriz(xt,w))    
    p_op = Prod_Matriz(xt,prodEscalarMatriz(w,x))
    print(p_op)
    inv = Matriz_Inversa(p_op)
    s_op = Prod_Matriz(xt,prodEscalarMatriz(w1,y))
    resultado = Prod_Matriz(inv,s_op)
    return resultado

XT = Matriz_trasposta(X) 
XTX = Prod_Matriz(XT,X)
print('xtx:',XTX)
XTY = Prod_Matriz(XT,y)
inversa = Matriz_Inversa(XTX)
coef = Prod_Matriz(inversa,XTY)
print(coef)

x1 = extrairX(X)
print(x1)

ypred = ypred(x1,coef)
print(ypred)

w = obterW(y,ypred)
print(w)
ajsW = ajusteW(w)
print(ajsW)

coef_robust = robusto(X,ajsW,w,y)
print(coef_robust)
print(coef)
print(x1)
print(ypred)

plt.figure()
plt.scatter(x1, y,color='green')

ypred_robust=[]
for i in range(len(x1)):
    val = coef_robust[0][0]+ x1[i]*coef_robust[1][0]
    ypred_robust.append(val)
print(ypred_robust)

#Robusto
plt.plot(x1,ypred_robust,color='black',linewidth=3,linestyle='--')
#Linear
plt.plot(x1, ypred, color='blue', linewidth=1)
plt.yticks(())
plt.yscale('linear')
plt.show()
