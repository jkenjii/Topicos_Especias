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

def Matriz_transposta(m):
    return list(map(list,zip(*m)))

#multiplicação matrizes XT(2,17) x X (17,2) tem matriz final (2,2)
def Prod_matriz(matrizA, matrizB):
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

def Matriz_Inversa(m):
    det = Determinante(m)
    
    if len(m) == 2:
        return [[m[1][1]/det, -1*m[0][1]/det],
                [-1*m[1][0]/det, m[0][0]/det]]

def ypred(x,coefi):
    y = []
    for i in range(len(x)):                 
        y_aux = coefi[0][0] + x[i]*coefi[1][0]
        y.append(y_aux)
    return y

def extrairX(X):
    x1 = []
    for i in range(len(X)):
        x1.append(float(X[i][1]))
    return x1

XT = Matriz_transposta(X) 
XTX = Prod_matriz(XT,X)
XTY = Prod_matriz(XT,y)
inversa = Matriz_Inversa(XTX)
coef = Prod_matriz(inversa,XTY)
print(coef)

x1 = extrairX(X)
print(x1)

ypred = ypred(x1,coef)
print(ypred)
print(y)

plt.figure()
plt.scatter(x1, y,color='black')
plt.plot(x1, ypred, color='blue', linewidth=1)
plt.yticks(())
plt.yscale('linear')
plt.show()