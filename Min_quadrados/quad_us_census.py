#Quad us census
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
xj = []
XX = []
for i in range(len(X)):		
	xj = [0,0,0]
	xj[0] = X[i][0]
	xj[1] = X[i][1]
	xj[2] = X[i][1]*X[i][1]
	XX.append(xj)
print((XX))

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
def Prod_Matriz(matrizA, matrizB):
    matrizR = []
    # Multiplica
    for i in range(len(matrizA)):
        matrizR.append([])
        for j in range(len(matrizB[0])):
            val = 0
            for k in range(len(matrizA[0])):
                    val += matrizA[i][k]*matrizB[k][j]
            matrizR[i].append(val)
    return matrizR

def subMatriz(m,i,j):
    return [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]

 #https://mathworld.wolfram.com/Determinant.html
def Determinante(m):
    if len(m) == 2:
        return m[0][0]*m[1][1]-m[0][1]*m[1][0]    
    det = 0
    for c in range(len(m)):
        det = det + ((-1)**c)*m[0][c]*Determinante(subMatriz(m,0,c))
    return det    

def Matrix2x2Inversa(m):
    det = Determinante(m)    
    if len(m) == 2:
        return [[m[1][1]/det, -1*m[0][1]/det],
                [-1*m[1][0]/det, m[0][0]/det]]	

#https://www.infoescola.com/matematica/matriz-inversa-inversao-por-matriz-adjunta/
def Matriz_Inversa(m):
	det = Determinante(m)
	coft = []
	for r in range(len(m)):
			coft_1 = []
			for c in range (len(m)):
				minor = subMatriz(m,r,c)
				coft_1.append(((-1)**(r+c))*Determinante(minor))
			coft.append(coft_1)
	coft = Matriz_transposta(coft)
	for r in range(len(coft)):
		for c in range(len(coft)):
			coft[r][c] = coft[r][c]/det
	return coft

def ypred_linear(x,coefi):
    y = []
    for i in range(len(x)):           
        y.append(coefi[0][0] + x[i]*coefi[1][0])
    return y

def ypred_quad(x,coef):
	y=[]
	for i in range(len(x)):
		y.append(coef[0][0]+x[i]*coef[1][0]+x[i]*x[i]*coef[2][0])
	return y


def extrairX(X):
    x1 = []
    for i in range(len(X)):
        x1.append(float(X[i][1]))
    return x1

#quadratico
XT = Matriz_transposta(XX) 
print(XT)
XTX = Prod_Matriz(XT,XX)
print('esse qui',XTX)
XTY = Prod_Matriz(XT,y)
print(XTY)
print(Determinante(XTX))
inversa = Matriz_Inversa(XTX)
print(inversa)
coef_quad = Prod_Matriz(inversa,XTY)
print(coef_quad)

#linear
XT_L = Matriz_transposta(X) 
print(XT_L)
XTX_L = Prod_Matriz(XT_L,X)
print('esse qui',XTX_L)
XTY_L = Prod_Matriz(XT_L,y)
print(XTY_L)
print(Determinante(XTX_L))
inversa_L = Matrix2x2Inversa(XTX_L)
print(inversa_L)
coef = Prod_Matriz(inversa_L,XTY_L)
print(coef)

x1 = extrairX(X)
print(x1)

y_linear = ypred_linear(x1,coef)
y_quad = ypred_quad(x1,coef_quad)
print(y_linear)

plt.figure()
plt.scatter(x1, y,color='black')

plt.plot(x1, y_linear, color='blue', linewidth=1)
plt.plot(x1,y_quad,color='black',linewidth=1)
plt.yticks(())
plt.yscale('linear')
plt.show()