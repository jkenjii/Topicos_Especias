#http://archive.ics.uci.edu/ml/datasets/Mammographic+Mass

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = np.genfromtxt('massa.csv', delimiter = ',', dtype = int)
print(data)

M = np.array(data)

atributos = ['Bi-Rads','Age','Shape','Margin','Density','Class']

mamadataset = pd.DataFrame(data= M, columns= atributos)
print(mamadataset)

benign = mamadataset['Class'][mamadataset['Class'] == 0].count()
malignant = mamadataset['Class'][mamadataset['Class'] == 1].count()
total = mamadataset['Class'].count()

P_benign = benign/total
P_malignant = malignant/total
print(P_benign,P_malignant)

data_means = mamadataset.groupby('Class').mean()
print(data_means)
data_std = mamadataset.groupby('Class').std()
print(data_std)

#Means benign
benign_bi_mean = data_means['Bi-Rads'][data_means.index == 0].values[0]
benign_age_mean = data_means['Age'][data_means.index == 0].values[0]
benign_sh_mean = data_means['Shape'][data_means.index == 0].values[0]
benign_ma_mean = data_means['Margin'][data_means.index == 0].values[0]
benign_den_mean = data_means['Density'][data_means.index == 0].values[0]

#means malignant
malignant_bi_mean = data_means['Bi-Rads'][data_means.index == 1].values[0]
malignant_age_mean = data_means['Age'][data_means.index == 1].values[0]
malignant_sh_mean = data_means['Shape'][data_means.index == 1].values[0]
malignant_ma_mean = data_means['Margin'][data_means.index == 1].values[0]
malignant_den_mean = data_means['Density'][data_means.index == 1].values[0]

#std benign
benign_bi_std = data_std['Bi-Rads'][data_std.index == 0].values[0]
benign_age_std= data_std['Age'][data_std.index == 0].values[0]
benign_sh_std = data_std['Shape'][data_std.index == 0].values[0]
benign_ma_std = data_std['Margin'][data_std.index == 0].values[0]
benign_den_std = data_std['Density'][data_std.index == 0].values[0]

#std malignant
malignant_bi_std = data_std['Bi-Rads'][data_std.index == 1].values[0]
malignant_age_std= data_std['Age'][data_std.index == 1].values[0]
malignant_sh_std = data_std['Shape'][data_std.index == 1].values[0]
malignant_ma_std = data_std['Margin'][data_std.index == 1].values[0]
malignant_den_std = data_std['Density'][data_std.index == 1].values[0]

def p_x_y(x,mean,std):

	p = 1/(std*np.sqrt(2*np.pi))*np.exp((-(x-mean)**2)/(2*(std)**2))
	return p


label = []
for i in range(len(mamadataset)):
	a = P_benign*p_x_y(mamadataset['Bi-Rads'][i], benign_bi_mean,benign_bi_std)*p_x_y(mamadataset['Age'][i],benign_age_mean,benign_age_std)*\
	p_x_y(mamadataset['Shape'][i],benign_sh_mean,benign_sh_std)*p_x_y(mamadataset['Margin'][i],benign_ma_mean,benign_ma_std)*\
	p_x_y(mamadataset['Density'][i],benign_den_mean,benign_den_std)
	b = P_malignant*p_x_y(mamadataset['Bi-Rads'][i], malignant_bi_mean,malignant_bi_std)*p_x_y(mamadataset['Age'][i],malignant_age_mean,malignant_age_std)*\
	p_x_y(mamadataset['Shape'][i],malignant_sh_mean,malignant_sh_std)*p_x_y(mamadataset['Margin'][i],malignant_ma_mean,malignant_ma_std)*\
	p_x_y(mamadataset['Density'][i],malignant_den_mean,malignant_den_std)
	label.append([a,b])

ordem = np.array(label)
classe = np.argmax(ordem,axis=1)
print(classe)

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

plt.scatter(reduzida[:,0],reduzida[:,1],c=classe)
plt.title('Naive Bayes ~ PCA')
plt.legend()
plt.show()