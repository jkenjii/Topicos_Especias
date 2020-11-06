import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = np.genfromtxt('cancer1.csv', delimiter = ',', dtype = int)
print(data)

M = np.array(data)

atributos = ['id','Clump_Thickness','Cell_size','Cell_shape','Marginal_adhesion','Sigle_epi_cell_size','Bare_nuclei','Bland_chromantin','Normal_nucleoli','Mitoses','Classes']

cancerdataset = pd.DataFrame(data= M, columns= atributos)
cancerdataset = cancerdataset.drop(['id'],axis=1)
print(cancerdataset)

benign = cancerdataset['Classes'][cancerdataset['Classes'] == 2].count()
malignant = cancerdataset['Classes'][cancerdataset['Classes'] == 4].count()
total = cancerdataset['Classes'].count()

P_benign = benign/total
P_malignant = malignant/total
print(P_benign,P_malignant)

data_means = cancerdataset.groupby('Classes').mean()
print(data_means)
data_std = cancerdataset.groupby('Classes').std()
print(data_std)

#Means benign
benign_ct_mean = data_means['Clump_Thickness'][data_means.index == 2].values[0]
benign_cs_mean = data_means['Cell_size'][data_means.index == 2].values[0]
benign_csh_mean = data_means['Cell_shape'][data_means.index == 2].values[0]
benign_ma_mean = data_means['Marginal_adhesion'][data_means.index == 2].values[0]
benign_secs_mean = data_means['Sigle_epi_cell_size'][data_means.index == 2].values[0]
benign_bn_mean = data_means['Bare_nuclei'][data_means.index == 2].values[0]
benign_bch_mean = data_means['Bland_chromantin'][data_means.index == 2].values[0]
benign_nn_mean = data_means['Normal_nucleoli'][data_means.index == 2].values[0]
benign_mi_mean = data_means['Mitoses'][data_means.index == 2].values[0]

#means malignant

malignant_ct_mean = data_means['Clump_Thickness'][data_means.index == 4].values[0]
malignant_cs_mean = data_means['Cell_size'][data_means.index == 4].values[0]
malignant_csh_mean = data_means['Cell_shape'][data_means.index == 4].values[0]
malignant_ma_mean = data_means['Marginal_adhesion'][data_means.index == 4].values[0]
malignant_secs_mean = data_means['Sigle_epi_cell_size'][data_means.index == 4].values[0]
malignant_bn_mean = data_means['Bare_nuclei'][data_means.index == 4].values[0]
malignant_bch_mean = data_means['Bland_chromantin'][data_means.index == 4].values[0]
malignant_nn_mean = data_means['Normal_nucleoli'][data_means.index == 4].values[0]
malignant_mi_mean = data_means['Mitoses'][data_means.index == 4].values[0]

#std benign

benign_ct_std = data_std['Clump_Thickness'][data_std.index == 2].values[0]
benign_cs_std = data_std['Cell_size'][data_std.index == 2].values[0]
benign_csh_std = data_std['Cell_shape'][data_std.index == 2].values[0]
benign_ma_std = data_std['Marginal_adhesion'][data_std.index == 2].values[0]
benign_secs_std = data_std['Sigle_epi_cell_size'][data_std.index == 2].values[0]
benign_bn_std = data_std['Bare_nuclei'][data_std.index == 2].values[0]
benign_bch_std = data_std['Bland_chromantin'][data_std.index == 2].values[0]
benign_nn_std = data_std['Normal_nucleoli'][data_std.index == 2].values[0]
benign_mi_std = data_std['Mitoses'][data_std.index == 2].values[0]

#std malignant

malignant_ct_std = data_std['Clump_Thickness'][data_std.index == 4].values[0]
malignant_cs_std = data_std['Cell_size'][data_std.index == 4].values[0]
malignant_csh_std = data_std['Cell_shape'][data_std.index == 4].values[0]
malignant_ma_std = data_std['Marginal_adhesion'][data_std.index == 4].values[0]
malignant_secs_std = data_std['Sigle_epi_cell_size'][data_std.index == 4].values[0]
malignant_bn_std = data_std['Bare_nuclei'][data_std.index == 4].values[0]
malignant_bch_std = data_std['Bland_chromantin'][data_std.index == 4].values[0]
malignant_nn_std = data_std['Normal_nucleoli'][data_std.index == 4].values[0]
malignant_mi_std = data_std['Mitoses'][data_std.index == 4].values[0]

def p_x_y(x,mean,std):

	p = 1/(std*np.sqrt(2*np.pi))*np.exp((-(x-mean)**2)/(2*(std)**2))
	return p

label = []
for i in range(len(cancerdataset)):
	a = P_benign*p_x_y(cancerdataset['Clump_Thickness'][i], benign_ct_mean,benign_ct_std)*p_x_y(cancerdataset['Cell_size'][i],benign_cs_mean,benign_ct_std)*\
	p_x_y(cancerdataset['Cell_shape'][i],benign_csh_mean,benign_csh_std)*p_x_y(cancerdataset['Marginal_adhesion'][i],benign_ma_mean,benign_ma_std)*\
	p_x_y(cancerdataset['Sigle_epi_cell_size'][i],benign_secs_mean,benign_secs_std)*p_x_y(cancerdataset['Bare_nuclei'][i],benign_bn_mean,benign_bn_std)*\
	p_x_y(cancerdataset['Bland_chromantin'][i],benign_bch_mean,benign_bch_std)*p_x_y(cancerdataset['Normal_nucleoli'][i],benign_nn_mean,benign_nn_std)*\
	p_x_y(cancerdataset['Mitoses'][i],benign_mi_mean,benign_mi_std)

	b = P_malignant*p_x_y(cancerdataset['Clump_Thickness'][i], malignant_ct_mean,malignant_ct_std)*p_x_y(cancerdataset['Cell_size'][i],malignant_cs_mean,malignant_ct_std)*\
	p_x_y(cancerdataset['Cell_shape'][i],malignant_csh_mean,malignant_csh_std)*p_x_y(cancerdataset['Marginal_adhesion'][i],malignant_ma_mean,malignant_ma_std)*\
	p_x_y(cancerdataset['Sigle_epi_cell_size'][i],malignant_secs_mean,malignant_secs_std)*p_x_y(cancerdataset['Bare_nuclei'][i],malignant_bn_mean,malignant_bn_std)*\
	p_x_y(cancerdataset['Bland_chromantin'][i],malignant_bch_mean,malignant_bch_std)*p_x_y(cancerdataset['Normal_nucleoli'][i],malignant_nn_mean,malignant_nn_std)*\
	p_x_y(cancerdataset['Mitoses'][i],malignant_mi_mean,benign_mi_std)
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