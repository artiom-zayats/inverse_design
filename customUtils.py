import numpy as np
import math
from sklearn import preprocessing


def print_dic_params(dic,to_print,delimitor = "_",kvdelimitor = "_" ):
	dic_name_params = []
	for k in dic.keys():
		dic_name_params.append(k)

	dic_key_values = [] 
	print_buf = 30
	dic_name_params.sort()
	
	for key in dic_name_params:
		if(to_print):
			print(key," "*(print_buf - len(str(key)))," : ",str(dic[str(key)]))
		dic_key_values += delimitor + str(key) + kvdelimitor + str(dic[key])
	return ''.join(dic_key_values)


def clean_params(input_dic):

	clean_dic = input_dic.copy()
	for k in clean_dic.keys():
		string_version = str(clean_dic[k])
		string_version = string_version.replace("(","")
		string_version = string_version.replace(")","")
		string_version = string_version.replace("array","")
		string_version = string_version.replace("[","")
		string_version = string_version.replace("]","")
		string_version = string_version.replace("'","")
		clean_dic[k] = string_version

	

	return clean_dic


def mse_p(ix,Y_pred,Y_true):
	#Errors
	angs_pred = Y_pred[:,ix[0]:ix[1]]
	angs_true = Y_true[:,ix[0]:ix[1]]


	angle = np.arctan2(np.sin(angs_true - angs_pred), np.cos(angs_true - angs_pred))

	mse_angs = np.abs(angle).mean(axis = 0)

	diff_amp= abs(Y_pred[:,0:ix[0]] - Y_true[:,0:ix[0]])
	mse_amp = ((diff_amp) ** 2).mean(axis=0)


	mse_total = np.concatenate((mse_amp,mse_angs))
	return mse_total

def mse_p_single(Y_pred,Y_true):
	#Errors
	angs_pred = Y_pred%(math.pi*2)
	angs_true = Y_true
	diff_angs = abs(angs_pred - angs_true)
	mse_angs = ((diff_angs) ** 2).mean(axis=0)

	mse_total = mse_angs
	return mse_total


def setScaler():

	#scaler = preprocessing.QuantileTransformer(n_quantiles=200,output_distribution= 'uniform')
	#scaler = preprocessing.MinMaxScaler()
	#scaler = preprocessing.RobustScaler()
	scaler = preprocessing.StandardScaler()
	return scaler