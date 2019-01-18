
import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical   
import pdb
import os
from sklearn import preprocessing


def loadData():
	#Encoding
	df_train = pd.read_pickle(os.path.join("data","train.pkl"))
	df_dev = pd.read_pickle(os.path.join("data","dev.pkl"))
	df_test = pd.read_pickle(os.path.join("data","test.pkl"))
	AP_train,TRP_train = encode(df_train) 
	AP_dev,TRP_dev = encode(df_dev)
	AP_test,TRP_test = encode(df_test)

	train = (AP_train,TRP_train)
	dev = (AP_dev,TRP_dev)
	test = (AP_test,TRP_test)

	return (train,dev,test)


#encoding data to
def encode(df):
	df = df.apply( pd.to_numeric, errors='coerce' )

	AP = np.array([df['A_1'],df['A_2'],df['A_3'],df['phi_1'],df['phi_2'],df['phi_3']])
	TRP = np.array([df['T'],df['R'],df['P']])

	AP = np.transpose(AP)
	TRP = np.transpose(TRP)

	return (AP, TRP )


def encode_classification(df,shuffle,radnom_seed,bins_num):
	df = df.apply( pd.to_numeric, errors='coerce' )

	namesList = ['T','R','P_1','P_2']
	#bins_num = 20
	output = {}

	for name in namesList:
		min_B = df[name].min()
		max_B = df[name].max()
		bins  = np.linspace(min_B,max_B,bins_num)
		digitezed = np.digitize(df[name],bins)
		data = to_categorical(digitezed-1, bins_num)

		output[name] = []
		output[name].append(min_B)
		output[name].append(max_B)
		output[name].append(bins_num)
		output[name].append(bins)
		output[name].append(data) #4


	if (shuffle):
		 df.sample(frac = 1,random_state = radnom_seed)

	x = np.array([df['A_1'],df['A_2'],df['A_3'],df['phi_1'],df['phi_2'],df['phi_3']])
	#y = np.array([df['T'],df['R'],df['P_1'],df['P_2']])
	y = np.array([output['T'][4],output['R'][4],output['P_1'][4],output['P_2'][4]])
	y = y.reshape((y.shape[1], y.shape[0],y.shape[2]))


	x = np.transpose(x)

	return (x, y , y[:,0] ,y[:,1] ,y[:,2] ,y[:,3] ) 


def convert_to_hot(arr,bins_num):
	pdb.set_trace()
	bins  = np.linspace(arr.min,max_B,bins_num)

def getScaler():

	#scaler = preprocessing.QuantileTransformer(n_quantiles=200,output_distribution= 'uniform')
	#scaler = preprocessing.MinMaxScaler()
	#scaler = preprocessing.RobustScaler()
	scaler = preprocessing.StandardScaler()
	return scaler



def scaleData(X_train,X_dev):
	#Scaling
	#scaler_X = preprocessing.StandardScaler()
	scaler_X = getScaler()
	scaler_X.fit(X_train)
	X_train = scaler_X.transform(X_train)
	X_dev = scaler_X.transform(X_dev)
	return X_train,X_dev,scaler_X






