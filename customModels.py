import numpy as np
np.random.seed(1003)
import pandas as pd
from sklearn.model_selection import ParameterGrid
import progressbar
import os
import time
import pickle
import prep
import sys
import pdb


# Normalize data between 0 and 1
from sklearn import preprocessing


#Suppress messages from tensorflow
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
from keras import callbacks
from keras import layers
from keras import optimizers
from keras.models import Sequential, Model
import keras.backend as K
sys.stderr = stderr
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Stop TensorFlow Debugging Info



def SimpleNNMultiOutputModel(model_params):

	input_layer = layers.Input(shape = (len(model_params['InputNames']),))

	x = layers.Dense(model_params["HiddenLayers"], activation = model_params["Activiation"])(input_layer)
	#Hidden - Layers
	for layer in range(model_params["NumOfHiddenLayers"]):
		if model_params["DropOut"]:
			x = layers.Dropout(0.2, seed = 1003)(x)
			#x = layers.Dropout(0.2)(x)
		x = layers.Dense(model_params["HiddenLayers"],activation = model_params["Activiation"])(x)

	# Output- Layer
	out_layers_list =[]
	mid_layers_name_list = model_params["OutputNames"]

	for i in range(len(mid_layers_name_list)):
		out_layers_list.append(layers.Dense(1, activation = model_params["Final_Activation"],name=mid_layers_name_list[i])(x))

	model = Model(inputs=input_layer, outputs=out_layers_list)

	#model.summary()

	return model



def SimpleNNEncodeDecoder(model_params):

	input_layer = layers.Input(shape = (len(model_params['InputNames']),))

	x = layers.Dense(model_params["HiddenLayers"], activation = model_params["Activiation"])(input_layer)
	#Hidden - Layers
	for layer in range(model_params["NumOfHiddenLayers"]):
		if model_params["DropOut"]:
			x = layers.Dropout(0.2, seed = 1003)(x)
		x = layers.Dense(model_params["HiddenLayers"],activation = model_params["Activiation"])(x)

	# Output- Layer
	mid_layers_list =[]
	mid_layers_name_list = model_params["OutputNames"]

	for i in range(len(mid_layers_name_list)):
		mid_layers_list.append(layers.Dense(1, activation = model_params["Final_Activation"],name=mid_layers_name_list[i])(x))

	input_layer2 = layers.Concatenate()(mid_layers_list)

	x = layers.Dense(model_params["HiddenLayers"], activation = model_params["Activiation"])(input_layer2)
	#Hidden - Layers
	for layer in range(model_params["NumOfHiddenLayers"]):
		if model_params["DropOut"]:
			x = layers.Dropout(0.2, seed = 1003)(x)
		x = layers.Dense(model_params["HiddenLayers"],activation = model_params["Activiation"])(x)

	# Output- Layer
	out_layers_list =[]
	out_layers_name_list = model_params["InputNames"]

	for i in range(len(out_layers_name_list)):
		out_layers_list.append(layers.Dense(1, activation = model_params["Final_Activation"],name=out_layers_name_list[i])(x))


	model = Model(inputs=input_layer, outputs=mid_layers_list+out_layers_list )

	#model.summary()

	return model


def DistributionalNNEncodeDecoder(model_params):

	input_layer = layers.Input(shape = (len(model_params['InputNames']),))

	x = layers.Dense(model_params["HiddenLayers"], activation = model_params["Activiation"])(input_layer)
	#Hidden - Layers
	for layer in range(model_params["NumOfHiddenLayers"]):
		if model_params["DropOut"]:
			x = layers.Dropout(0.2, seed = 1003)(x)
		x = layers.Dense(model_params["HiddenLayers"],activation = model_params["Activiation"])(x)

	# Output- Layer
	mid_layers_list =[]
	mid_layers_name_list = model_params["OutputNames"]    #T,R,P

	for i in range(len(mid_layers_name_list)):
		mu = layers.Dense(1, activation = "softplus", name = "mean_"+mid_layers_name_list[i])(x)
		variance = layers.Dense(1, activation = "softplus" , name = "var_"+mid_layers_name_list[i])(x)
		dist = layers.Concatenate(name=mid_layers_name_list[i])([mu,variance])
		mid_layers_list.append(dist)


	input_layer2 = layers.Concatenate()(mid_layers_list)

	x = layers.Dense(model_params["HiddenLayers"], activation = model_params["Activiation"])(input_layer2)
	#Hidden - Layers
	for layer in range(model_params["NumOfHiddenLayers"]):
		if model_params["DropOut"]:
			x = layers.Dropout(0.2, seed = 1003)(x)
		x = layers.Dense(model_params["HiddenLayers"],activation = model_params["Activiation"])(x)

	# Output- Layer
	out_layers_list =[]
	out_layers_name_list = model_params["InputNames"]

	for i in range(len(out_layers_name_list)):
		out_layers_list.append(layers.Dense(1, activation = model_params["Final_Activation"],name=out_layers_name_list[i])(x))


	model = Model(inputs=input_layer, outputs=mid_layers_list + out_layers_list)



	return model


def DistNNMultiOutputModel(model_params):

	input_layer = layers.Input(shape = (len(model_params['InputNames']),))

	x = layers.Dense(model_params["HiddenLayers"], activation = model_params["Activiation"])(input_layer)
	#Hidden - Layers
	for layer in range(model_params["NumOfHiddenLayers"]):
		if model_params["DropOut"]:
			x = layers.Dropout(0.2, seed = 1003)(x)
		x = layers.Dense(model_params["HiddenLayers"],activation = model_params["Activiation"])(x)

	# Output- Layer
	out_layers_list =[]
	mid_layers_name_list = model_params["OutputNames"]

	for i in range(len(mid_layers_name_list)):
		mu = layers.Dense(1, activation = "softplus", name = "mean_"+mid_layers_name_list[i])(x)
		variance = layers.Dense(1, activation = "softplus" , name = "var_"+mid_layers_name_list[i])(x)
		dist = layers.Concatenate(name=mid_layers_name_list[i])([mu,variance])
		out_layers_list.append(dist)

	model = Model(inputs=input_layer, outputs=out_layers_list)

	#model.summary()

	return model





def SimpleNNMultiOutputModelConcat(model_params):

	input_layer = layers.Input(shape = (len(model_params['InputNames']),))

	x = layers.Dense(model_params["HiddenLayers"], activation = model_params["Activiation"])(input_layer)
	#Hidden - Layers
	for layer in range(model_params["NumOfHiddenLayers"]):
		if model_params["DropOut"]:
			x = layers.Dropout(0.2, seed = 1003)(x)
			#x = layers.Dropout(0.2)(x)
		x = layers.Dense(model_params["HiddenLayers"],activation = model_params["Activiation"])(x)

	# Output- Layer
	out_layers_list =[]
	mid_layers_name_list = model_params["OutputNames"]

	for i in range(len(mid_layers_name_list)):
		out_layers_list.append(layers.Dense(1, activation = model_params["Final_Activation"],name=mid_layers_name_list[i])(x))

	output_layer = layers.Concatenate(name = "out_"+"".join(model_params["OutputNames"]))(out_layers_list)


	model = Model(inputs=input_layer, outputs=output_layer , name = "".join(model_params["OutputNames"]))

	#model.summary()

	return model



def DistNNMultiOutputModelCustom(model_params):

	input_layer = layers.Input(shape = (len(model_params['InputNames']),))

	x = layers.Dense(model_params["HiddenLayers"], activation = model_params["Activiation"])(input_layer)
	#Hidden - Layers
	for layer in range(model_params["NumOfHiddenLayers"]):
		if model_params["DropOut"]:
			x = layers.Dropout(0.2, seed = 1003)(x)
		x = layers.Dense(model_params["HiddenLayers"],activation = model_params["Activiation"])(x)

	# Output- Layer
	out_layers_list =[]
	mid_layers_name_list = model_params["OutputNames"]


	def sampling(args):
		mu, variance = args
		epsilon = K.random_normal(shape=K.tf.shape(mu))

		return mu + K.sqrt(variance) * epsilon

	for i in range(len(mid_layers_name_list)):
		mu = layers.Dense(1, activation = "softplus", name = "mean_"+mid_layers_name_list[i])(x)
		variance = layers.Dense(1, activation = "softplus" , name = "var_"+mid_layers_name_list[i])(x)

		sample = layers.Lambda(sampling, output_shape=(1,),name = 'sampling_'+mid_layers_name_list[i])([mu, variance])

		out_layers_list.append(sample)

	output_layer = layers.Concatenate(name = "out_"+"".join(model_params["OutputNames"]))(out_layers_list)

	model = Model(inputs=input_layer, outputs=output_layer,name = "".join(model_params["OutputNames"]))

	#model.summary()

	return model