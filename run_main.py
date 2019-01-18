import numpy as np

np.random.seed(1003)
import pandas as pd
import os
import time
import datetime
import pickle
import math
import sys
import pdb
import json
import re


#Suppress messages from tensorflow
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
from keras import callbacks
from keras import layers
from keras import optimizers
from keras.models import  Model
from keras import backend as K
from keras.models import load_model

import tensorflow as tf
sys.stderr = stderr
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Stop TensorFlow Debugging Info

import sklearn.linear_model
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import ParameterGrid
from sklearn import preprocessing

## Mine IMPORTS:
import customModels
import customCallbacks
import customUtils
from customMetrics import atan_mse,modulo_2pi_error,log_likelihood_normal_cost,nll,mean_squared_error_mu,acc_dist,cv_test,custom_loss,phase_mse,smape,custom_loss_TRP,custom_metroc_TRP
import prep
from run_eval import run_eval_base
import run_eval
import eval_ensemble


import email_logger

RANDOM_SEED = 1003


EPSILON = np.finfo(np.float32).eps









def runSimpleModelWithParams(model_params,data):

	
	#model_params
	BATCH_SIZE = model_params["batchSize"]
	np.random.seed(RANDOM_SEED)
	NUM_EPOCHS = model_params['num_epochs']
	#data
	AP_train,TRP_train = data[0]
	AP_dev,TRP_dev = data[1]

	#Check direction
	if (model_params["DirectionForward"]):
		X_train,Y_train,X_dev,Y_dev = TRP_train,AP_train,TRP_dev,AP_dev
	else:
		X_train,Y_train,X_dev,Y_dev = AP_train,TRP_train,AP_dev,TRP_dev
		model_params["OutputNames"],model_params["InputNames"] = model_params["InputNames"],model_params["OutputNames"]


	model = customModels.SimpleNNMultiOutputModel(model_params)

	if model_params["scaling"]:
		X_train,X_dev,_ = prep.scaleData(X_train,X_dev)


	metrics_vect = {}
	for k in (model_params["OutputNames"]):
		if "ph" in k:
			metrics_vect[k] = [atan_mse] + ["mse"] + [modulo_2pi_error] + [phase_mse]+ [smape]
		else:
			metrics_vect[k] =  ["mse"] + [smape] + [custom_loss]

	losses_vect = {}
	losses_weights = {}
	for k in (model_params["OutputNames"]):
		if "ph" in k:
			if model_params["phase_loss"] == "mse":
				losses_vect[k] = "mse"
			elif model_params["phase_loss"] == "phase_mse":
				losses_vect[k] = phase_mse
			elif model_params["phase_loss"] == "atan_mse":
				losses_vect[k] =  atan_mse
			else:
				raise NotImplementedError("not valid phase_loss")
			#losses_weights[k] = 1
		elif k in model_params["OutputNames"]:
			losses_vect[k] =  "mse"
			#losses_vect[k] =  custom_loss
			#losses_weights[k] = 1
		else:
			losses_vect[k] =  "mse"
			#losses_weights[k] = 1

	if model_params["use_weights"]:
		wights =  (Y_train.mean(axis = 0)).tolist()
	else:
		wights = [1]*len(model_params["OutputNames"])

	for i,k in enumerate(model_params["OutputNames"]):
		f = model_params["weights_factor"]
		w = wights[i]
		losses_weights[k] = (1/w**2)


	#losses_vect["phi_1"] = atan_mse
	#losses_vect["phi_2"] = atan_mse
	#losses_vect["phi_3"] = phase_mse



	model.compile(
	 optimizer = "adam",
	 loss = losses_vect,
	 loss_weights = losses_weights,
	 metrics = metrics_vect
	)

	#CALBACKS SETUP
	print("Local Training Params:")
	customUtils.print_dic_params(model_params,True,delimitor = "_",kvdelimitor = "_" )
	print("="*50)


	callback_list = customCallbacks.addCallBacks(model_params)

	results = model.fit(
	 X_train,
	 dict(zip(model_params["OutputNames"],[row for row in Y_train.T])),
	 epochs = NUM_EPOCHS ,
	 batch_size = model_params["batchSize"],
	 validation_data = (
	 X_dev, 
	 dict(zip(model_params["OutputNames"],[row for row in Y_dev.T]))
	 ),
	 verbose =0,
	 callbacks = callback_list,
	 shuffle = True
	)

	#run eval:
	model_location = os.path.join('models',model_params["model_name"] +  '.hdf5')
	with open(os.path.join('model_params',model_params["model_name"] +  '.json'), 'w') as fp:
		json.dump(model_params, fp, sort_keys=True)


	#eval_ensemble.run_ensemble_eval(models_locations = [model_location])
	#pdb.set_trace()


	mse_total_train = run_eval_base(model_location,dataset = "train",email = model_params["email"])
	mse_total_dev = run_eval_base(model_location,dataset = "dev",email = model_params["email"])
	mse_total_test = run_eval_base(model_location,dataset = "test",email = model_params["email"])

	return (
		mse_total_train[0:len(model_params["OutputNames"])],
		mse_total_dev[0:len(model_params["OutputNames"])],
		mse_total_test[0:len(model_params["OutputNames"])]
		)

def runSimpleEncoderDecoderModelWithParams(model_params,data):

	np.random.seed(RANDOM_SEED)
	NUM_EPOCHS = model_params['num_epochs']
	#data
	AP_train,TRP_train = data[0]
	AP_dev,TRP_dev = data[1]




	#Check direction
	if (model_params["DirectionForward"]):
		X_train,Y_train,X_dev,Y_dev = TRP_train,AP_train,TRP_dev,AP_dev
	else:
		X_train,Y_train,X_dev,Y_dev = AP_train,TRP_train,AP_dev,TRP_dev
		model_params["OutputNames"],model_params["InputNames"] = model_params["InputNames"],model_params["OutputNames"]

	model = customModels.SimpleNNEncodeDecoder(model_params)


	if model_params["scaling"]:
		X_train,X_dev,_ = prep.scaleData(X_train,X_dev)


	#wights = [1]*len((model_params["OutputNames"] + model_params["InputNames"]))


	metrics_vect = {}
	for k in (model_params["OutputNames"] + model_params["InputNames"]):
		if "ph" in k:
			metrics_vect[k] = [atan_mse] + ["mse"] + [modulo_2pi_error] + [phase_mse] + [smape]
			#metrics_vect[k] = ["mse"] 
		else:
			metrics_vect[k] =  ["mse"] + [smape]

	losses_vect = {}
	losses_weights = {}
	for k in (model_params["OutputNames"] + model_params["InputNames"]):
		if "ph" in k:

			if model_params["phase_loss"] == "mse":
				losses_vect[k] = "mse"
			elif model_params["phase_loss"] == "phase_mse":
				losses_vect[k] = phase_mse
			elif model_params["phase_loss"] == "atan_mse":
				losses_vect[k] =  atan_mse
			else:
				raise NotImplementedError("not valid phase_loss")
			#losses_weights[k] = 1
		elif k in model_params["OutputNames"]:
			#losses_vect[k] =  "mse"
			losses_vect[k] =  custom_loss

			#losses_weights[k] = 1
		else:
			losses_vect[k] =  "mse"
			#losses_weights[k] = 1

	if model_params["use_weights"]:
		wights =  (Y_train.mean(axis = 0)).tolist() + X_train.mean(axis = 0).tolist()
	else:
		wights = [1]*len((model_params["OutputNames"] + model_params["InputNames"]))

	for i,k in enumerate(model_params["OutputNames"] + model_params["InputNames"]):
		f = model_params["weights_factor"]
		w = wights[i]
		if k in model_params["OutputNames"]:
			losses_weights[k] = f*(1/w**2)
		else:
			if model_params["scaling"]:
				losses_weights[k] = 1
			else:
				losses_weights[k] = (1/w**2)







	model.compile(
	 optimizer = "adam",
	 loss = losses_vect,
	 loss_weights = losses_weights,
	 metrics = metrics_vect
	)
	#CALBACKS SETUP
	print("Local Training Params:")
	customUtils.print_dic_params(model_params,True,delimitor = "_",kvdelimitor = "_" )
	print("="*50)
	callback_list = customCallbacks.addCallBacks(model_params)

	results = model.fit(
	 X_train,
	 dict(zip(model_params["OutputNames"]+model_params["InputNames"],[row for row in Y_train.T] + [row for row in X_train.T])),
	 epochs = NUM_EPOCHS ,
	 batch_size = model_params["batchSize"],
	 validation_data = (
	 X_dev, 
	 dict(zip(model_params["OutputNames"]+model_params["InputNames"],[row for row in Y_dev.T] + [row for row in X_dev.T]))),
	 verbose = 0,
	 callbacks = callback_list,
	 shuffle = True
	)

	#run eval:

	model_location = os.path.join('models',model_params["model_name"] +  '.hdf5')
	with open(os.path.join('model_params',model_params["model_name"] +  '.json'), 'w') as fp:
		json.dump(model_params, fp, sort_keys=True)


	mse_total_train = run_eval_base(model_location,dataset = "train", email = model_params["email"])
	mse_total_dev = run_eval_base(model_location,dataset = "dev",email = model_params["email"])
	mse_total_test = run_eval_base(model_location,dataset = "test",email = model_params["email"])

	return (
		mse_total_train[0:len(model_params["OutputNames"])],
		mse_total_dev[0:len(model_params["OutputNames"])],
		mse_total_test[0:len(model_params["OutputNames"])]
		)





def runDistEncoderDecoderModelWithParams(model_params,data):


	#model_params
	np.random.seed(RANDOM_SEED)
	NUM_EPOCHS = model_params['num_epochs']
	#data
	AP_train,TRP_train = data[0]
	AP_dev,TRP_dev = data[1]

	#Check direction
	if (model_params["DirectionForward"]):
		X_train,Y_train,X_dev,Y_dev = TRP_train,AP_train,TRP_dev,AP_dev
	else:
		X_train,Y_train,X_dev,Y_dev = AP_train,TRP_train,AP_dev,TRP_dev
		model_params["OutputNames"],model_params["InputNames"] = model_params["InputNames"],model_params["OutputNames"]

	model = customModels.DistributionalNNEncodeDecoder(model_params)

	if model_params["scaling"]:
		X_train,X_dev,_ = prep.scaleData(X_train,X_dev)

	metrics_vect = {}
	for k in (model_params["OutputNames"] + model_params["InputNames"]):
		if "ph" in k:
			metrics_vect[k] = [atan_mse] + ["mse"] + [modulo_2pi_error] + [smape]

		elif k in model_params["OutputNames"]:
			metrics_vect[k] =  [nll] + [log_likelihood_normal_cost] + [mean_squared_error_mu] +[acc_dist] + [cv_test] + ["mse"] + [smape]
		else:
			metrics_vect[k] =  ["mse"] + [smape]

	losses_vect = {}
	losses_weights = {}
	for k in (model_params["OutputNames"] + model_params["InputNames"]):
		if "ph" in k:
			if model_params["phase_loss"] == "mse":
				losses_vect[k] = "mse"
			elif model_params["phase_loss"] == "phase_mse":
				losses_vect[k] = phase_mse
			elif model_params["phase_loss"] == "atan_mse":
				losses_vect[k] =  atan_mse
			else:
				raise NotImplementedError("not valid phase_loss")
		if k in model_params["OutputNames"]:
			losses_vect[k] =  log_likelihood_normal_cost
			#losses_vect[k] =  custom_loss
		else:
			losses_vect[k] =  "mse"
			#losses_weights[k] = 1



	if model_params["use_weights"]:
		wights =  (Y_train.mean(axis = 0)).tolist() + X_train.mean(axis = 0).tolist()
	else:
		wights = [1]*len((model_params["OutputNames"] + model_params["InputNames"]))


	f = model_params["weights_factor"]
	for i,k in enumerate(model_params["OutputNames"] + model_params["InputNames"]):
		w = wights[i]
		f = model_params["weights_factor"]

		if k in model_params["OutputNames"]:
			#losses_weights[k] = (1/model_params["weights_factor"])*(-1/math.log(wights[i]**2))
			#losses_weights[k] = (model_params["weights_factor"])*(1/wights[i]**2)
			losses_weights[k] = f*(w)
		else:
			if model_params["scaling"]:
				losses_weights[k] = 1
			else:
				losses_weights[k] = (1/w**2)


	model_params["weight_for_loss"] = losses_weights

	




	model.compile(
	 optimizer = "adam",
	 loss = losses_vect,
	 loss_weights = losses_weights,
	 metrics = metrics_vect
	)

	#CALBACKS SETUP
	print("Local Training Params:")
	customUtils.print_dic_params(model_params,True,delimitor = "_",kvdelimitor = "_" )
	print("="*50)

	callback_list = customCallbacks.addCallBacks(model_params)



	results = model.fit(
	 X_train,
	 dict(zip(model_params["OutputNames"]+model_params["InputNames"],[np.stack((row,row), axis = 0).T for row in Y_train.T] + [row for row in X_train.T])),
	 epochs = NUM_EPOCHS ,
	 batch_size = model_params["batchSize"],
	 validation_data = 	 (X_dev, 
	 dict(zip(model_params["OutputNames"]+model_params["InputNames"],[np.stack((row,row), axis = 0).T for row in Y_dev.T] + [row for row in X_dev.T])),
	 ),
	 verbose =0,
	 callbacks = callback_list,
	 shuffle = True
	)





	


	model_location = os.path.join('models',model_params["model_name"] +  '.hdf5')
	with open(os.path.join('model_params',model_params["model_name"] +  '.json'), 'w') as fp:
		json.dump(model_params, fp, sort_keys=True)


	mse_total_train = run_eval_base(model_location,dataset = "train",email = model_params["email"])

	mse_total_dev = run_eval_base(model_location,dataset = "dev",email = model_params["email"])

	mse_total_test = run_eval_base(model_location,dataset = "test",email = model_params["email"])

	return (
		mse_total_train[0:len(model_params["OutputNames"])],
		mse_total_dev[0:len(model_params["OutputNames"])],
		mse_total_test[0:len(model_params["OutputNames"])]
		)


def runDistModelWithParams(model_params,data):


	#model_params
	np.random.seed(RANDOM_SEED)
	NUM_EPOCHS = model_params['num_epochs']
	#data
	AP_train,TRP_train = data[0]
	AP_dev,TRP_dev = data[1]

	#Check direction
	if (model_params["DirectionForward"]):
		X_train,Y_train,X_dev,Y_dev = TRP_train,AP_train,TRP_dev,AP_dev
	else:
		X_train,Y_train,X_dev,Y_dev = AP_train,TRP_train,AP_dev,TRP_dev
		model_params["OutputNames"],model_params["InputNames"] = model_params["InputNames"],model_params["OutputNames"]

	if model_params["scaling"]:
		X_train,X_dev,_ = prep.scaleData(X_train,X_dev)


	model = customModels.DistNNMultiOutputModel(model_params)

	metrics_vect = {}
	for k in (model_params["OutputNames"]):
		if "ph" in k:
			metrics_vect[k] = [atan_mse] + ["mse"] + [modulo_2pi_error] + [mean_squared_error_mu] + [smape]
		elif k in ["T","R","P"]:
			metrics_vect[k] =  [nll] + [log_likelihood_normal_cost] + [mean_squared_error_mu] +[acc_dist] + [cv_test] + ["mse"] + [smape]
		else:
			metrics_vect[k] =  ["mse"] + [mean_squared_error_mu] + [smape]

	losses_vect = {}
	losses_weights = {}

	for k in (model_params["OutputNames"]):
		if "ph" in k:
			losses_vect[k] = "mse"
			#losses_weights[k] = 1
		elif k in ["T","R","P"]:
			losses_vect[k] =  log_likelihood_normal_cost
			#losses_vect[k] =  custom_loss
			#losses_weights[k] = 1
		else:
			losses_vect[k] =  "mse"
			#losses_weights[k] = 1


	if model_params["use_weights"]:
		wights =  (Y_train.mean(axis = 0)).tolist()
	else:
		wights = [1]*len(model_params["OutputNames"])


	for i,k in enumerate(model_params["OutputNames"]):
		f = model_params["weights_factor"]
		w = wights[i]
		losses_weights[k] = w  #0.4/0.12/0.4





	model.compile(
	 optimizer = "adam",
	 loss = losses_vect,
	 loss_weights = losses_weights,
	 metrics = metrics_vect
	)

	#CALBACKS SETUP
	print("Local Training Params:")
	customUtils.print_dic_params(model_params,True,delimitor = "_",kvdelimitor = "_" )
	print("="*50)



	callback_list = customCallbacks.addCallBacks(model_params)


	results = model.fit(
	 X_train,
	 dict(zip(model_params["OutputNames"],[np.stack((row,row), axis = 0).T for row in Y_train.T])),
	 epochs = NUM_EPOCHS ,
	 batch_size = model_params["batchSize"],
	 validation_data = (
	 X_dev, 
	 dict(zip(model_params["OutputNames"],[np.stack((row,row), axis = 0).T for row in Y_dev.T])),
	 ),
	 verbose =0,
	 callbacks = callback_list,
	 shuffle = True
	)

	


	model_location = os.path.join('models',model_params["model_name"] +  '.hdf5')
	with open(os.path.join('model_params',model_params["model_name"] +  '.json'), 'w') as fp:
		json.dump(model_params, fp, sort_keys=True)


	mse_total_train = run_eval_base(model_location,dataset = "train",email = model_params["email"])
	mse_total_dev = run_eval_base(model_location,dataset = "dev",email = model_params["email"])
	mse_total_test = run_eval_base(model_location,dataset = "test",email = model_params["email"])

	return (
		mse_total_train[0:len(model_params["OutputNames"])],
		mse_total_dev[0:len(model_params["OutputNames"])],
		mse_total_test[0:len(model_params["OutputNames"])]
		)



def runCustomAutpEncoderWithParams(model_params,data):


	#model_params
	np.random.seed(RANDOM_SEED)
	NUM_EPOCHS = model_params['num_epochs']
	#data
	AP_train,TRP_train = data[0]
	AP_dev,TRP_dev = data[1]

	#Check direction
	if (model_params["DirectionForward"]):
		X_train,Y_train,X_dev,Y_dev = TRP_train,AP_train,TRP_dev,AP_dev
	else:
		X_train,Y_train,X_dev,Y_dev = AP_train,TRP_train,AP_dev,TRP_dev
		model_params["OutputNames"],model_params["InputNames"] = model_params["InputNames"],model_params["OutputNames"]

	if model_params["scaling"]:
		X_train,X_dev,_ = prep.scaleData(X_train,X_dev)


	model_params["InverseModelLocation"]  = "models/Forward_BaseLine_MSE_run_0_time_2018_11_07_00_01_09.hdf5" # mse based
	#model_params["InverseModelLocation"] = "models/Fowrard_BaseLine_run_0_time_2018_11_05_19_05_19.hdf5" # atanmse based
	model_params["ForwardModelLocation"] = os.path.join('models',model_params["model_name"] + "forward" +  '.hdf5')

	model_location = os.path.join('models',model_params["model_name"] +  '.hdf5')
	weights_location = os.path.join('weights',model_params["model_name"] +  '.h5')

	#model_forward = customModels.SimpleNNMultiOutputModelConcat(model_params)

	model_forward = customModels.DistNNMultiOutputModelCustom(model_params)


	#pdb.set_trace()





	model_reverse = run_eval.loadBestModel(model_params["InverseModelLocation"])
	model_reverse.trainable = False
	#model_reverse.compile(optimizer = "adam", loss = ["mse","mse","mse",atan_mse,atan_mse,phase_mse])

	inputs = model_forward.input
	outputs = model_forward.outputs + model_reverse(model_forward(model_forward.inputs))
	

	combined_model = keras.models.Model(inputs = inputs  , outputs = outputs )
	#combined_model = keras.models.Model(inputs = model_forward.inputs  , outputs = model_forward.outputs + model_reverse(model_forward.outputs) )

	keras.utils.plot_model(combined_model,to_file=os.path.join('images',model_params["model_name"] +  '.png'),show_shapes=True)
	#email_logger.send_email(text_mesage = "Test", title_text = "ModelImage",image_path = 'demo.png')

	#pdb.set_trace()



	model_forward.save(model_params["ForwardModelLocation"])




	"""
	combined_model.compile(
	 optimizer = keras.optimizers.adam(),
	 loss = ["mse","mse","mse","mse","mse","mse",atan_mse,atan_mse,phase_mse],
	 loss_weights = [10,10,10,1,1,1,10,10,10],
	 #metrics = "mse"
	)
	"""

	combined_model.compile(
	 #optimizer = keras.optimizers.Adadelta(lr = 1.0),
	 optimizer = keras.optimizers.RMSprop(lr = 0.001),
	 #optimizer = keras.optimizers.Nadam(),
	 #optimizer = keras.optimizers.Adam(lr=0.001),
	 loss = [custom_loss,"mse","mse","mse","mse","mse","mse"],
	 #custom loss - zero
	 loss_weights = [1,0.1,0.1,0.1,0.3,0.3,0.3],
	 metrics = {"out_TRP":[custom_metroc_TRP,custom_loss_TRP]}
	)
	combined_model.save(model_location)



	#CALBACKS SETUP
	print("Local Training Params:")
	customUtils.print_dic_params(model_params,False	,delimitor = "_",kvdelimitor = "_" )
	print("="*50)



	callback_list = customCallbacks.addCallBacks(model_params)


	
	input_train = [X_train]
	output_train = [Y_train,X_train[:,0],X_train[:,1],X_train[:,2],X_train[:,3],X_train[:,4],X_train[:,5]]
	input_dev = [X_dev]
	output_dev = [Y_dev,X_dev[:,0],X_dev[:,1],X_dev[:,2],X_dev[:,3],X_dev[:,4],X_dev[:,5]]
	


	"""
	input_train = [X_train]
	output_train = [Y_train[:,0],Y_train[:,1],Y_train[:,2],X_train[:,0],X_train[:,1],X_train[:,2],X_train[:,3],X_train[:,4],X_train[:,5]]
	input_dev = [X_dev]
	output_dev = [Y_dev[:,0],Y_dev[:,1],Y_dev[:,2],X_dev[:,0],X_dev[:,1],X_dev[:,2],X_dev[:,3],X_dev[:,4],X_dev[:,5]]
	"""

	"""
	input_train = [X_train]
	output_train = [X_train[:,0],X_train[:,1],X_train[:,2],X_train[:,3],X_train[:,4],X_train[:,5]]
	input_dev = [X_dev]
	output_dev = [X_dev[:,0],X_dev[:,1],X_dev[:,2],X_dev[:,3],X_dev[:,4],X_dev[:,5]]
	"""


	results = combined_model.fit(
	 input_train,output_train,
	 epochs = NUM_EPOCHS ,
	 batch_size = model_params["batchSize"],
	 validation_data = (input_dev,output_dev),
	 verbose =2,
	 #callbacks = [customCallbacks.CustomSaveModel(model_params = model_params,model = combined_model)],
	 callbacks = callback_list,
	 shuffle = True
	)
	
	




	print("*"*50)
	print("Finished Training - Saving Models")
	print("*"*50)


	with open(os.path.join('model_params',model_params["model_name"] +  '.json'), 'w') as fp:
		json.dump(model_params, fp, sort_keys=True)

	mse_total_dev = run_eval_base(model_location,dataset = "dev",email = model_params["email"])
	mse_total_train = run_eval_base(model_location,dataset = "train",email = model_params["email"])
	mse_total_test = run_eval_base(model_location,dataset = "test",email = model_params["email"])


	pdb.set_trace()

	return (
		mse_total_train[0:len(model_params["OutputNames"])],
		mse_total_dev[0:len(model_params["OutputNames"])],
		mse_total_test[0:len(model_params["OutputNames"])]
		)

	


def runBaseLineRegression(model_params,data,estimator):

	#regr = MultiOutputRegressor(sklearn.linear_model.LinearRegression())
	regr = MultiOutputRegressor(estimator)
	#regr = MultiOutputRegressor(sklearn.linear_model.BayesianRidge())
	#regr = MultiOutputRegressor(sklearn.linear_model.Lasso())

	#data
	AP_train,TRP_train = data[0]
	AP_dev,TRP_dev = data[1]

	if model_params["DirectionForward"]:
		X_train,Y_train,X_dev,Y_dev = TRP_train,AP_train,TRP_dev,AP_dev
	else:
		X_train,Y_train,X_dev,Y_dev = AP_train,TRP_train,AP_dev,TRP_dev
		model_params["OutputNames"],model_params["InputNames"] = model_params["InputNames"],model_params["OutputNames"]

	regr.fit(X_train,Y_train)
	Y_dev_pred = regr.predict(X_dev)
	Y_train_pred = regr.predict(X_train)

	if model_params["DirectionForward"]:
		#train
		mse_totoal_train = customUtils.mse_p(ix = (3,6),Y_pred = Y_train_pred,Y_true = Y_train)
		#dev
		mse_totoal_dev = customUtils.mse_p(ix = (3,6),Y_pred = Y_dev_pred,Y_true = Y_dev)

	else:
		mse_totoal_train = mse(Y_train,Y_train_pred,multioutput = 'raw_values')
		mse_totoal_dev = mse(Y_dev,Y_dev_pred,multioutput = 'raw_values')

	
	model_location = os.path.join('models',model_params["model_name"] +  '.json')


	with open(os.path.join('model_params',model_params["model_name"] +  '.json'), 'w') as fp:
		json.dump(model_params, fp, sort_keys=True)

	_ = run_eval_base(model_location,dataset = "train",email = model_params["email"])
	_ = run_eval_base(model_location,dataset = "test",email = model_params["email"])
	mse_total = run_eval_base(model_location,dataset = "dev",email = model_params["email"])

	
	return (mse_totoal_train.tolist(),mse_totoal_dev.tolist(),mse_totoal_train.sum(),mse_totoal_dev.sum())



	


def main():



	"""
		Mean A,A,A,phi,phi,phi = 0.79798654,0.84098498 0.90992756,3.01879273,3.00808806, 3.34166802
		Mean T,R,P = 0.49927981,  0.12808498,  0.51283834
	"""


	model_param_grid  = {
		'model_tag': ['Custom'],
		'HiddenLayers':[500], 
		'NumOfHiddenLayers': [3],
		'DropOut': [False],
		'Activiation':["relu"],
		'Final_Activation':["softplus"],
		'OutputNames':[['A_1','A_2','A_3','phi_1','phi_2','phi_3']],
		#'OutputNames':[['phi_1','phi_2','phi_3']],
		'InputNames':[['T','R','P']],
		'DirectionForward':[False],
		'RunType':["Custom"],
		#"SimpleNN","SimpleAutoencoder","DistNN","DistAutoencoder"  #Custom
		'num_epochs': [1000],
		'early_stopping': [False],
		'rlr': [False],
		'debug': [False],
		'batchSize':[2500],
		'notes': ["Forward"],
		'save_weight_only':[True],
		"TensorBoard_logging": [True],
		"CustomPrint" :[False],
		"email" :["false"],
		"use_weights" :[False],
		"weights_factor":[1],
		"email_summary":[False],
		"phase_loss":["mse"],
		"scaling":[False]
	}

	"""
	print("="*50)
	print("Starting Grid Training")
	print("Grid Training Params:")
	customUtils.print_dic_params(model_param_grid,True)
	print("="*50)
	"""


	if model_param_grid["debug"] == True:
		print("*"*50)
		print("DEBUUUUUUUUG MODE --- DATA NOT SAVED")
		print("*"*50)
		time.sleep(2)

	
	min_dev_array = []
	mse_array = []


	train_data,dev_data,test_data = prep.loadData()
	data = [train_data,dev_data]







	#MAIN LOOP OF GRID
	for run_num in range(len(ParameterGrid(model_param_grid))):
		print("*"*50)
		print("Run Num: ",run_num," of ", len(ParameterGrid(model_param_grid)))
		print("*"*50)
		model_params_this_run_total = ParameterGrid(model_param_grid)[run_num]
		
		model_name = model_params_this_run_total["model_tag"]
		#Add time signature
		now = str(datetime.datetime.now())
		now = now.replace(":","_")
		now = now.replace(" ","_")
		now = now.replace("-", "_")
		now = now.split(".")[0]
		model_name =  model_name + "_run_" + str(run_num) + "_time_" + now
		model_params_this_run_total["model_name"] = model_name


		if model_params_this_run_total["RunType"] == "SimpleAutoencoder":
			mse_total_train,mse_total_dev,mse_total_test = runSimpleEncoderDecoderModelWithParams(model_params_this_run_total,data)
		elif model_params_this_run_total["RunType"] == "SimpleNN":
			mse_total_train,mse_total_dev,mse_total_test = runSimpleModelWithParams(model_params_this_run_total,data)
		elif model_params_this_run_total["RunType"] == "DistAutoencoder":
			mse_total_train,mse_total_dev,mse_total_test = runDistEncoderDecoderModelWithParams(model_params_this_run_total,data)
		elif model_params_this_run_total["RunType"] == "DistNN":
			mse_total_train,mse_total_dev,mse_total_test = runDistModelWithParams(model_params_this_run_total,data)
		elif model_params_this_run_total["RunType"] == "Custom":
			mse_total_train,mse_total_dev,mse_total_test = runCustomAutpEncoderWithParams(model_params_this_run_total,data)
		elif model_params_this_run_total["RunType"] == "LR":
			_,result,_,_ = runBaseLineRegression(model_params_this_run_total,data,sklearn.linear_model.LinearRegression())
			mse_total_dev = np.array(result)

		else:
			raise NotImplementedError("Not Correct Run Type")

		message = ""
		message += "<br>"
		params = customUtils.print_dic_params(model_params_this_run_total,False,delimitor = "<br>",kvdelimitor = ":" )
		message += "<br>" + str(params)
		message += "<br>" "train mse: "+ str(mse_total_train)
		message += "<br>" "dev mse: "+ str(mse_total_dev)
		message += "<br>" "test mse: "+ str(mse_total_test)

		if model_params_this_run_total["email_summary"] == True:
			print("Sending Email")
			email_logger.send_email(text_mesage = message, title_text = str(model_params_this_run_total["model_name"]),image_path = None)
			print(message.replace("<br>","\n"))
		
		mse_array.append([mse_total_train,mse_total_dev,mse_total_test])
		min_dev_array.append(np.min(mse_total_dev.sum()))
		print("Sim: " + str(run_num+1)+ " Min MSE TOTAL Loss: " + str(min_dev_array[run_num]))
		
	
	#print results of Search
	print("----------------------------------------------------------------------------------")
	print("results are:" + str(min_dev_array))
	#print("sorting order:" + str(np.argsort(min_dev_array)))
	#print("sorted results: " + str(min_dev_array[np.argsort(min_dev_array)]))
	print("min is: "+ str(np.min(min_dev_array)))
	print("for num: "+ str(np.argmin(min_dev_array)))
	print("params are:")
	
	
	#for k, v in ParameterGrid(model_param_grid)[np.argmin(min_dev_array)].items():
	#	print(k, v)
	




	message = ""
	for i in range(len(mse_array)):
		message += "<br>" + "run num :" + str(i) + " train mse: " + str(mse_array[i][0]) + " dev mse: " + str(mse_array[i][1]) + "test mse: " + str(mse_array[i][2])


	#message += "<br>" + "sorting order:" + str(np.argsort(min_dev_array))
	#message += "<br>" + "sorted results: " + str(min_dev_array[np.argsort(min_dev_array)])

	if model_param_grid["email_summary"][0] == True:
		email_logger.send_email(text_mesage = message, title_text = "Results for: " + str(model_param_grid["model_tag"]),image_path = None)
	print(message.replace("<br>","\n"))



if __name__ == "__main__":
	main()

