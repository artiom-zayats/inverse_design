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
from customMetrics import atan_mse,modulo_2pi_error,log_likelihood_normal_cost,nll,mean_squared_error_mu,acc_dist,cv_test,custom_loss,phase_mse

import prep
import run_eval


#extra
import argparse
import argcomplete

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import email_logger

EPSILON = np.finfo(np.float32).eps



def run_ensemble_eval(models_locations = None):


	mse_total_list = []
	smaple_total_list = []
	rmse_total_list = []

	for dataset in ["train","dev","test"]:

		Y_pred_list = []
		#Load Data
		Y_true_dev = 0

		for ix,model_location in enumerate(models_locations):

			best_model = run_eval.loadBestModel(model_location)


			path, model_name_with_ext = os.path.split(model_location)
			model_name, file_extension = os.path.splitext(model_name_with_ext)

			model_params_loc = os.path.join('model_params',model_name+".json")



			with open(model_params_loc) as jf:  
				model_params = json.load(jf)

			X_train,Y_train,X_dev,Y_dev = run_eval.loadData(model_params,dataset = dataset)
			Y_true_dev = np.transpose(Y_dev)

			if model_params["scaling"]:
				X_train,X_dev,scaler = prep.scaleData(X_train,X_dev)

			Y_pred_dev = best_model.predict(X_dev)
			


			output = np.zeros((Y_pred_dev[0].shape[0],len(model_params["OutputNames"])))
			for i in range(len(model_params["OutputNames"])):
				temp = Y_pred_dev[i]
				output[:,i] = temp[:,0]

			Y_pred_dev = np.transpose(output)

			#print mse_models

			mse = ((Y_pred_dev - Y_true_dev) ** 2).mean(axis=-1)
			mape = 100*(np.abs(Y_pred_dev - Y_true_dev)/(Y_true_dev + EPSILON)).mean(axis = -1)
			smape = 2*100*(np.abs(Y_pred_dev - Y_true_dev)/np.abs(Y_true_dev + Y_pred_dev + EPSILON)).mean(axis = -1)
			rmse = (np.sqrt((Y_pred_dev - Y_true_dev) ** 2)).mean(axis=-1)


			non_valid_p_percent = 100* np.where(Y_pred_dev[1,:] > Y_pred_dev[2,:])[0].shape[0]/Y_pred_dev[1,:].shape[0]

			#Print Errors

			print("Dataset: ",dataset)
			print("Model: ",ix,":", model_location)
			print("non Valid P: ",non_valid_p_percent )
			print("MSE: ",mse)
			print("SMAPE: ",smape)
			#print("RMSE: ", rmse)




			Y_pred_list.append(Y_pred_dev)
			

		#pdb.set_trace()

		#?,3

		if (len(models_locations)>1):

			Y_pred_enseble = sum(Y_pred_list)/len(Y_pred_list)

			mse = ((Y_pred_enseble - Y_true_dev) ** 2).mean(axis=-1)
			smape = 2*100*(np.abs(Y_pred_enseble - Y_true_dev)/np.abs(Y_true_dev + Y_pred_enseble + EPSILON)).mean(axis = -1)
			rmse = (np.sqrt((Y_pred_enseble - Y_true_dev) ** 2)).mean(axis=-1)
			mse_total_list.append(mse)
			smaple_total_list.append(smape)
			rmse_total_list.append(rmse)
			print("Dataset: ",dataset)
			print("Model: Ensamble")
			print("non Valid P: ",non_valid_p_percent )
			print("MSE: ",mse)
			print("SMAPE: ",smape)
			#print("RMSE: ", rmse)


	#print("*"*50)
	#print("For Excell")
	#print(mse_total_list[0],mse_total_list[1],mse_total_list[2])





def run_ensemble_eval_reverse_model(models_locations = None):



	#LOAD REvERSE MODEL
	reverse_model_loc = "models/Fowrard_BaseLine_run_0_time_2018_11_05_19_05_19.hdf5"
	best_reverse_model = run_eval.loadBestModel(reverse_model_loc)
	path, model_name_with_ext = os.path.split(reverse_model_loc)
	model_name, file_extension = os.path.splitext(model_name_with_ext)
	model_params_loc = os.path.join('model_params',model_name+".json")
	with open(model_params_loc) as jf:  
		reverse_model_params = json.load(jf)






	#for dataset in ["train","dev","test"]:
	for dataset in ["dev"]:

		Y_pred_list = []
		#Load Data
		Y_true_dev = 0

		for ix,model_location in enumerate(models_locations):

			best_model = run_eval.loadBestModel(model_location)


			path, model_name_with_ext = os.path.split(model_location)
			model_name, file_extension = os.path.splitext(model_name_with_ext)

			model_params_loc = os.path.join('model_params',model_name+".json")



			with open(model_params_loc) as jf:  
				model_params = json.load(jf)

			X_train,Y_train,X_dev,Y_dev = run_eval.loadData(model_params,dataset = dataset)

			if model_params["scaling"]:
				X_train,X_dev,scaler = prep.scaleData(X_train,X_dev)

			Y_pred_dev = best_model.predict(X_dev)
			


			output = np.zeros((Y_pred_dev[0].shape[0],len(model_params["OutputNames"])))
			for i in range(len(model_params["OutputNames"])):
				temp = Y_pred_dev[i]
				output[:,i] = temp[:,0]

			Y_pred_dev = output

			## Add reverse Stuff
			if reverse_model_params["scaling"]:
				_,Y_dev,scaler_reverse = prep.scaleData(Y_train,Y_dev)
				_,Y_pred_dev,scaler_reverse = prep.scaleData(Y_train,Y_pred_dev)

			X_from_true = best_reverse_model.predict(Y_dev)
			X_from_pred = best_reverse_model.predict(Y_pred_dev)

			#pdb.set_trace()

			output1 = np.zeros((X_from_true[0].shape[0],len(reverse_model_params["OutputNames"])))
			output2 = np.zeros((X_from_true[0].shape[0],len(reverse_model_params["OutputNames"])))
			for i in range(len(reverse_model_params["OutputNames"])):
				temp1 = X_from_true[i]
				temp2 = X_from_pred[i]
				output1[:,i] = temp1[:,0]
				output2[:,i] = temp2[:,0]

			X_from_true = output1
			X_from_pred = output2


			#pdb.set_trace()


			##NEED TO ADD CONVERT TO NUMPY

			angle = np.arctan2(np.sin(X_from_true[:,3:6] - X_from_pred[:,3:6]), np.cos(X_from_true[:,3:6] - X_from_pred[:,3:6]))
			mae_angs = np.abs(angle).mean(axis = 0)

			mse_amp = ((X_from_pred[:,0:3] - X_from_true[:,0:3]) ** 2).mean(axis=0)

			errors = np.sqrt(mse_amp).tolist()+mae_angs.tolist()

			print("Dataset: ",dataset)
			print("Model: ",ix,":", model_location)
			print("Errors: ",errors)



		if (len(models_locations)>1):

				print("Dataset: ",dataset)
				print("Model: Ensamble")


def main():

	#Defauls
	model_loc_list = []
	model_loc_list.append("models/Reverse_Scaling_run_0_time_2018_11_04_20_20_32.hdf5")
	#model_loc_list.append("models/Reverse_Scaling_run_1_time_2018_11_04_20_22_43.hdf5")
	#model_loc_list.append("models/Reverse_Scaling_run_2_time_2018_11_04_20_27_37.hdf5")
	#model_loc_list.append("models/Reverse_Scaling_run_3_time_2018_11_04_20_30_38.hdf5")


	#run_ensemble_eval(models_locations = model_loc_list)
	run_ensemble_eval_reverse_model(models_locations = model_loc_list)



if __name__ == "__main__":
	main()