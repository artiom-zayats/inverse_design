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

## Mine IMPORTS:
import customModels
import customCallbacks
import customUtils
from customMetrics import atan_mse,modulo_2pi_error,log_likelihood_normal_cost,nll,mean_squared_error_mu,acc_dist,cv_test,custom_loss,phase_mse
import prep
from run_eval import run_eval_base


import email_logger




class CustomSaveModel(callbacks.Callback):

	def __init__(self,  model_params = None, model = None):
		super(callbacks.Callback,self).__init__()
		self.model_params = model_params
		self.model = model
		self.val_loss = []
		self.min_val_loss = None
		self.model_location = os.path.join('weights',model_params["model_name"] + '.h5')




	def on_epoch_end(self, epoch, logs={}):

		self.val_loss = logs['val_loss']

		if self.min_val_loss == None:
			self.min_val_loss = logs['val_loss']
		if self.min_val_loss > logs['val_loss']:
			self.min_val_loss = logs['val_loss']

			saver = tf.train.Saver()
			sess = keras.backend.get_session()
			saver.save(sess, './keras_model')
			model = self.model
			model.save_weights(self.model_location)
			print("model_saved")






class CustomPrintCallback(callbacks.Callback):

	def __init__(self,  model_params = None, metric = ["loss"]):
		super(callbacks.Callback,self).__init__()
		self.model_params = model_params
		self.metric = metric



	def on_epoch_end(self, epoch, logs={}):

		print("Epoch: " + str(epoch) + " Loss: "+ "{:.4f}".format(logs['loss'])+ " Dev_Loss: "+
		 "{:.4f}\n".format(logs['val_loss']))



		train_logs = []
		val_logs = []

		if self.model_params["RunType"] == "SimpleAutoencoder":
			name_list = self.model_params["OutputNames"] + self.model_params["InputNames"]
			self.metric =["mean_squared_error"]
		elif self.model_params["RunType"] == "SimpleNN":
			name_list = self.model_params["OutputNames"]
			self.metric =["mean_squared_error"]
		elif self.model_params["RunType"] in ["DistAutoencoder","DistNN"]:
			name_list = self.model_params["OutputNames"]
			self.metric = ["mean_squared_error_mu"]

		else:
			raise NotImplemented("Not Walid RunType")
		
		#loss_name = "log_likelihood_normal_cost"
		#T_loss = self.model_params["weight_for_loss"]["T"]*logs["T_"+loss_name]
		#R_loss = self.model_params["weight_for_loss"]["R"]*logs["R_"+loss_name]
		#P_loss = self.model_params["weight_for_loss"]["P"]*logs["P_"+loss_name]

		#trp_loss = np.array([T_loss,R_loss,P_loss])
		#print("mean:",trp_loss.mean(),"max:",trp_loss.max(),"min:",trp_loss.min())
		#print(name_list,"adjusted_losses:",trp_loss)
		#print("CV%:",100*trp_loss.std()/trp_loss.mean())





		for metric in self.metric:
			for name in name_list:
				if self.model_params["DirectionForward"]:
					if "ph" in name:
						val_logs.append(logs["val_"+name+"_"+"atan_mse"])
						train_logs.append(logs[name+"_"+"atan_mse"])
					else:
						val_logs.append(logs["val_"+name+"_"+metric])
						train_logs.append(logs[name+"_"+metric])
				else:
					val_logs.append(logs["val_"+name+"_"+metric])
					train_logs.append(logs[name+"_"+metric])

		def printStr(log_list):
			print_str = "\n"
			for i in range(len(log_list)):
				print_str += "{:.3f}|".format(log_list[i])
			return print_str

		
		#print("this Model is bettery by: \n")
		print("Metrics: ",self.metric)
		print("Train Metrics:",', '.join(name_list),":",printStr(train_logs))
		print("Dev Metrics:",', '.join(name_list),":",printStr(val_logs))
		print("Ratio Metrics:",', '.join(name_list),":",printStr([x/y for x, y in zip(train_logs, val_logs)]))
		



class LoggingTensorBoard(callbacks.TensorBoard):    

	def __init__(self, log_dir, model_params, **kwargs):
		super(LoggingTensorBoard, self).__init__(log_dir, **kwargs)

		self.log_dir = log_dir
		self.val_losses = {}
		self.model_params = model_params
		self.mse_p = {}


	def set_model(self, model):
		self.writer = tf.summary.FileWriter(self.log_dir)
		super(LoggingTensorBoard, self).set_model(model)


	def on_train_begin(self, logs=None):
		callbacks.TensorBoard.on_train_begin(self, logs=logs)
		




	def on_train_end(self,logs = None):
		
		min_val_losses = {}

		for k in self.val_losses.keys():
			vals = np.array(self.val_losses[k])
			min_val_losses[k]= [vals.min(),vals.argmin()]


		#find min of val_log

		MinValues = [tf.convert_to_tensor([k, str(min_val_losses[k])]) for k in sorted(min_val_losses.keys())]
		summary = tf.summary.text('Run Results Min Values', tf.stack(MinValues))
		
		with  tf.Session() as sess:
			s = sess.run(summary)
			self.writer.add_summary(s)


		clean_params = customUtils.clean_params(self.model_params)

		hyperparameters = [tf.convert_to_tensor([k, str(self.model_params[k])]) for k in sorted(clean_params.keys())]
		summary = tf.summary.text('Model_Params', tf.stack(hyperparameters))
		
		with  tf.Session() as sess:
			s = sess.run(summary)
			self.writer.add_summary(s)

		callbacks.TensorBoard.on_train_end(self,logs)
		


	def on_epoch_end(self, epoch,logs = None):
		callbacks.TensorBoard.on_epoch_end(self,epoch,logs = logs)


		for k in logs.keys():

			if k in self.val_losses.keys():
				self.val_losses[k].append(logs[k])
			else:
				self.val_losses[k] = [logs[k]]





def addCallBacks(model_params):


	callback_list = []
	if model_params["debug"] == False:
		if model_params["save_weight_only"]:
		#callback_list.append(callbacks.CSVLogger(filename=os.path.join('logging',model_params["model_name"]+'.csv')))
			callback_list.append(callbacks.ModelCheckpoint(os.path.join('weights',model_params["model_name"] + '.h5'),save_best_only=True , save_weights_only=True))
		else:
			callback_list.append(callbacks.ModelCheckpoint(os.path.join('models',model_params["model_name"] + '.hdf5'),save_best_only=True , save_weights_only=False))
	if model_params["TensorBoard_logging"] == True:
		testLogger = customCallbacks.LoggingTensorBoard(
				log_dir=os.path.join('tensorboard_logs',model_params["model_name"]),model_params = model_params,
				write_graph=False)
		callback_list.append(testLogger)
	if model_params["CustomPrint"] == True:
		my_callback = customCallbacks.CustomPrintCallback(model_params,metric = "loss")
		callback_list.append(my_callback)

	if (model_params["early_stopping"]):
		callback_list.append(callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=10, verbose=1, mode='auto'))

	if (model_params["rlr"]):
		callback_list.append(callbacks.ReduceLROnPlateau(
			monitor='loss',patience=10, verbose=1,factor=0.5,min_lr=0.000000001
			))
	
	return callback_list