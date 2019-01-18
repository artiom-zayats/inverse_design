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
from customMetrics import atan_mse,modulo_2pi_error,log_likelihood_normal_cost,nll,mean_squared_error_mu,acc_dist,cv_test,custom_loss,phase_mse,smape,custom_loss_TRP,custom_metroc_TRP

import prep


#extra
import argparse
import argcomplete

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import email_logger


##Helpers Function

def plotFunct2(model_params,pred,true,input,dataset = "dev"):


	names1 = model_params["OutputNames"]
	names2 = model_params["InputNames"]

	data_name = ["pred","true"]

	#pdb.set_trace()

	long_range = np.arange(len(true[:,0])*len(model_params["OutputNames"]))

	true_list = []
	sort_ix = np.argsort(true[:,0])

	for i in range(len(model_params["OutputNames"])):
		val1 = true[:,0]
		val2 = true[:,1]
		val3 = true[:,2]
		temp_long = np.concatenate((val1[sort_ix],val2[sort_ix],val3[sort_ix]),axis = 0)
		true_list.append(temp_long)

	pred_list = []
	for i in range(len(model_params["OutputNames"])):
		val1 = pred[:,0]
		val2 = pred[:,1]
		val3 = pred[:,2]
		temp_long = np.concatenate((val1[sort_ix],val2[sort_ix],val3[sort_ix]),axis = 0)
		pred_list.append(temp_long)


	for i in range(1):
		plt.plot(long_range,true_list[i],'bo',markersize = 1);
		plt.plot(long_range,pred_list[i],'ro', markersize = 1);





				
	image_path = os.path.join("images",model_params["model_name"] + "input" + dataset +  '.png')
	plt.savefig(image_path)

	plt.gcf().clear()
			#sendtext_mesage, title_text
	email_logger.send_email(text_mesage = dataset+" vs input", title_text = model_params["model_name"],image_path = image_path)

	#pdb.set_trace()



def plotFunct1(model_params,pred,true,dataset = "dev"):


	names = model_params["OutputNames"]

	data_name = ["pred","true"]
	data_list = [pred,true]


	for ix,data in enumerate(data_list):
		for i,name1 in enumerate(names):
			for j,name2 in enumerate(names):

				plt.subplot(len(names), len(names), i*len(names)+ j+1)
				plt.scatter(data[:,j], data[:,i], marker=".")
				plt.xlabel(name2)
				plt.ylabel(name1)

		plt.suptitle(dataset) # or plt.suptitle('Main title')
				
		image_path = os.path.join("images",model_params["model_name"] + data_name[ix] + dataset +  '.png')
		plt.savefig(image_path)

		plt.gcf().clear()
			#sendtext_mesage, title_text
		email_logger.send_email(text_mesage = dataset + data_name[ix], title_text = model_params["model_name"],image_path = image_path)





def plotAndSendPic(name,true_val,mu,model_params,variance = None, percent = 1,isdots = True, dataset = "dev"):

	#pdb.set_trace()
	#choose only 30%
	np.random.seed(1003)
	msk = np.random.rand(len(true_val)) < 0.03 #change to chnage percentage

	true_val = true_val[msk]
	mu = mu[msk]


	try:
		variance = variance[msk]
		lower = mu - 3*np.sqrt(variance)
		upper = mu + 3*np.sqrt(variance)
	except:
		print("No Variance")

	if "ph" in name:
		print("test")
		old_mu = mu
		mu = mu%(math.pi*2)
		print((mu - old_mu).sum())
		#pdb.set_trace()

	ape = 100.*np.abs(mu - true_val)/(true_val+ np.finfo(float).eps)
	ape = np.clip(ape,0,100)
	error = np.abs(mu - true_val)

	message = "For data: "+ dataset +  " Name :" + str(name) + " better than 50%: " + str(100.*ape[ape<50].shape[0]/ape.shape[0])
	print(message)

	#pdb.set_trace()
	sort_ix = np.argsort(true_val)
	x = np.arange(len(sort_ix))
	#plot

	plt.plot(x,true_val[sort_ix] , 'bo', markersize = 1)
	plt.errorbar(x, mu[sort_ix], yerr=[lower[sort_ix], upper[sort_ix]], color='green',markersize=2,ecolor='r', capthick=2)

	"""
	if isdots:
		plt.plot(x,mu[sort_ix], 'ro', markersize = 1)
		plt.plot(x,true_val[sort_ix] , 'bo', markersize = 1)
	else:
		plt.plot(x,mu[sort_ix], color = 'red', markersize = 1)
		plt.plot(x,true_val[sort_ix] , color = 'blue', markersize = 1)
	"""



	#plt.scatter(mu,true_val)

	#plt.legend([ r'$\mu ', r'$\mu+3$\variance', r'$\mu+3$\variance', r'true'], loc='upper left')
	plt.ylabel('Value of ' + str(name))
	plt.xlabel('sample N')
	plt.legend([ 'predicted ', 'true'], loc='upper left')
	plt.title(model_params["model_name"] + "_plot_for: "+ str(name) +" data: " + dataset)
	image_path = os.path.join("images",model_params["model_name"] +str(name) + dataset + '.png')
	plt.savefig(image_path)

	plt.gcf().clear()
	#sendtext_mesage, title_text
	email_logger.send_email(text_mesage = message, title_text =model_params["model_name"],image_path = image_path)

def loadData(model_params,dataset):
	train_data,dev_data,test_data = prep.loadData()

	if dataset == "test":
		data = [train_data,test_data]
	elif dataset == "train":
		data = [train_data,train_data]
	elif dataset == "dev":
		data = [train_data,dev_data]
	else:
		raise NameError("Not correct dataset")
	# change to use Test:

	#data
	AP_train,TRP_train = data[0]
	AP_dev,TRP_dev = data[1]

	#Check direction
	if (model_params["DirectionForward"]):
		X_train,Y_train,X_dev,Y_dev = TRP_train,AP_train,TRP_dev,AP_dev
	else:
		X_train,Y_train,X_dev,Y_dev = AP_train,TRP_train,AP_dev,TRP_dev


	return (X_train,Y_train,X_dev,Y_dev)

def loadBestModel(model_location):

	best_model =  load_model(model_location,
	custom_objects={
	'nll': nll,
	'atan_mse':atan_mse,
	'modulo_2pi_error':modulo_2pi_error,
	'log_likelihood_normal_cost':log_likelihood_normal_cost,
	'mean_squared_error_mu':mean_squared_error_mu,
	'acc_dist':acc_dist,
	'cv_test':cv_test,
	'custom_loss':atan_mse,
	'phase_mse':phase_mse,
	'smape':smape,
	})
	return best_model

def loadBestModel_Custom(model_location):

	best_model =  load_model(model_location,
	custom_objects={
	'nll': nll,
	'atan_mse':atan_mse,
	'modulo_2pi_error':modulo_2pi_error,
	'log_likelihood_normal_cost':log_likelihood_normal_cost,
	'mean_squared_error_mu':mean_squared_error_mu,
	'acc_dist':acc_dist,
	'cv_test':cv_test,
	'custom_loss':custom_loss,
	'phase_mse':phase_mse,
	'smape':smape,
	'custom_loss_TRP':custom_loss_TRP,
	'custom_metroc_TRP':custom_metroc_TRP,
	})
	return best_model


#Eval Runners


def runSimpleNNModelEval(model_location,model_params,dataset,email):

	X_train,Y_train,X_dev,Y_dev = loadData(model_params,dataset)



	if model_params["scaling"]:
		X_train,X_dev,scaler = prep.scaleData(X_train,X_dev)
	



	best_model =  loadBestModel(model_location)

	Y_pred_dev = best_model.predict(X_dev)
	Y_pred_dev = np.array(Y_pred_dev)
	Y_pred_dev = np.squeeze(Y_pred_dev)
	Y_pred_dev = np.transpose(Y_pred_dev)
	Y_true_dev = np.array([row for row in Y_dev.T])
	Y_true_dev = np.transpose(Y_true_dev)

	#pdb.set_trace()

	if (model_params["DirectionForward"]):
		mse_total = customUtils.mse_p(ix = (3,6),Y_pred = Y_pred_dev,Y_true = Y_true_dev)
	else:
		diff = Y_pred_dev - Y_true_dev
		mse_total = np.sqrt(((diff) ** 2)).mean(axis=0)

	
	#plotFunct2(model_params,Y_pred_dev,Y_true_dev,X_dev,dataset)
	#plotFunct1(model_params,Y_pred_dev,Y_true_dev,dataset)
	

	if email == 'true':
		for i,name in enumerate(model_params["OutputNames"]):
			plotAndSendPic(name = name,true_val = Y_true_dev[:,i],mu = Y_pred_dev[:,i],model_params = model_params,percent = 1,isdots = True,dataset = dataset)
			#plotAndSendPic(name = name,true_val = Y_true_dev[:,i],mu = Y_pred_dev[:,i],model_params = model_params,percent = 0.1,isdots = False,dataset = dataset)


	print("*"*50)
	print(model_params["OutputNames"]+model_params["InputNames"])
	print("MSE TOTAL DEV: " ,mse_total)
	print("*"*50)
	print("Model Was:",
	str(",".join(model_params["InputNames"])),
	"->",str(",".join(model_params["OutputNames"])),
	"->",str(",".join(model_params["InputNames"])))


	return mse_total



def runCustomEval(model_location,model_params,dataset,email):

	X_train,Y_train,X_dev,Y_dev = loadData(model_params,dataset)




	weights_location = os.path.join('weights',model_params["model_name"] +  '.h5')




	if model_params["scaling"]:
		X_train,X_dev,scaler = prep.scaleData(X_train,X_dev)
	

	#pdb.set_trace()

	combined_model =  loadBestModel_Custom(model_location)
	combined_model.load_weights(weights_location)

	model_forward = loadBestModel_Custom(model_params["ForwardModelLocation"])
	model_forward.load_weights(weights_location,by_name = True)

	model_reverse = loadBestModel(model_params["InverseModelLocation"]) #change to load weights


	Y_pred = model_forward.predict(X_dev)
	#Y_pred = (np.asarray(Y_pred)[:,:,0]).T
	X_from_pred = model_reverse.predict(Y_pred)
	angs_pred = (np.asarray(X_from_pred)[3:6,:,0]).T

	X_from_true = model_reverse.predict(Y_dev)
	angs_true = (np.asarray(X_from_true)[3:6,:,0]).T


	angle_error = np.arctan2(np.sin(angs_true - angs_pred), np.cos(angs_true - angs_pred))
	rad_error = np.abs(angle_error).mean(axis = 0)


	#pdb.set_trace()

	mse_amp = ((np.asarray(X_from_pred)[0:3,:,0].T - np.asarray(X_from_true)[0:3,:,0].T) ** 2).mean(axis=0)

	Y_pred_dev = Y_pred
	Y_true_dev = Y_dev

	mse_total = ((Y_pred_dev - Y_true_dev) ** 2).mean(axis=0)

	smape = 2*100*(np.abs(Y_pred_dev - Y_true_dev)/np.abs(Y_true_dev + Y_pred_dev + np.finfo(np.float32).eps)).mean(axis = 0)
	non_valid = 100* np.where(Y_pred_dev[:,1] >= Y_pred_dev[:,2])[0].shape[0]/Y_pred_dev[:,1].shape[0]
	valid  = 100* np.where(Y_pred_dev[:,1] < Y_pred_dev[:,2])[0].shape[0]/Y_pred_dev[:,1].shape[0]

	print(dataset)
	print("mse_amp" ,mse_amp )
	print("rad_error",rad_error)
	print("mse_TRP",mse_total)
	print("smape",smape)
	print("non valid",non_valid,"valid",valid)
	


	#pdb.set_trace()

	if email == 'true':
		for i,name in enumerate(model_params["OutputNames"]):
			plotAndSendPic(name = name,true_val = Y_true_dev[:,i],mu = Y_pred_dev[:,i],model_params = model_params,percent = 1,isdots = True,dataset = dataset)
			#plotAndSendPic(name = name,true_val = Y_true_dev[:,i],mu = Y_pred_dev[:,i],model_params = model_params,percent = 0.1,isdots = False,dataset = dataset)




	print("*"*50)
	#print(model_params["OutputNames"]+model_params["InputNames"])
	#print("MSE TOTAL DEV: " ,mse_total)
	#print("*"*50)
	#print("Model Was:",
	#str(",".join(model_params["InputNames"])),
	#"->",str(",".join(model_params["OutputNames"])),
	#"->",str(",".join(model_params["InputNames"])))


	return mse_total	

def runSimpleEncoderDecoderEval(model_location,model_params,dataset,email):

	X_train,Y_train,X_dev,Y_dev = loadData(model_params,dataset)
	best_model =  loadBestModel(model_location)

	if model_params["scaling"]:
		X_train,X_dev,scaler = prep.scaleData(X_train,X_dev)


	Y_pred_dev = best_model.predict(X_dev)
	Y_pred_dev = np.array(Y_pred_dev)
	Y_pred_dev = np.squeeze(Y_pred_dev)
	Y_pred_dev = np.transpose(Y_pred_dev)
	Y_true_dev = np.array([row for row in Y_dev.T]+[row for row in X_dev.T])
	Y_true_dev = np.transpose(Y_true_dev)




	if (model_params["DirectionForward"]):
		mse_total = customUtils.mse_p(ix = (3,6),Y_pred = Y_pred_dev,Y_true = Y_true_dev)
	else:
		mse_total = customUtils.mse_p(ix = (6,9),Y_pred = Y_pred_dev,Y_true = Y_true_dev)

	#plotFunct1(model_params,Y_pred_dev,Y_true_dev,dataset)

	if email == 'true':
		for i,name in enumerate(model_params["OutputNames"]):
			plotAndSendPic(name = name,true_val = Y_true_dev[:,i],mu = Y_pred_dev[:,i],model_params = model_params,percent = 1,isdots = True,dataset = dataset)
			#plotAndSendPic(name = name,true_val = Y_true_dev[:,i],mu = Y_pred_dev[:,i],model_params = model_params,percent = 0.1,isdots = False,dataset = dataset)
		

	print("*"*50)
	print(model_params["OutputNames"]+model_params["InputNames"])
	print("MSE TOTAL DEV: " ,mse_total)
	print("*"*50)
	print("Model Was:",
	str(",".join(model_params["InputNames"])),
	"->",str(",".join(model_params["OutputNames"])),
	"->",str(",".join(model_params["InputNames"])))

	return mse_total



def runDistModelEval(model_location,model_params,dataset,email):

	X_train,Y_train,X_dev,Y_dev = loadData(model_params,dataset)
	best_model =  loadBestModel(model_location)



	if model_params["scaling"]:
		X_train,X_dev,scaler = prep.scaleData(X_train,X_dev)



	Y_true_dev = np.array([row for row in Y_dev.T])
	Y_true_dev = np.transpose(Y_true_dev)
	Y_pred_dev = best_model.predict(X_dev)

	"""
	pdb.set_trace()

	Y_pred_dev = np.array(Y_pred_dev)
	Y_pred_dev = np.squeeze(Y_pred_dev)
	Y_pred_dev = np.transpose(Y_pred_dev)


	pdb.set_trace()
	#ReverseScaling
	Y_pred_dev = scaler_Y.inverse_transform(Y_pred_dev)
	Y_true_dev = scaler_Y.inverse_transform(Y_true_dev)

	"""

	
	number_to_pick = 1
	output_pred = np.zeros(( Y_pred_dev[0].shape[0],number_to_pick*len(model_params["OutputNames"])))

	for i,name in enumerate(model_params["OutputNames"]):
		temp = Y_pred_dev[i][:]
		mu = temp[:,0]
		variance = temp[:,1]
		CV = np.sqrt(variance)/mu
		print("CV for:",name,"=",CV.mean()*100)
		upper = mu + 3*np.sqrt(variance)
		lower = mu - 3*np.sqrt(variance)
		true_val = Y_true_dev[:,i]
		correct_num = np.sum(np.logical_and(true_val < upper, true_val > lower))
		acc = 100*correct_num/len(true_val)
		print("Acc for:", name , "=", acc, "%")
		dist_size = 1
		distributions = [np.random.normal(m, s, dist_size) for m, s in zip(mu, np.sqrt(variance))]
		distributions_array  = np.stack(distributions,axis = 1).T

		if email == 'true':
			plotAndSendPic(name,true_val,mu,model_params,variance,percent = 1,isdots = True,dataset = dataset)
			#plotAndSendPic(name,true_val,mu,model_params,variance,percent = 0.1,isdots = False,dataset = dataset)

		output_pred[:,i] = mu


		
		

	Y_pred_dev = output_pred


	"""	
	#ReverseScaling
	Y_pred_dev = scaler_Y.inverse_transform(Y_pred_dev)
	Y_true_dev = scaler_Y.inverse_transform(Y_true_dev)
	"""

	#pdb.set_trace()
	#plotFunct1(model_params,Y_pred_dev[:,0:len(model_params["OutputNames"])],Y_true_dev[:,0:len(model_params["OutputNames"])],dataset)


	if (model_params["DirectionForward"]):
		mse_total = customUtils.mse_p(ix = (3,6),Y_pred = Y_pred_dev,Y_true = Y_true_dev)
	else:
		diff = Y_pred_dev - Y_true_dev
		mse_total = ((diff) ** 2).mean(axis=0)
		
	print("*"*50)
	print(model_params["OutputNames"]+model_params["InputNames"])
	print("MSE TOTAL DEV: " ,mse_total)
	print("*"*50)
	print("Model Was:",
	str(",".join(model_params["InputNames"])),
	"->",str(",".join(model_params["OutputNames"])),
	"->",str(",".join(model_params["InputNames"])))

	return mse_total



def runDistEncoderDecoderEval(model_location,model_params,dataset,email):

	X_train,Y_train,X_dev,Y_dev = loadData(model_params,dataset)
	best_model =  loadBestModel(model_location)

	if model_params["scaling"]:
		X_train,X_dev,scaler = prep.scaleData(X_train,X_dev)

	Y_true_dev = np.array([row for row in Y_dev.T]+[row for row in X_dev.T])
	Y_true_dev = np.transpose(Y_true_dev)


	Y_pred_dev = best_model.predict(X_dev)

	number_to_pick = 1
	output_pred = np.zeros(( Y_pred_dev[0].shape[0],number_to_pick*len(model_params["OutputNames"])))

	for i,name in enumerate(model_params["OutputNames"]):
		temp = Y_pred_dev[i][:]
		mu = temp[:,0]
		variance = temp[:,1]
		CV = np.sqrt(variance)/mu
		print("CV for:",name,"=",CV.mean()*100)
		upper = mu + 3*np.sqrt(variance)
		lower = mu - 3*np.sqrt(variance)
		true_val = Y_true_dev[:,i]
		correct_num = np.sum(np.logical_and(true_val < upper, true_val > lower))
		acc = 100*correct_num/len(true_val)
		print("Acc for:", name , "=", acc, "%")

		#tempDist = np.random.normal(mu, variance, 1000)
		dist_size = 1
		
		distributions = [np.random.normal(m, s, dist_size) for m, s in zip(mu, np.sqrt(variance))]
		distributions_array  = np.stack(distributions,axis = 1).T
		#output_pred[:,i:i+number_to_pick] = distributions_array

		if email == 'true':
			plotAndSendPic(name,true_val,mu,model_params,variance,percent = 1,isdots = True,dataset = dataset)
			#plotAndSendPic(name,true_val,mu,model_params,variance,percent = 0.1,isdots = False,dataset = dataset)

		output_pred[:,i] = mu

	input_pred = np.zeros(( Y_pred_dev[0].shape[0],len(model_params["InputNames"])))


	for i,name in enumerate(model_params["InputNames"]):
		input_pred[:,i] = np.squeeze(Y_pred_dev[len(model_params["OutputNames"])+i][:])


	Y_pred_dev = np.concatenate((output_pred,input_pred),axis = 1)

	#plotFunct1(model_params,Y_pred_dev[:,0:len(model_params["OutputNames"])],Y_true_dev[:,0:len(model_params["OutputNames"])],dataset)


	if (model_params["DirectionForward"]):
		mse_total = customUtils.mse_p(ix = (3,6),Y_pred = Y_pred_dev,Y_true = Y_true_dev)
	else:
		mse_total = customUtils.mse_p(ix = (6,9),Y_pred = Y_pred_dev,Y_true = Y_true_dev)
	print("*"*50)
	print(model_params["OutputNames"]+model_params["InputNames"])
	print("MSE TOTAL DEV: " ,mse_total)
	print("*"*50)
	print("Model Was:",
	str(",".join(model_params["InputNames"])),
	"->",str(",".join(model_params["OutputNames"])),
	"->",str(",".join(model_params["InputNames"])))

	return mse_total


def runLREval(model_location,model_params,dataset,email):

	X_train,Y_train,X_dev,Y_dev = loadData(model_params,dataset)


	regr = MultiOutputRegressor(sklearn.linear_model.LinearRegression())
	regr.fit(X_train,Y_train)
	Y_pred_dev = regr.predict(X_dev)
	Y_true_dev = Y_dev

	#plotFunct1(model_params,Y_pred_dev,Y_true_dev,dataset)

	if email == 'true':
		for i,name in enumerate(model_params["OutputNames"]):
			plotAndSendPic(name = name,true_val = Y_true_dev[:,i],mu = Y_pred_dev[:,i],model_params = model_params,percent = 1,isdots = True,dataset = dataset)
			#plotAndSendPic(name = name,true_val = Y_true_dev[:,i],mu = Y_pred_dev[:,i],model_params = model_params,percent = 0.1,isdots = False,dataset = dataset)

	if (model_params["DirectionForward"]):
		mse_total = customUtils.mse_p(ix = (3,6),Y_pred = Y_pred_dev,Y_true = Y_true_dev)
	else:
		mse_total = customUtils.mse_p(ix = (6,9),Y_pred = Y_pred_dev,Y_true = Y_true_dev)
	print("*"*50)
	print(model_params["OutputNames"]+model_params["InputNames"])
	print("MSE TOTAL DEV: " ,mse_total)
	print("*"*50)
	print("Model Was:",
	str(",".join(model_params["InputNames"])),
	"->",str(",".join(model_params["OutputNames"])),
	"->",str(",".join(model_params["InputNames"])))

	return mse_total



def run_eval_base(model_location,dataset,email = 'false'):

	path, model_name_with_ext = os.path.split(model_location)
	model_name, file_extension = os.path.splitext(model_name_with_ext)

	model_params_loc = os.path.join('model_params',model_name+".json")

	with open(model_params_loc) as jf:  
		model_params = json.load(jf)

	if model_params["RunType"] == "SimpleAutoencoder":
		mse_total = runSimpleEncoderDecoderEval(model_location,model_params,dataset,email)
	elif model_params["RunType"] == "SimpleNN":
		mse_total = runSimpleNNModelEval(model_location,model_params,dataset,email)
	elif model_params["RunType"] == "DistAutoencoder":
		mse_total =  runDistEncoderDecoderEval(model_location,model_params,dataset,email)
	elif model_params["RunType"] == "DistNN":
		mse_total = runDistModelEval(model_location,model_params,dataset,email)
	elif model_params["RunType"] == "LR":
		mse_total = runLREval(model_location,model_params,dataset,email)
	elif model_params["RunType"] == "Custom":
		mse_total = runCustomEval(model_location,model_params,dataset,email)	
	else:
		raise NotImplementedError("Not Correct Run Type")

	if email == 'true':
		email_logger.send_email(text_mesage ="Results for:" +  dataset + str(model_params["OutputNames"]) + str(mse_total), title_text = model_params["model_name"])

	return mse_total




def main(args):

	if args.dataset not in ["test","dev","train"]:
		raise NameError("Not correct datasetName")

	run_eval_base(args.model_file,args.dataset,args.email)




	



if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(description = 'LOads Model and DataFile')
	parser.add_argument('-mf','--model_file',type = str, help='model_file_location')
	parser.add_argument('-d', '--dataset', type = str, help = 'choose from train/dev/test')
	parser.add_argument('-e', '--email', type = str, help = 'send email with results',default = 'false')
	argcomplete.autocomplete(parser)
	args = parser.parse_args()

	main(args)



