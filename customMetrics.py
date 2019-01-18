import numpy as np
np.random.seed(1003)
import pdb
import math
import tensorflow as tf
import keras.backend as K
import keras.losses as Losses




def modulo_2pi_error(y_true, y_pred):

	two_pi_tensor = tf.ones_like(y_pred)
	two_pi_tensor = tf.multiply(two_pi_tensor, math.pi*2)
	y_pereodic_pred = tf.mod(y_pred,two_pi_tensor)

	#return Losses.mean_squared_error(y_true,y_pereodic_pred)
	return K.mean(tf.abs(y_pereodic_pred - y_pred))


def atan_mse(y_true, y_pred):

	angle = tf.atan2(tf.sin(y_true - y_pred), tf.cos(y_true - y_pred))

	result = K.mean(tf.abs(angle))

	return result


def log_likelihood_normal_cost(y_true, y_pred):
	miu = y_pred[:,0]
	variance = y_pred[:,1]
	true = y_true[:,0]
	cost = K.log(2*math.pi*variance)/2 + K.square(true - miu)/(2 * variance)
	return K.sum(cost, axis=-1)


def phase_mse(y_true,y_pred):

	#pdb.set_trace()
	pi_tensor = tf.ones_like(y_pred)
	pi_tensor = tf.multiply(pi_tensor, math.pi)

	abs_err = K.maximum(y_true,y_pred) - K.minimum(y_true,y_pred)
	

	mask = K.tf.cast(K.greater(abs_err, pi_tensor),tf.float32)

	abs_err = mask*(K.tf.add(abs_err,pi_tensor)) + (1-mask)*(abs_err)
	sq_abs_err = K.abs(abs_err)
	#pdb.set_trace()
	result  = K.mean(sq_abs_err, axis = -1)

	return result



	"""
	abs_err = max(\phi_t,\phi_p)-min(\phi_t,\phi_p)
	if abs_err > \pi the abs_err = abs_err-\pi
	sq_err = abs_err^2
	"""



"""
def vae_loss(x, x_decoded_mean):
    xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    return xent_loss + kl_loss
"""


def nll(y_true, y_pred):

	miu = y_pred[:,0]
	variance = y_pred[:,1]
	true = y_true[:,0]


	likelihood = K.tf.distributions.Normal(miu, K.sqrt(variance))

	result = -1*K.sum(likelihood.log_prob(true), axis=-1)



	return result



def mean_squared_error_mu(y_true, y_pred):
	true = y_true[:,0]
	pred = y_pred[:,0]

	return Losses.mean_squared_error(true,pred)

def acc_dist(y_true, y_pred):
	miu = y_pred[:,0]
	sigma = y_pred[:,1]
	true = y_true[:,0]

	upper = miu + 3*sigma
	lower = miu - 3*sigma

	correct_ix = K.tf.logical_and(true < upper, true > lower)
	dummy = K.tf.ones_like(true)
	
	hit = 100*tf.reduce_sum(tf.cast(correct_ix, tf.float32))/tf.reduce_sum(dummy)

	return hit


def cv_test(y_true, y_pred):
	miu = y_pred[:,0]
	sigma = y_pred[:,1]

	cv = 100.*K.mean(sigma/miu)
	return cv


def custom_loss(y_true, y_pred):
	

	return y_true


def custom_loss_TRP(y_true, y_pred):

	#K.transpose(

	T_true = y_true[:,0]
	R_true = y_true[:,1]
	P_true = y_true[:,2]

	T_pred = y_pred[:,0]
	R_pred = y_pred[:,1]
	P_pred = y_pred[:,2]

	mask = K.tf.cast(K.greater_equal(R_pred, P_pred),tf.float32)

	non_valid = 100*tf.reduce_sum(mask)/tf.reduce_sum(K.tf.ones_like(P_pred))

	smapeT = smape(T_true,T_pred)
	smapeR = smape(R_true,R_pred)
	smapeP = smape(P_true,P_pred)

	mseT = Losses.mean_squared_error(T_true, T_pred)
	mseP = Losses.mean_squared_error(P_true, P_pred)
	mseR = Losses.mean_squared_error(R_true, R_pred)

	mse = Losses.mean_squared_error(y_true,y_pred)

	result = non_valid

	return non_valid


def custom_metroc_TRP(y_true, y_pred):

	#K.transpose(

	T_true = y_true[:,0]
	R_true = y_true[:,1]
	P_true = y_true[:,2]

	T_pred = y_pred[:,0]
	R_pred = y_pred[:,1]
	P_pred = y_pred[:,2]

	mask = K.tf.cast(K.greater(R_pred, P_pred),tf.float32)

	non_valid = 100*tf.reduce_sum(mask)/tf.reduce_sum(K.tf.ones_like(P_pred))

	smapeT = smape(T_true,T_pred)
	smapeR = smape(R_true,R_pred)
	smapeP = smape(P_true,P_pred)

	mseT = Losses.mean_squared_error(T_true, T_pred)
	mseP = Losses.mean_squared_error(P_true, P_pred)
	mseR = Losses.mean_squared_error(R_true, R_pred)

	mse = Losses.mean_squared_error(y_true,y_pred)

	result = non_valid + mseT+ mseP*10 + mseR

	return result


def smape(y_true, y_pred):
	smape = 2*K.mean((K.abs(y_pred - y_true)/(K.abs(y_true) + K.abs(y_pred) + K.epsilon())),axis = -1)
	return smape




