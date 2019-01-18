import numpy as np
import math
import pdb


length = 100
factor = 0.5
error_range = np.linspace(math.pi*factor,math.pi*factor,num = length)
true = np.array([0]*length)
pred = true + error_range


atan_error = np.arctan2(np.sin(true-pred),np.cos(true-pred))  
atan_error = np.sqrt(np.mean(np.square(atan_error)))


mse_error = np.sqrt(np.mean((np.square(true-pred))))

print("mse_error",mse_error,"atan_error",atan_error)


#pdb.set_trace()
"""
angle = tf.atan2(tf.sin(y_true - y_pred), tf.cos(y_true - y_pred))
result = K.mean(tf.abs(angle))
"""