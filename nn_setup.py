import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from keras.models import Sequential
from matplotlib import pyplot
from sklearn import preprocessing

import sys
import email_logger

import pdb

#TESTLINE


#params
BATCH_SIZE = 200
OUTPUT_DIM = 4
INPUT_DIM = 6
HIDDEN_LAYERS = 500
DROP_OUT = False
NUM_OF_LAYERS = 6
RANDOM_SEED = 1003
NUM_EPOCHS = 100
ACTIVATION = "relu"



np.random.seed(RANDOM_SEED)
#TODO add vars
#opening pickled dataframe
df = pd.read_pickle(os.path.join(cwd,"data","train.pkl"))

#splitting data into TRAIN and DEV
#90/10
msk = np.random.rand(len(df)) < 0.90
df_train = df[msk]
df_dev = df[~msk]



#encoding data
def encode(df,shuffle):

	if (shuffle):
		df.sample(frac = 1,random_state = RANDOM_SEED)

	x = np.array([df['A_1'],df['A_2'],df['A_3'],df['phi_1'],df['phi_2'],df['phi_3']])
	y = np.array([df['T'],df['R'],df['P_1'],df['P_2']])

	x = np.transpose(x)
	y = np.transpose(y)

	return (x.astype(float), y.astype(float))

#generate datapoints
def generate_batch_data(batch_size,x,y):
	xs = []
	ys = []

	for i in range (batch_size):
		xs.append(x)
		yx.append(y)

	return (np.array(xs),np.array(xy))




#model
model = Sequential()

#Input
model.add(layers.Dense(HIDDEN_LAYERS,activation = None, input_dim = INPUT_DIM))

#Hidden - Layers
for layer in range(NUM_OF_LAYERS):
	if DROP_OUT:
		model.add(layers.Dropout(0.5, seed = RANDOM_SEED))
	model.add(layers.Dense(HIDDEN_LAYERS, activation = ACTIVATION))
#model.add(layers.Dropout(0.5, seed = RANDOM_SEED))




# Output- Layer
model.add(layers.Dense(OUTPUT_DIM, activation = None))
model.summary()



model.compile(
 optimizer = "adam",
 loss = "mse",
 metrics = ["mse",'mae']
)


#Training

x_train,y_train = encode(df_train, True) 
x_dev,y_dev = encode(df_dev, False)


#print("Before Normlaization")
#print("Mean Values of Input",[np.mean(x_train[:,i]) for i in range(5)])
#print("STD of Input",[np.std(x_train[:,i]) for i in range(5)])


#scaler = preprocessing.MinMaxScaler().fit(x_train)
#x_train = scaler.fit_transform(x_train)
#scaler = preprocessing.MinMaxScaler().fit(x_dev)
#x_dev = scaler.fit_transform(x_dev)

#print("After Normlaization")
#print("Mean Values of Input",[np.mean(x_train_s[:,i]) for i in range(5)])
#print("STD of Input",[np.std(x_train_s[:,i]) for i in range(5)])




results = model.fit(
 x_train,y_train,
 epochs = NUM_EPOCHS ,
 batch_size = BATCH_SIZE,
 validation_data = (x_dev, y_dev),
 verbose = 2
)


y_dev_pred = model.predict(x_dev)

#pdb.set_trace()

print("T Error:", np.mean((y_dev[:,0] - y_dev_pred[:,0])**2))
print("R Error:", np.mean((y_dev[:,1] - y_dev_pred[:,1])**2))
print("P_1 Error:", np.mean((y_dev[:,2] - y_dev_pred[:,2])**2))
print("P_2 Error:", np.mean((y_dev[:,3] - y_dev_pred[:,3])**2))




# summarize history for loss

pyplot.plot(results.history['loss'])
pyplot.plot(results.history['val_loss'])


min_label_text = "min val: "+"{:.4f}".format(np.min(results.history['val_loss']))



pyplot.annotate(min_label_text,
             xy=(np.argmin(results.history['val_loss']), np.min(results.history['val_loss'])), xycoords='data',
             xytext=(0.5, 0.5), textcoords='axes fraction', fontsize=6,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

pyplot.title('model loss(MSE)')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'dev'], loc='upper left')
picname = "Hidden Layers " + str(NUM_OF_LAYERS) + "Hidden N " + str(HIDDEN_LAYERS) + "Epoch " + str(NUM_EPOCHS) +"DropOut " + str(DROP_OUT) + '.png'
pyplot.savefig(os.path.join(cwd,"pictures",picname))

#pyplot.plot(results.history['loss'])
#pyplot.plot(results.history['val_loss'])
#pyplot.title('model loss(MSE)')
#pyplot.ylabel('loss')
#pyplot.xlabel('epoch')
#pyplot.legend(['train', 'dev'], loc='upper left')
