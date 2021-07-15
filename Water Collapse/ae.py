import os
import sys
# sys.path.append("/home/david/thesis/lib/python3.8/site-packages")
import vtktools
import numpy as np
import matplotlib.pyplot as plt
import datetime
import math
import keras.backend as K
import process_data as Pcd
# import mkdirs
from keras.layers import Input, Dense, LSTM, Dropout
from keras import regularizers
from keras.models import Model, Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
from keras.callbacks import LearningRateScheduler


def variable_value():

	path =  "D:/Master-Thesis/water_collapse/water_data"
	verify_rate = 0.3
	my_epochs = 100
	encoding_dim = 10
	# originalFile = "D:/Master Thesis/Fluidity-ubuntu/lock_exchange/Vertify Data"# original file
	# destinationFile = "D:/Master Thesis/Fluidity-ubuntu/lock_exchange/predict" # destination file

	return path, verify_rate, my_epochs, encoding_dim


def ae_vol(vol, my_epochs, encoding_dim):# dim = 3

	input_img = Input(shape=(vol.shape[1], ))
	# encoded = Dense(encoding_dim * 32, activation='relu')(input_img)
	# encoded = Dense(encoding_dim * 8, activation='relu')(encoded)
	# encoded = Dense(encoding_dim * 4, activation='relu')(input_img)
	encoded = Dense(encoding_dim * 2, activation='relu')(input_img)
	encoded = Dense(encoding_dim)(encoded)
	
	# "decoded" is the lossy reconstruction of the input
	decoded = Dense(encoding_dim * 2, activation='relu',)(encoded)
	# decoded = Dense(encoding_dim * 4, activation='relu')(decoded)
	# decoded = Dense(encoding_dim * 16, activation='relu')(decoded)
	# decoded = Dense(encoding_dim * 32, activation='relu')(decoded)
	decoded = Dense(vol.shape[1], activation='tanh')(decoded)

	# this model maps an input to its reconstruction
	autoencoder = Model(input_img, decoded)
	encoder = Model(input_img, encoded)
	encoded_input = Input(shape=(encoding_dim, ))
 
	decoder_layer1 = autoencoder.layers[-1]
	decoder_layer2 = autoencoder.layers[-2]
	# decoder_layer3 = autoencoder.layers[-3]
	# decoder_layer4 = autoencoder.layers[-4]
	# decoder_layer5 = autoencoder.layers[-5]

	# create the decoder model
	decoder = Model(encoded_input, decoder_layer1(decoder_layer2((encoded_input))))

	# configure model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer
	autoencoder.compile(optimizer='adam', loss = 'mean_absolute_error', metrics = ['accuracy'])

	# train the model
	x_train = vol

	
	history = autoencoder.fit(x_train, x_train, epochs=my_epochs, batch_size=32, validation_split=0.1)
	# temp = np.hstack((history.history['loss'],history.history['val_loss']))
	# np.save('deep_AE_12_loss.npy', temp)
	# Pcd.draw_acc_loss(history)

	encoder.save('vel_encoder(AE_10).h5') 
	autoencoder.save('vel_ae(AE_10).h5')
	decoder.save('vel_decoder(AE_10).h5')

	print("ae-model train succeed")  

def train_vel(vol):
	my_epochs, encoding_dim = variable_value()[2],variable_value()[3]

	if not os.path.exists('vel_encoder(AE_10).h5'):
		ae_vol(vol, my_epochs, encoding_dim)


if __name__=="__main__":  

	path, verify_rate, my_epochs, encoding_dim = variable_value()

	#load data
	print("Data loading...")
	vol = Pcd.get_data(path)
	print('data shape = ' +str(vol.shape))
	dataset, verify = Pcd.train_and_vertify(vol,verify_rate) 
	print("training_dataset shape:",dataset.shape, "   vertify_dataset shape:", verify.shape)

	#process data

	# scaler = MinMaxScaler() # normalization
	# scalered_temp = scaler.fit_transform(dataset) # normalization

	#train model
	print("Model Building and training... ")
	train_vel(dataset)







	