import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import process_data as PCD
import vtktools
import os
import sys
from sklearn.preprocessing import MinMaxScaler
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Input, Flatten, Dense, Lambda, Reshape
from keras.layers import BatchNormalization
from keras.models import Model, Model, Sequential, load_model
from keras.datasets import mnist
from keras.losses import binary_crossentropy
from keras import backend as K
from keras.callbacks import LearningRateScheduler
import lstm_vel_batch_ENKF as LSTM_ENKF

# define parameter
def parameter():
	path = 'D:/Master Thesis/Fluidity-ubuntu/circledata'
	epoch = 1
	verify_rate = 0.4
	code_dim = 32

	return path, epoch, verify_rate, code_dim	

# reduced order based on pca
def PCA_ANY(X_train, X_test):
	

	# set pca parameter
	pca = PCA(n_components = code_dim)
	# scale data
	scaler_train = MinMaxScaler()
	X_scaled = scaler_train.fit_transform(X_train)

	# train pca model
	X_reduced = pca.fit_transform(X_scaled)
	# save pca model
	with open ('pca.pkl', 'wb') as pickle_file:
		pickle.dump(pca, pickle_file)

	# reload pca model
	with open('pca.pkl', 'rb') as pickle_file:
		pca_load = pickle.load(pickle_file)

	## reduced order of test data
	scaler_test = MinMaxScaler()
	X_test_scaled = scaler_test.fit_transform(X_test)
	X_test_reduced = pca_load.transform(X_test_scaled)


	X_test_original = np.dot(X_test_reduced, pca_load.components_) + pca_load.mean_
	# rescale data
	X_test_scaled_output = scaler_test.inverse_transform(X_test_original)

	print('shape of output = ' + str(X_test_scaled_output))
	return X_test_scaled_output

def scheduler(epoch, lr): # run with each epoch
	if epoch < 5:
		return lr * (2 - tf.math.exp(-0.1))
	else:
		return lr *  tf.math.exp(-0.2)

#
if __name__=="__main__":
	path, epoch, verify_rate, code_dim = parameter()
	
	#get data
	X_train, X_test = PCD.train_and_vertify(PCD.get_data(path), verify_rate)
	print('Dataset have already created')


	#
	class VAE(keras.Model):
		def __init__(self, encoder, decoder, **kwargs):
			super(VAE, self).__init__(**kwargs)
			self.encoder = encoder
			self.decoder = decoder

		def train_step(self, data):
			if isinstance(data, tuple):
				data = data[0]
			with tf.GradientTape() as tape:
				z_mean, z_log_var, z = encoder(data)
				reconstruction = decoder(z)
				reconstruction_loss = tf.reduce_mean(
		    		keras.losses.binary_crossentropy(data, reconstruction)
				)
				reconstruction_loss *= 28 * 28
				kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
				kl_loss = tf.reduce_mean(kl_loss)
				kl_loss *= -0.5
				total_loss = reconstruction_loss +  kl_loss
			grads = tape.gradient(total_loss, self.trainable_weights)
			self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
			return {
					"loss": total_loss,
					"reconstruction_loss": reconstruction_loss,
					"kl_loss": kl_loss,
					}
	class Sampling(layers.Layer):  # create the code with n*latent dim, subjected to gaussian distibution(posulated code)
    # """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
		def call(self, inputs):
			z_mean, z_log_var = inputs
			batch = tf.shape(z_mean)[0]
			dim = tf.shape(z_mean)[1]
			epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
			return z_mean + tf.exp(0.5 * z_log_var) * epsilon

	latent_dim = code_dim
	# encoding layer
	encoder_inputs = keras.Input(shape=(X_train.shape[1])) # input layer
	# encode = layers.Dense(latent_dim*32, activation="relu", kernel_regularizer='l2')(encoder_inputs)
	# encode = layers.Dense(latent_dim*16, activation="relu", kernel_regularizer='l2')(encoder_inputs)
	encode = layers.Dense(latent_dim*8, activation="relu", kernel_regularizer='l2')(encoder_inputs)
	encode = layers.Dense(latent_dim*4, activation="relu", kernel_regularizer='l2')(encode)
	encode = layers.Dense(latent_dim*2, activation="relu", kernel_regularizer='l2')(encode)
	encode = layers.Dense(latent_dim, activation="relu", kernel_regularizer='l2')(encode)

	z_mean = layers.Dense(latent_dim, name="z_mean")(encode) 
	z_log_var = layers.Dense(latent_dim, name="z_log_var")(encode)
	z = Sampling()([z_mean, z_log_var])	# construct latent space
	
	encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
	
	# decoder layer
	latent_inputs = keras.Input(shape=(latent_dim))
	decode = layers.Dense(latent_dim, activation="relu")(latent_inputs)
	decode = layers.Dense(latent_dim*2, activation="relu", kernel_regularizer='l2')(decode)
	decode = layers.Dense(latent_dim*4, activation="relu", kernel_regularizer='l2')(decode)
	decode = layers.Dense(latent_dim*8, activation="relu", kernel_regularizer='l2')(decode)	
	
	decoder_outputs = layers.Dense(X_train.shape[1], activation="tanh")(decode) 
	
	decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
	
	# combine encoder and decoder
	vae = VAE(encoder, decoder)
	vae.compile(optimizer=keras.optimizers.Adam())
	#define dynamic leraning rate
	callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
	# training model
	history = vae.fit(X_train, X_train, epochs = epoch, batch_size=32, callbacks=[callback])
	



	


	print(X_test.shape)
	code = encoder.predict(X_test)
	print(np.array(code).shape)

	a1 = np.array(code)[0,:, 0]
	a2 = np.array(code)[1,:, 0]
	a3 = np.array(code)[2,:, 0]
	print(a3.shape)
	plt.figure(1)
	plt.plot(a1)
	
	plt.figure(2)
	plt.plot(a2)

	plt.figure(3)
	plt.plot(a3)

	plt.show()
	# reconstruction_data = decoder.predict(code[2])

	# plt.figure(1)
	# plt.plot(reconstruction_data[20:100, 1000], label= 'VAE data X')
	# plt.plot(X_test[20:100, 1000], label = 'Original data X')
	# plt.grid()
	# plt.legend()

	# plt.figure(2)
	# plt.plot(reconstruction_data[20:100, 12568+1000], label= 'VAE data Y')
	# plt.plot(X_test[20:100, 12568+1000], label = 'Original data Y')
	# plt.grid()
	# plt.legend()

	# plt.figure(3)
	# plt.plot(history.history['reconstruction_loss'])
	# plt.title('reconstruction_loss')
	# plt.grid()

	# plt.figure(4)
	# plt.plot(history.history['kl_loss'])
	# plt.title('kl_loss')
	# plt.grid()

	# plt.show()


