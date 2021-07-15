import numpy as np
from keras.layers import Input, Dense, LSTM, Dropout
from keras import regularizers
from keras.callbacks import LearningRateScheduler
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential, load_model
from numpy.linalg import inv


## for 1D input data
def EnKF(input_dim, percentage_H_matrix, batch_size, LSTM_epochs, train_input, train_validation, test_input, test_validation, model_name):
#EnKF part start
	ensemble_sample = input_dim # smaller than input_dim

	H = np.zeros((input_dim,input_dim)) # create matrix for converting predicted data to comparew with observed data
	np.fill_diagonal(H, 1*percentage_H_matrix) # set H as diagonal matrx

	
	batch_size = batch_size 
	batch_num = train_input.shape[0]
	element_num = int(batch_num / batch_size) + 1
	indice = np.linspace(0, batch_num, element_num)
	indice = np.round(indice)
	indice = indice.astype(int)
	
	# train model
	# ENKF part
	for i in range(len(indice) - 2):
		history = model_name.fit(train_input[indice[i] : indice[i+1], :, :], train_validation[indice[i] : indice[i+1], :], 
			epochs=LSTM_epochs, batch_size=batch_size, validation_data = (test_input, test_validation),verbose = 2)
		
		predicted_data = np.array(model_name.predict(train_input[indice[i+1]: indice[i+1]+1 , :, :])) # 1 row - multiple column data

		
		#assimilation part
		if i < len(train_input) - 1:
			
			observed_data = train_input[indice[i+1]+1, -1, :ensemble_sample]
			
			predicted_data = predicted_data.T
			observed_data = observed_data.T
			
			observed_data = np.reshape(observed_data, (observed_data.shape[0], 1))

			R = (observed_data.dot(np.transpose(observed_data)))/(ensemble_sample - 1)
			
			
			A = predicted_data[:ensemble_sample, :] - (predicted_data[:ensemble_sample, :]/ensemble_sample)
			C = (A.dot(np.transpose(A)))/ (ensemble_sample - 1)

			# print(predicted_data.shape, C.shape, H.shape, R.shape, observed_data.shape)
			# print((H))
			EnKF_gain = np.zeros((predicted_data.shape[0], predicted_data.shape[1]))
			#inverse has singular value??
			Temp_gain = C.dot(H.T).dot(inv(H.dot(C).dot(H.T) + R)).dot(observed_data- H.dot(predicted_data[:ensemble_sample, :]))

			EnKF_gain[:Temp_gain.shape[0],:Temp_gain.shape[1]] = Temp_gain

			assimlated_data = predicted_data + EnKF_gain
			assimlated_data = assimlated_data.T
			train_input[indice[i+1]+1: indice[i+1]+2, -1, :] = assimlated_data # update

	return model_name
	#ENKF part finish

def ENKF_raw(input_dim, percentage_H_matrix, batch_size, LSTM_epochs, train_input, train_validation, test_input, test_validation, model_name, raw_data):
#EnKF part start
	# train_input dimension = raw_data dimension
	# workflow: train-->output-->convert from code to raw dimension
	# --> ENKF on output with raw dimension and raw data
	# --> encode ENKF data to code dimension --> replace the next train data

	# import encoder and decoder
	encoder = load_model('vel_encoder(AE_10).h5', compile = False)
	decoder = load_model('vel_decoder(AE_10).h5', compile = False)

	## set assimilated part dimension
	ensemble_sample = input_dim # smaller than input_dim

	H = np.zeros((input_dim,input_dim)) # create matrix for converting predicted data to comparew with observed data
	np.fill_diagonal(H, 1*percentage_H_matrix) # set H as diagonal matrx

	
	batch_size = batch_size 
	batch_num = train_input.shape[0]
	element_num = int(batch_num / batch_size) + 1
	indice = np.linspace(0, batch_num, element_num)
	indice = np.round(indice)
	indice = indice.astype(int)
	
	# train model
	# ENKF part
	for i in range(len(indice) - 2):
		history = model_name.fit(train_input[indice[i] : indice[i+1], :, :], train_validation[indice[i] : indice[i+1], :], 
			epochs=LSTM_epochs, batch_size=batch_size, validation_data = (test_input, test_validation),verbose = 2)
		
		predicted_data = np.array(model_name.predict(train_input[indice[i+1]: indice[i+1]+1 , :, :])) # 1 row - multiple column data
		# output is code
		
		# convert code to fluidity data
		data_raw_dimension = decoder.predict(predicted_data)
		#assimilation part
		if i < len(raw_data) - 1:
			
			observed_data = raw_data[indice[i+1]+1, -1, :ensemble_sample]
			
			data_raw_dimension = data_raw_dimension.T
			observed_data = observed_data.T
			
			observed_data = np.reshape(observed_data, (observed_data.shape[0], 1))

			R = (observed_data.dot(np.transpose(observed_data)))/(ensemble_sample - 1)
			
			
			A = data_raw_dimension[:ensemble_sample, :] - (data_raw_dimension[:ensemble_sample, :]/ensemble_sample)
			C = (A.dot(np.transpose(A)))/ (ensemble_sample - 1)

			# print(predicted_data.shape, C.shape, H.shape, R.shape, observed_data.shape)
			# print((H))
			EnKF_gain = np.zeros((data_raw_dimension.shape[0], data_raw_dimension.shape[1]))
			
			Temp_gain = C.dot(H.T).dot(inv(H.dot(C).dot(H.T) + R)).dot(observed_data- H.dot(data_raw_dimension[:ensemble_sample, :]))

			EnKF_gain[:Temp_gain.shape[0],:Temp_gain.shape[1]] = Temp_gain

			assimlated_data = data_raw_dimension + EnKF_gain
			assimlated_data = assimlated_data.T
			
			assimlated_data = encoder.predict(assimlated_data)

			train_input[indice[i+1]+1: indice[i+1]+2, -1, :] = assimlated_data # update

		loss= history.history['loss'] if i ==0 else np.vstack((loss, history.history['loss'])) 
		val_loss= history.history['val_loss'] if i ==0 else np.vstack((loss, history.history['val_loss'])) 
	np.save('D:/Master-Thesis/water_collapse/code/model_loss', loss)	
	np.save('D:/Master-Thesis/water_collapse/code/model__val_loss', val_loss)
	return model_name
	#ENKF part finish