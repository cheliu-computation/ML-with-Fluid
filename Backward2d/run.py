import os
import sys
sys.path.append("/home/david/thesis/lib/python3.8/site-packages")
import vtktools
import numpy as np
import matplotlib.pyplot as plt
import datetime
import math
import keras.backend as K
import mkdirs_org
import process_data as Pcd
# import mkdirs_org
from keras.layers import Input, Dense, LSTM, Dropout
from keras import regularizers
from keras.models import Model, Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
from keras.callbacks import LearningRateScheduler

def variable_value():
	path =  "D:/Master-Thesis/backward_facing_step_2d"
	verify_rate = 0.3
	sequence_length = 20
	originalFile =  "D:/Master-Thesis/backward_facing_step_2d/fluidity_data(target)" # original file
	destinationFile = "D:/Master-Thesis/backward_facing_step_2d/data_no_enkf" 

	return path, verify_rate, sequence_length, originalFile, destinationFile

def predict_sequences_multiple(model, data, sequence_length, predict_num):
	
	origianl_input = data[0:sequence_length-1, :] # set initial inpit as 1-19 code
	initial_input = origianl_input.reshape(1,origianl_input.shape[0],origianl_input.shape[1])
	print('[Model] Predicting Sequences Multiple...')
	for i in range(predict_num):
		if i == 0:
			model_input = initial_input
			code = model.predict(model_input)
			# code = code.reshape(1,1,code.shape[1])
		else:
			code = code.reshape(1,1,code.shape[1])
			model_input = np.concatenate((model_input,code), axis = 1)
			code = model.predict(model_input[:, i:, :])

		temp_code = code
		final_output_code = temp_code if i == 0 else np.vstack((final_output_code, temp_code))

	return final_output_code


def predict_vol(dataset, predict_num, sequence_length):

	print(np.max(dataset),np.min(dataset), np.mean(dataset), np.median(dataset))

	encoder = load_model('vel_encoder(AE_10).h5', compile = False) # encoder verified data
	code = encoder.predict(dataset, batch_size=512) # dimension with verified data after scaled
	

	LSTM = load_model('magni_lstm.h5(new)', compile=False)
	outputs = predict_sequences_multiple(LSTM, code, sequence_length, predict_num)
	

	decoder = load_model('vel_decoder(AE_10).h5', compile = False) # decoder predicted data
	predicted_vol = decoder.predict(outputs, batch_size=512)
	np.save('D:/Master-Thesis/backward_facing_step_2d/code/magnitude_new.npy', predicted_vol)
	return predicted_vol, 

if __name__=="__main__":  

	path, verify_rate, sequence_length, originalFile, destinationFile = variable_value()

	print("Data Preprocessing...")  
	vol = Pcd.get_data(path)
	dataset, verify = Pcd.train_and_vertify(vol,verify_rate) # predict and verify


	print("Data Predicting...")  

	print('dataset shape = ' + str(dataset.shape))
	print('vertify shape = ' + str(verify.shape))

	
	scaler = MinMaxScaler().fit(verify)
	verify = scaler.transform(verify)

	predicted_vol = predict_vol(verify, verify.shape[0]- sequence_length, sequence_length)
	predicted_vol = np.array(predicted_vol)
	

	print('shape = '+ str(np.shape(predicted_vol)))
	predicted_vol = predicted_vol.reshape(predicted_vol.shape[1], predicted_vol.shape[2])
	predicted_vol = scaler.inverse_transform(predicted_vol)
	print('shape = '+ str(np.shape(predicted_vol)))
	

	# mkdirs_org.mkdir(destinationFile)
	# mkdirs_org.copyFiles(originalFile,destinationFile)
	# vtu_num = Pcd.get_vtu_num(originalFile)
	# num = predicted_vol.shape[0] - sequence_length
	# mkdirs_org.transform(predicted_vol, num, destinationFile, sequence_length)
