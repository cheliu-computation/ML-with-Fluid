import os
import sys
sys.path.append("/home/david/thesis/lib/python3.8/site-packages")
import vtktools
import numpy as np
import matplotlib.pyplot as plt
import datetime
import math
import keras.backend as K
import process_data as Pcd
import mkdirs_org
from keras.layers import Input, Dense, LSTM, Dropout
from keras import regularizers
from keras.models import Model, Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
from keras.callbacks import LearningRateScheduler

def variable_value():
	path =  "D:/Master-Thesis/water_collapse"
	verify_rate = 0.3
	sequence_length = 3
	originalFile =  "D:/Master-Thesis/water_collapse/water_data" # original file
	destinationFile = "D:/Master-Thesis/water_collapse/data_enkf_full" 

	return path, verify_rate, sequence_length, originalFile, destinationFile

def predict_sequences_multiple(model, data, sequence_length, predict_num):
	
	origianl_input = data[0:sequence_length-1, :] # 
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


		np.save('checkInput', model_input)

		temp_code = code
		final_output_code = temp_code if i == 0 else np.vstack((final_output_code, temp_code))

	return final_output_code


def predict_vol(dataset, predict_num, sequence_length):

	# print(np.max(dataset),np.min(dataset), np.mean(dataset), np.median(dataset))
	
	encoder = load_model('vel_encoder(AE_10).h5', compile = False) # encoder verified data
	code = encoder.predict(dataset) # dimension with verified data after scaled
	# np.save('original_code.npy', code)
	# print(np.max(code),np.min(code),np.mean(code), np.median(code), code.shape)

	LSTM = load_model('Material_lstm(ENKF_full).h5', compile=False)
	outputs = predict_sequences_multiple(LSTM, code, sequence_length, predict_num)
	# print(np.max(outputs),np.min(outputs),np.mean(outputs), np.median(outputs), outputs.shape)
	

	decoder = load_model('vel_decoder(AE_10).h5', compile = False) # decoder predicted data
	
	predicted_vol = decoder.predict(outputs)
	# print(np.max(predicted_vol),np.min(predicted_vol),np.mean(predicted_vol), np.median(predicted_vol))
	np.save('D:/Master-Thesis/water_collapse/code/Material_ENKF_full.npy', predicted_vol)
	return predicted_vol

if __name__=="__main__":  

	path, verify_rate, sequence_length, originalFile, destinationFile = variable_value()
	#-------------------------------------#

	print("Data Preprocessing...")  
	vol = Pcd.get_data(path)
	dataset, verify = Pcd.train_and_vertify(vol,verify_rate) # predict and verify


	# scaler_data = MinMaxScaler()
	# scaler_vol = scaler_data.fit_transform(verify)

	# scaler_vol = verify

	print("Data Predicting...")  

	print('dataset shape = ' + str(dataset.shape))
	print('vertify shape = ' + str(verify.shape))


	predicted_vol = predict_vol(verify, verify.shape[0]- sequence_length, sequence_length)
	predicted_vol = np.array(predicted_vol)
	
	# print('shape = '+ str(np.shape(predicted_vol)))
	#------------------------------------------#

	# scaler_outpus = predicted_vol
	

	# value = mean_squared_error(verify, scaler_outpus)
	# print(value)

	# predicted_vol = np.load('D:/Master-Thesis/water_collapse/code/Material_ENKF.npy')
	# mkdirs_org.mkdir(destinationFile)

	#------------------------------------------#
		#------------------------------------------#
	mkdirs_org.copyFiles(originalFile,destinationFile)
	vtu_num = Pcd.get_vtu_num(originalFile)
	num = vtu_num - sequence_length-210
	mkdirs_org.transform(predicted_vol, num, destinationFile, sequence_length)
