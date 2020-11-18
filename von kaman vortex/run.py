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
	path =  "/media/sf_Fluidity-ubuntu/circledata"
	verify_rate = 0.4
	sequence_length = 20
	originalFile =  "/media/sf_Fluidity-ubuntu/Vertify Data" # original file
	destinationFile = "/media/sf_Fluidity-ubuntu/predict(ENKF)" 

	return path, verify_rate, sequence_length, originalFile, destinationFile

def predict_sequences_multiple(model, data, sequence_length, predict_num):
	
	print(data.shape)
	# print('feed data = ' + str(data.shape))
	print('[Model] Predicting Sequences Multiple...')
	for i in range(predict_num):
		feed_data = data[i: i+ sequence_length-1, :]
		print(i)
		
		feed_data =  feed_data.reshape(1, feed_data.shape[0], feed_data.shape[1])
		tem_output = model.predict(feed_data)
		print(feed_data)
		print(tem_output)
		data_for_decode = tem_output if i == 0 else np.vstack((data_for_decode, tem_output))
		# tem_output = mode.predict(feed_data)


	return data_for_decode

def predict_vol(dataset, predict_num, sequence_length):

	print(np.max(dataset),np.min(dataset), np.mean(dataset), np.median(dataset))

	encoder = load_model('vel_encoder(AE-64).h5', compile = False) # encoder verified data
	code = encoder.predict(dataset) # dimension with verified data after scaled

	print(np.max(code),np.min(code),np.mean(code), np.median(code), code.shape)

	LSTM = load_model('vel_lstm(ENKF).h5', compile=False)
	outputs = predict_sequences_multiple(LSTM, code, sequence_length, predict_num)
	print(np.max(outputs),np.min(outputs),np.mean(outputs), np.median(outputs), outputs.shape)

	decoder = load_model('vel_decoder(AE-64).h5', compile = False) # decoder predicted data
	predicted_vol = decoder.predict(outputs)
	print(np.max(predicted_vol),np.min(predicted_vol),np.mean(predicted_vol), np.median(predicted_vol))

	return predicted_vol, 

if __name__=="__main__":  

	path, verify_rate, sequence_length, originalFile, destinationFile = variable_value()

	print("Data Preprocessing...")  
	vol = Pcd.get_data(path)
	dataset, verify = Pcd.train_and_vertify(vol,verify_rate) # predict and verify


	scaler_data = MinMaxScaler()
	scaler_vol = scaler_data.fit_transform(verify)

	# scaler_vol = verify

	print("Data Predicting...")  

	print('dataset shape = ' + str(dataset.shape))
	print('vertify shape = ' + str(verify.shape))


	predicted_vol = predict_vol(scaler_vol, verify.shape[0]- sequence_length, sequence_length)
	predicted_vol = np.array(predicted_vol)
	predicted_vol = predicted_vol.reshape(predicted_vol.shape[1], predicted_vol.shape[2])
	print('shape = '+ str(np.shape(predicted_vol)))
	scaler_outpus = scaler_data.inverse_transform(predicted_vol)
	# scaler_outpus = predicted_vol
	

	# value = mean_squared_error(verify, scaler_outpus)
	# print(value)

	mkdirs_org.mkdir(destinationFile)
	mkdirs_org.copyFiles(originalFile,destinationFile)
	vtu_num = Pcd.get_vtu_num(originalFile)
	num = vtu_num - sequence_length
	mkdirs_org.transform(scaler_outpus, num, destinationFile, sequence_length)
