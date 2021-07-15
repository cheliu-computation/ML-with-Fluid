import os
import sys
sys.path.append("/home/david/thesis/lib/python3.8/site-packages")
import vtktools
import numpy as np
import matplotlib.pyplot as plt

def get_vtu_num(path):
# count the number of vtu files
	f_list = os.listdir(path) 
	vtu_num = 0
	for i in f_list:
		if os.path.splitext(i)[1] == '.vtu':
			vtu_num = vtu_num+1
	
	return vtu_num

def get_vol_data(path, vtu_num):

	for n in range(vtu_num):   
		filename = path + "/backward_facing_step_2d_" + str(n)+ ".vtu"# name of vtu files
		data = vtktools.vtu(filename)
		
		
		uvw = data.GetVectorField('Velocity')
		
		ui = np.hsplit(uvw,3)[0].T #velocity of x axis
		vi = np.hsplit(uvw,3)[1].T #velocity of y axis
		wi = np.hsplit(uvw,3)[2].T #velocity of z axis
		veli = np.hstack((ui,vi,wi)) #combine all into 1-d array
		print(n)
		vel = veli if n==0 else np.vstack((vel,veli))
	w = vel[:,int(vel.shape[1]/3)*2:]
	outputs = vel[:,:int(vel.shape[1]/3)*2] if np.all(w) == 0 else vel
	np.save('Velocity.npy',outputs)
	print(outputs.shape)
	return outputs
def get_mag_data(path, vtu_num):

	for n in range(vtu_num):   
		filename = path + "/backward_facing_step_2d_" + str(n)+ ".vtu"# name of vtu files
		data = vtktools.vtu(filename)
		def prod(a,b):
			res = np.sqrt(np.square(a)+np.square(b))
			return res
		
		uvw = data.GetVectorField('Velocity')
		
		ui = np.hsplit(uvw,3)[0].T #velocity of x axis
		vi = np.hsplit(uvw,3)[1].T #velocity of y axis
		
		veli = np.array(list(map(prod, ui,vi))) #combine all into 1-d array
		print(n)
		vel = veli if n==0 else np.vstack((vel,veli))
	
	outputs = vel
	np.save('magnitude.npy',outputs)
	print(outputs.shape)
	return outputs


def get_data(path):

	vtu_num = get_vtu_num(path)
	vol_fraction = np.load('magnitude.npy') if os.path.exists('magnitude.npy') else get_mag_data(path, vtu_num)

	return vol_fraction


def train_and_vertify(dataset,vertify_rate):

	# divide dataset into train_dataset and vertify_dataset(70%,30%)
	vertify_point = int(dataset.shape[0] * (1 - vertify_rate))
	train = dataset[:vertify_point,:]
	vertify = dataset[vertify_point:,:]

	return np.array(train), np.array(vertify)

def draw_acc_loss(history):

	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()

	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()
