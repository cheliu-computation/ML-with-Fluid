import os
import sys
sys.path.append("/home/david/thesis/lib/python3.8/site-packages")
import vtktools
import numpy as np
import shutil


# copy original files
def copyFiles(sourceDir,targetDir):
    if sourceDir.find("exceptionfolder")>0:
        return

    for file in os.listdir(sourceDir):
        sourceFile = os.path.join(sourceDir,file)
        targetFile = os.path.join(targetDir,file)

        if os.path.isfile(sourceFile):
            if not os.path.exists(targetDir):
                os.makedirs(targetDir)
            if not os.path.exists(targetFile) or (os.path.exists(targetFile) and (os.path.getsize(targetFile) !=os.path.getsize(sourceFile))):
                open(targetFile, "wb").write(open(sourceFile, "rb").read())
                # print(targetFile+" copy succeeded")

        if os.path.isdir(sourceFile):
            copyFiles(sourceFile, targetFile)

# create new folder
def mkdir(path):
	folder = os.path.exists(path)

	if folder:                   
		print ("---  We already have this folder name  ---")
		shutil.rmtree(path, ignore_errors=True)
		print ("---  We already delete this folder  ---")
	else:
		print ("---  create new folder...---")
		os.makedirs(path)
		("---  OK  ---")

def transform(predicted_data, num, destinationFile, sequence_length):

	# print(type(predicted_data))
	# print(type(num))
	# print(type(destinationFile))
	print(num)
# 	# replace velocity with new output data 
	for i in range(num):
		
		f_filename=destinationFile + "/water_collapse_" + str(i+ 210+ sequence_length)+ ".vtu"
		f_file = vtktools.vtu(f_filename) 
		
		newdata = predicted_data[i:i+1,:].T
		
		print(i)
		
		f_file.AddScalarField("new_material_enkf", newdata)
		f_file.Write(f_filename)
		
	print('transform succeed')	