import pandas as pd
import numpy as np
import tensorflow as tf

from Functions.conn import db_upload
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import cross_val_score
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import load_model
from sklearn.preprocessing import minmax_scale
print(tf.__version__)

verbosity = 1
groupSize = 288

index_names = [ 'Date', 'ASA51', 'XNDX', 'SPXT', 'CACR', 'DAX', 'NKYTR', 'HSI1', 'KSP2TR']

subset = [ 'ASA51', 'XNDX', 'SPXT', 'CACR', 'DAX', 'NKYTR', 'HSI1', 'KSP2TR' ]

#note, raw.csv omits 'date' label in cell A1 of csv but incl's all index names in A2++ so csv can be read more easily but re "dropna"
#rule below, the header row is omitted from all program calculations
index_data = pd.read_csv( 'raw.csv', names = index_names ) # read file, insert header row
index_data.dropna( inplace = True ) # this removes rows with >=1 #na, being 234/1,304 data points in v1 dataset, inplace= True, 'do operation inplace and return None.''
# index_data.describe() #prints some table metrics, not part of any calculation hereafter.
print("index_data shape is ")
print(index_data.shape)
index_data.to_csv('out.csv', index=False)


ASAMin=0
ASAMax=0
ASArange=0

#function to create trainingArray that groups data into lists of 'groupSize' date rows (atm 288) * all indices' data (8)
#creating a master list of 780 rows of 2304 columns, essentially 780 moving windows of all data
def groupData( data, groupSize ):
	trainingArray = []
	for i in range( data.size ):
		if i >= groupSize and i < data.shape[0] :
			trainingListSingleEntry = []
			for ii in range( groupSize ):
				trainingListSingleEntry = np.concatenate( ( trainingListSingleEntry, data[ i - ii -1] ), axis=None )
			trainingArray.append( trainingListSingleEntry )
	return trainingArray


def groupLabels( data, groupSize ):
	trainingArray = []
	for i in range( data.size):
		if i >= groupSize:
			trainingListSingleEntry = data[i]
			trainingArray.append( trainingListSingleEntry )
	return trainingArray

def doTraining():
	trainingData = index_data[ subset ].values
	trainingData = minmax_scale( trainingData )
	trainingData = groupData( trainingData, groupSize = groupSize )
	trainingData = np.array( trainingData )
# 	print("training data 1", trainingData[0])
# 	print(trainingData.shape)

#trainingLabels being target, output dataset
	trainingLabels = index_data[ 'ASA51' ].values
	trainingLabels = minmax_scale( trainingLabels )
# 	print("training label size is ")
# 	print(trainingLabels.size)
	trainingLabels = groupLabels( trainingLabels, groupSize = groupSize )
	trainingLabels = np.array( trainingLabels )
	print("training label row 1",trainingLabels[0])

# 	print("training label shape is ")
# 	print(trainingLabels.shape)

	model = Sequential()
	print("input layer set to:", len( trainingData[0] ))
	# going into a 32-unit layer
	model.add( Dense( 32, input_dim = len( trainingData[0] ), kernel_initializer = 'normal', activation = 'relu' ) )
	# Another hidden layer of 16 units
	model.add( Dense( 32, kernel_initializer = 'normal', activation = 'relu' ) )
	# Another hidden layer of 16 units
	model.add( Dense( 32, kernel_initializer = 'normal', activation = 'relu' ) )
	# Output layer with a binary classification ( Democrat or Republican )
	model.add( Dense( 1 ) )

	# Compile model
	model.compile( loss = 'mse', optimizer = 'rmsprop', metrics = [ 'mae' ] )

	# Train model
	model.fit( trainingData, trainingLabels, epochs = 100, batch_size = 50, verbose = verbosity )

	# Grade the model
	scores = model.evaluate( trainingData, trainingLabels, verbose = verbosity )
	print( "%s: %.2f%%" % ( model.metrics_names[1], scores[1]*100 ) )

	# Save the model
	model.save( 'BHMarket_Model.h5' )
	# plot_model(model, to_file='modelplotbh.png')
    
def doPrediction():
	trainingData = index_data[ subset ].values
	ASAmin =  float((trainingData.min(axis=0))[0])
	print("min in trainingData is: ", ASAmin)
	ASAmax =  float((trainingData.max(axis=0))[0])
	print("max in trainingData is: ", ASAmax)
	ASArange =  ASAmax-ASAmin
	trainingData = minmax_scale( trainingData )
	trainingData = groupData( trainingData, groupSize = groupSize )
	inputData = trainingData[-2]
	trainingLabels = index_data[ 'Date' ].values
	trainingLabels = groupLabels( trainingLabels, groupSize = groupSize )
	date = trainingLabels[-1]
# 	print( "training data shape is", inputData.shape )

	loaded_model = load_model( 'BHMarket_Model.h5' )

	# loaded_model = load_model( 'LSMarket_Model.h5' )
	# evaluate loaded model on test data
	loaded_model.compile( loss = 'mse', optimizer = 'rmsprop', metrics = [ 'mae' ] )
	# Predict things...
# 	print( inputData )
# 	print( inputData.shape )
	prediction_1 = loaded_model.predict( inputData.reshape( (1, 2304) ), batch_size = None, verbose = verbosity, steps = None )
	print("+1 day prediction scaled is:", prediction_1)    
	prediction_1=prediction_1*ASArange+ASAmin
	print("+1 day prediction is:", prediction_1)

doTraining()

print("prediction is: ")
doPrediction()