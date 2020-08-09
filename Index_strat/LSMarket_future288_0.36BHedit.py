import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import load_model
# from pprint import pprint as pp
import numpy as np
from sklearn.preprocessing import minmax_scale


verbosity = 1
groupSize = 288

feature_names = [ 'Date', 'BHP', 'RIO', 'OSH', 'WPL' ]
subset = [ 'RIO', 'OSH', 'WPL' ]

voting_data = pd.read_csv( 'a3d_oil_equity.csv', names = feature_names )
voting_data.head()
voting_data.dropna( inplace = True ) # this removes incomplete rows... interesting
voting_data.describe()

def groupData( data, groupSize ):
	mergedArray = []
	for i in range( data.size ):
		if i > groupSize and i < data.shape[0] - 1:
			mergedList = []
			for ii in range( groupSize ):
				mergedList = np.concatenate( ( mergedList, data[ i - ii ] ), axis=None )
			mergedArray.append( mergedList )
	return mergedArray

def groupLabels( data, groupSize ):
	mergedArray = []
	for i in range( data.size ):
		i = i - 1
		if i > groupSize:
			mergedList = data[i]
			mergedArray.append( mergedList )
	return mergedArray


def doTraining():
	trainingData = voting_data[ subset ].values
	trainingData = minmax_scale( trainingData )
	trainingData = groupData( trainingData, groupSize = groupSize )
	trainingData = np.array( trainingData )
	# print( trainingData.shape )

	trainingLabels = voting_data[ 'BHP' ].values
	trainingLabels = minmax_scale( trainingLabels )
	trainingLabels = groupLabels( trainingLabels, groupSize = groupSize )
	trainingLabels = np.array( trainingLabels )
	# print( trainingLabels.shape )

	model = Sequential()

	# 17 feature inputs (votes) going into a 32-unit layer
	model.add( Dense( 576, input_dim = len( trainingData[0] ), kernel_initializer = 'normal', activation = 'relu' ) )

	# Another hidden layer of 16 units
	model.add( Dense( 192, kernel_initializer = 'normal', activation = 'relu' ) )

	# Another hidden layer of 16 units
	model.add( Dense( 64, kernel_initializer = 'normal', activation = 'relu' ) )

	# Output layer with a binary classification ( Democrat or Republican )
	model.add( Dense( 1 ) )

	# Compile model
	model.compile( loss = 'mse', optimizer = 'rmsprop', metrics = [ 'mae' ] )

	# Train model
	model.fit( trainingData, trainingLabels, epochs = 5000, batch_size = 50, verbose = verbosity )

	# Grade the model
	scores = model.evaluate( trainingData, trainingLabels, verbose = verbosity )
	print( "%s: %.2f%%" % ( model.metrics_names[1], scores[1]*100 ) )

	# Save the model
	model.save( 'BHMarket_Model.h5' )

def doPrediction():
	
	trainingData = voting_data[ subset ].values
	originalValue = trainingData[0][0]
	trainingData = minmax_scale( trainingData )
	normalizedValue = trainingData[0][0]

	multiple = originalValue / normalizedValue

	print( originalValue )
	print( normalizedValue )
	print( multiple )

	trainingData = groupData( trainingData, groupSize = groupSize )
	inputData = trainingData[-2]

	trainingLabels = voting_data[ 'Date' ].values
	trainingLabels = groupLabels( trainingLabels, groupSize = groupSize )
	date = trainingLabels[-1]
	
	print( inputData.shape )

	loaded_model = load_model( 'LSMarket_Model.h5' )
	# evaluate loaded model on test data
	loaded_model.compile( loss = 'mse', optimizer = 'rmsprop', metrics = [ 'mae' ] )
	# Predict things...
	print( inputData )
	print( inputData.shape )
	thegoods = loaded_model.predict( inputData.reshape( (1, 864) ), batch_size = None, verbose = verbosity, steps = None )
	print ( date, thegoods * multiple )

doTraining()


# doPrediction()