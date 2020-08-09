from Functions.conn import db_upload
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import load_model
# from pprint import pprint as pp
import numpy as np
from sklearn.preprocessing import minmax_scale
# from keras.utils import plot_model


verbosity = 1
groupSize = 288

feature_names = [ 'Date', 'ASA51', 'XNDX', 'SPXT', 'CACR', 'DAX', 'NKYTR', 'HSI1', 'KSP2TR']

subset = [ 'ASA51', 'XNDX', 'SPXT', 'CACR', 'DAX', 'NKYTR', 'HSI1', 'KSP2TR' ]

#note, raw.csv omits 'date' label in cell A1 of csv but incl's all index names in A2++ so csv can be read more easily but re "dropna"
#rule below, the header row is omitted from all program calculations
voting_data = pd.read_csv( 'raw.csv', names = feature_names ) # read file
voting_data.head()
voting_data.dropna( inplace = True ) # this removes rows with >=1 #na, being 234/1,304 data points in v1 dataset
voting_data.describe()
print("voting data shape is ")
print(voting_data.shape)

#BH diagnostics, not part of prog'm
# trainingLabels = voting_data[ 'ASA51' ].values
# trainingLabels = minmax_scale( trainingLabels )
# print(max(trainingLabels))
#end of BH diagnostics, not part of prog'm

def groupData( data, groupSize ):
	mergedArray = []
	for i in range( data.size ):
		if i >= groupSize and i < data.shape[0] :
			mergedList = []
			for ii in range( groupSize ):
				mergedList = np.concatenate( ( mergedList, data[ i - ii -1] ), axis=None )
			mergedArray.append( mergedList )
	return mergedArray

def groupLabels( data, groupSize ):
	mergedArray = []
	for i in range( data.size):
		if i >= groupSize:
			mergedList = data[i]
			mergedArray.append( mergedList )
	return mergedArray

def doTraining():
	trainingData = voting_data[ subset ].values
	trainingData = minmax_scale( trainingData )
	print("training data size before group data function is")
	print(trainingData.size)
	trainingData = groupData( trainingData, groupSize = groupSize )
	trainingData = np.array( trainingData )
	print("training data shape is ")
	print(trainingData.shape)

#trainingLabels being target, output dataset
	trainingLabels = voting_data[ 'ASA51' ].values
	trainingLabels = minmax_scale( trainingLabels )
	print("training label size is ")
	print(trainingLabels.size)
	trainingLabels = groupLabels( trainingLabels, groupSize = groupSize )
	trainingLabels = np.array( trainingLabels )
	print("training label shape is ")
	print(trainingLabels.shape)

	model = Sequential()

	# 17 feature inputs (votes) going into a 32-unit layer
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
	model.fit( trainingData, trainingLabels, epochs = 200, batch_size = 50, verbose = verbosity )

	# Grade the model
	scores = model.evaluate( trainingData, trainingLabels, verbose = verbosity )
	print( "%s: %.2f%%" % ( model.metrics_names[1], scores[1]*100 ) )

	# Save the model
	model.save( 'BHMarket_Model.h5' )
	# plot_model(model, to_file='modelplotbh.png')

def doPrediction():
	
	trainingData = voting_data[ subset ].values
	print(trainingData)
	originalValue = float(trainingData[0][0])
	print(originalValue)

	trainingData = minmax_scale( trainingData )
	normalizedValue = float(trainingData[0][0])
	print(normalizedValue)

	multiple = originalValue / normalizedValue

	# print( originalValue )
	# print( normalizedValue )
	# print( "multiple is")
	# print( multiple )

	trainingData = groupData( trainingData, groupSize = groupSize )
	inputData = trainingData[-2]

	trainingLabels = voting_data[ 'Date' ].values
	trainingLabels = groupLabels( trainingLabels, groupSize = groupSize )
	date = trainingLabels[-1]
	
	print( inputData.shape )

	loaded_model = load_model( 'BHMarket_Model.h5' )

	# loaded_model = load_model( 'LSMarket_Model.h5' )
	# evaluate loaded model on test data
	loaded_model.compile( loss = 'mse', optimizer = 'rmsprop', metrics = [ 'mae' ] )
	# Predict things...
	print( inputData )
	print( inputData.shape )
	thegoods = loaded_model.predict( inputData.reshape( (1, 2304) ), batch_size = None, verbose = verbosity, steps = None )
	print ( date, thegoods * multiple )
	thegoodsmultiplied=int(thegoods * multiple)
	print(thegoodsmultiplied)

	#sql insertion
	sql = "INSERT INTO daily_ai_indicators (metric, value) VALUES (%s, %s)"
	val = ("delta_strategy", thegoodsmultiplied)
	db_upload(sql,val)

doTraining()

doPrediction()


# sql = "INSERT INTO test_data (id, name, fav_num) VALUES (%s, %s, %s)"
# val = (111, "briman", 22)
# db_upload(sql,val)

