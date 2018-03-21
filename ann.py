# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#since our dataset involves caterogical independent variable
#we have to encode the caterogical data

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1]) #encode the 2nd column of the dataset
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])

#we need to create the dummy variable for the independent variable country of column1
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()

#remove one dummy variable to remove the dummy variable trap
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#making ANN

#importing the keras libraries and packages
import keras
from keras.models import Sequential  #to initialize the ANN
from keras.layers import Dense     # to create layers in ANN

#initializing the ANN (2 methods defining the sequence of layers or defining a graph)
# we will go with the defining the sequence of layers

classifier = Sequential()  # we are defining our neural network model as a classifier

#adding the input layer and the 1st hidden layer
#output_dim is the no. of node in layer
#totally based on practice but we can take it as avg of no.of node in input and output layer
#activation is the activation func to be used
#for the ist hidden layer we have to specify input_dim ie. from where to expect 
#the input in this case from 11 input nodes

classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))

#adding the second hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))

#adding the output layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

#compiling the ANN
classifier.compile(optimizer='adam' , loss='binary_crossentropy' ,metrics=['accuracy'])





# Fitting ANN to the Training set

classifier.fit(X_train,y_train,batch_size=10,nb_epoch=100)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

y_pred = (y_pred>.5) #true if y_pred > .5 otherwise false

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)