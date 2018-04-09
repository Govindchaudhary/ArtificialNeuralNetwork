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
#init='uniform' is to initialize the weights according to uniform distribution and also choose the values close to 0
# relu is the activation func to be used (rectifier output funcie. f(x) = max(x,0)
#for the ist hidden layer we have to specify input_dim ie. from where to expect 
#the input in this case from 11 input nodes

classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))

#adding the second hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))

#adding the output layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

#compiling the ANN
#basically applying the stocashtic gradient descent on the whole ANN
# optimizer:algorithm you want to use to find the optimal set ofweights in our neural network
# we are using stocashtic gradient algo to optomize the set of weights
#adam is a special type of stocashtic gradient algo
# loss is the cost funcion within the stocashtic gradient algo and basically we have to optimize this func using stocashtic gradient algo to find the optimal set of weights
# for eg loss func in linear regression is sum of squared differnce
#in this case our cost func or loss func is same as logistic regression
#since we are using a simple neural network with only one neuron and we are using sigmoid func for the output layer
# so we will use loss func same as logistic regression which is lograthmic loss func denoted by crossentropy
# further if we are dealing with binary outcome or dependent varaible we use binary_crossentropy
# and for more than two we use categorical_crossentropy
#finally matrices is the criterion you want to use to evaluate your model
#our algo will use the accuracy criterin to improve our model

classifier.compile(optimizer='adam' , loss='binary_crossentropy' ,metrics=['accuracy'])






# Fitting ANN to the Training set(conneting our ann model to training set
#batch size is the no. of observation after which you want to optimze your weights
#basically we have two options either to update the weights after each observation(reinforcement learning)
#or to update it after a batch of observations (batch learning)
# an epoch is basically a round when the whole training set is passed through the ann
#nb_epoch is here the no. of epochs(based on practice)

classifier.fit(X_train,y_train,batch_size=10,nb_epoch=100)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

y_pred = (y_pred>.5) #true if y_pred > .5 otherwise false basically choosing threshold = 0.5

#classifier.predict expects a 2d array as input 
#so here we are making our data as a 2d array but with only one row entry(France,600,male,40,3,60000,2,1,1,50000)
new_prediction = classifier.predict(sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
new_prediction = (new_prediction>.5)

'''
we need to optimize a way to evaluate our model performance
bcos what we did so far is we splitted our datset into training set and test set
and then we train our data on training set and test it on test set that's
the correct way of evaluating the model performance but not the best way
bcos we actually have the variance problem and this can be explain by saying that
if we evaluate our model then it can give a certain accuracy and then if we again 
test it on another test set we might get a very different result.
so judging our model performance only on one accuracy on one test set is not
super relevant.
so there is a technique called k-fold cross validation that improves it a lot.
it will fix the variance problem and it does it by splitting our training set 
into k folds(generally k=10) and now training our model on 9 folds and test the 
result on remaining fold .Thus we have 10 diff. combn of 9 training fold and 1 test fold
thus we evaluate the accuracy of our model as average of these 10 accuracies.

'''

#here we implemnted our model with keras and k-fold cross validation function belongs to scikit-learn
#therefore we need to somehow connect these two together
#and there is a perfect module belonging to keras for this
#it is a keras wrapper that will wrap the k-fold cross validation by sckit-learn into keras model
#in other words with the help of this we can use k-fold cross validation in our keras model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

#now we have to make a func which builds the architecture of our ANN bcos KerasClassifier expects its first argument a func which return the classifier with all the architecture of ANN

def build_classifier():
    classifier = Sequential() 
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
    classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
    classifier.compile(optimizer='adam' , loss='binary_crossentropy' ,metrics=['accuracy'])
    return classifier


#this classifier is same as the one we make above but the diff is that we will not
#train it to whole training set but rather we use k-fold cross validation for training
#this is basically wrapping our classifier to kerasclassifier to use the k-fold cross validation
    
classifier = KerasClassifier(build_fn = build_classifier,batch_size=10,nb_epoch=100)
#now lets apply k-fold cross validation uisng cross_val_score
#contain all the 10 accuracies return by k-fold technique
# here estimator is The object to use to fit the data , cv is the number of fold and n_jobs=-1 indicates that we want to use all the cores of our cpu for this
accuracies = cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10,n_jobs=-1)
accuracies.mean()
accuracies.std()

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
