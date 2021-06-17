#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import MaxPool2D
from keras.layers.convolutional import Conv2D
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier


# In[2]:


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()


# In[3]:


print("Training input data shape: " + str(X_train.shape))
print("Training target data shape: " + str(y_train.shape))


# In[4]:


print("Testing input data shape: " + str(X_test.shape))
print("Testing target data shape: " + str(y_test.shape))


# In[5]:


for i in range(9):
  plt.subplot(330 + 1 + i)
  plt.imshow(X_train[i])
plt.show()


# In[6]:


classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for x in y_train[0:5]:
  print('Example training images and their labels: ' + str([x[0]]))
for x in y_train[0:5]:
  print('Corresponding classes for the labels: ' + str(classes[x[0]]))
for i in range(5):
    img = X_train[i]
    plt.subplot(330 + 1 + i)
    plt.imshow(img)


# In[7]:


# Transform images from (32,32,3) to 3072-dimensional vectors (32*32*3)
# Normalization of pixel values (to [0-1] range)
X_train=X_train/255.0
print("Initial X_train shape "+str(X_train.shape))
X_test=X_test/255.0
print("Initial X_test shape"+str(X_test.shape))
X_train = np.reshape(X_train,(50000,3072))
print("Final X_train shape "+str(X_train.shape))
X_test = np.reshape(X_test,(10000,3072))
print("Final X_test shape "+str(X_test.shape))
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# In[8]:


y_train=y_train.reshape(-1,)
y_test=y_test.reshape(-1,)


# In[9]:


#The models we are going to use are: MLP,Random Forest, CNN 
#All models with and without pca


# In[10]:


#MLP 
#Training a MLP classifier from keras with three hidden layers 
def MLP_train(X_train,y_train,X_test,y_test):
  
  #encoding basically one hot encoding
  y_train_encoded=[]
  for i in y_train:
    temp=[]
    for j in range(10):
      if(j!=i):
        temp.append(0)
      else:
        temp.append(1)
    y_train_encoded.append(temp)
  y_test_encoded=[]
  for i in y_test:
    temp=[]
    for j in range(10):
      if(j!=i):
        temp.append(0)
      else:
        temp.append(1)
    y_test_encoded.append(temp)

  #preparing the data for training and k-fold
  X_train=np.array(X_train)
  X_test=np.array(X_test)
  y_train_encoded=np.array(y_train_encoded)
  y_test_encoded=np.array(y_test_encoded)
  inputs = np.concatenate((X_train, X_test), axis=0)
  targets = np.concatenate((y_train_encoded, y_test_encoded), axis=0)

  #array for storing the acuracies and losses per fold
  accuracy_per_fold = []
  loss_per_fold = []
  #number of folds taken is 6
  number_folds=6

  kfold = KFold(n_splits=number_folds, shuffle=True)
  fold_no = 1
  for train, test in kfold.split(inputs, targets):
    #defining the network
    mlp = Sequential()
    mlp.add(Dense(2050,input_dim=X_train.shape[1],activation='relu'))
    mlp.add(Dropout(0.2))
    mlp.add(Dense(1030, activation='relu'))
    mlp.add(Dropout(0.2))
    mlp.add(Dense(510,activation='relu'))
    mlp.add(Dropout(0.2))
    mlp.add(Dense(10, activation='softmax'))

    #compiling the network
    mlp.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    #fitting the network
    mlp.fit(inputs[train], targets[train], epochs=10,batch_size=64,verbose=2)

    #evaluating the network
    score = mlp.evaluate(inputs[test], targets[test], verbose=0)
    print(f'Score for fold {fold_no}: {mlp.metrics_names[0]} of {score[0]}; {mlp.metrics_names[1]} of {score[1]*100}%')
    accuracy_per_fold.append(score[1] * 100)
    loss_per_fold.append(score[0])
    fold_no = fold_no + 1
  
  #Average results
  print('------------------------------------------------------------------------')
  print('Score per fold')
  for i in range(0, len(accuracy_per_fold)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {accuracy_per_fold[i]}%')
  print('------------------------------------------------------------------------')
  print('Average scores for all folds:')
  print(f'> Accuracy: {np.mean(accuracy_per_fold)} (+- {np.std(accuracy_per_fold)})')
  print(f'> Loss: {np.mean(loss_per_fold)}')
  print('------------------------------------------------------------------------')


# In[11]:


#Random Forest
#defining the svm and training and cross validating it
def RF_train(X_train, y_train, X_test, y_test):
  #getting the inputs and targets for kfold
  inputs = np.concatenate((X_train, X_test), axis=0)
  targets = np.concatenate((y_train, y_test), axis=0)
  accuracy_per_fold = []
  number_folds=6

  #number of folds chosen is 6
  fold_no = 1
  kfold = KFold(n_splits=number_folds, shuffle=True)

  for train, test in kfold.split(inputs, targets):
    #declaring and training the model
    rf = RandomForestClassifier(n_estimators=300)
    rf.fit(inputs[train], targets[train])
    print(f'Accuracy for fold {fold_no}',rf.score(inputs[test],targets[test]))
    accuracy_per_fold.append(rf.score(inputs[test],targets[test]))
    fold_no=fold_no+1

  #average results
  print('------------------------------------------------------------------------')
  print('Score per fold')
  for i in range(0, len(accuracy_per_fold)):
    print(f'> Fold {i+1} -  Accuracy: {accuracy_per_fold[i]}%')
  print('------------------------------------------------------------------------')
  print('Average scores for all folds:')
  print(f'> Accuracy: {np.mean(accuracy_per_fold)} (+- {np.std(accuracy_per_fold)})')
  print('------------------------------------------------------------------------')


# In[12]:


#CNN
#Defining training and cross validating a cnn model
def CNN_train(X_train, y_train, X_test, y_test):

  #encoding basically one hot encoding 
  y_train_encoded=[]
  for i in y_train:
    temp=[]
    for j in range(10):
      if(j!=i):
        temp.append(0)
      else:
        temp.append(1)
    y_train_encoded.append(temp)
  y_test_encoded=[]
  for i in y_test:
    temp=[]
    for j in range(10):
      if(j!=i):
        temp.append(0)
      else:
        temp.append(1)
    y_test_encoded.append(temp) 


  #preparing the data for training and k-fold
  X_train=np.array(X_train)
  X_test=np.array(X_test)
  y_train_encoded=np.array(y_train_encoded)
  y_test_encoded=np.array(y_test_encoded)
  X_train=X_train.reshape(-1,32,32,3)
  X_test=X_test.reshape(-1,32,32,3)
  inputs = np.concatenate((X_train, X_test), axis=0)
  targets = np.concatenate((y_train_encoded, y_test_encoded), axis=0)

  #array for storing the acuracies and losses per fold
  acc_per_fold = []
  loss_per_fold = []

  #number of folds taken is 6
  num_folds=6

  kfold = KFold(n_splits=num_folds, shuffle=True)
  fold_no = 1
  for train, test in kfold.split(inputs, targets):
    #defining the network
    cnn = Sequential()
    cnn.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(32,32,3), activation='relu',))
    cnn.add(MaxPool2D(pool_size=(2, 2)))
    cnn.add(Conv2D(filters=32, kernel_size=(4,4), activation='relu',))
    cnn.add(MaxPool2D(pool_size=(2, 2)))
    cnn.add(Flatten())
    cnn.add(Dense(128, activation='relu'))
    cnn.add(Dropout(0.2))
    cnn.add(Dense(64, activation='relu'))
    cnn.add(Dropout(0.2))
    cnn.add(Dense(32, activation='relu'))
    cnn.add(Dropout(0.2))
    cnn.add(Dense(10, activation='softmax'))

    #compiling the network
    cnn.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    #fitting the network
    cnn.fit(inputs[train], targets[train], batch_size=128, epochs=20)

    #evaluating the network
    score = cnn.evaluate(inputs[test], targets[test], verbose=0)
    print(f'Score for fold {fold_no}: {cnn.metrics_names[0]} of {score[0]}; {cnn.metrics_names[1]} of {score[1]*100}%')
    acc_per_fold.append(score[1] * 100)
    loss_per_fold.append(score[0])
    fold_no = fold_no + 1

  #Average results
  print('------------------------------------------------------------------------')
  print('Score per fold')
  for i in range(0, len(acc_per_fold)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
  print('------------------------------------------------------------------------')
  print('Average scores for all folds:')
  print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
  print(f'> Loss: {np.mean(loss_per_fold)}')
  print('------------------------------------------------------------------------')


# In[13]:


#PCA
def get_PCA_data(X_train, X_test):
  pca = PCA()
  X_train_pca = pca.fit_transform(X_train)
  X_test_pca = pca.transform(X_test)
  return X_train_pca, X_test_pca


# In[14]:


#Caling normally
#First MLP
MLP_train(X_train,y_train,X_test,y_test)


# In[15]:


#Caling normally
#Second rf
RF_train(X_train,y_train,X_test,y_test)


# In[16]:


#Caling normally
#Third svm
CNN_train(X_train,y_train,X_test,y_test)


# In[17]:


#calling with pca
#First MLP
X_train_pca, X_test_pca=get_PCA_data(X_train,X_test)
MLP_train(X_train_pca,y_train,X_test_pca,y_test)


# In[18]:


#calling with pca
#Second rf
RF_train(X_train_pca,y_train,X_test_pca,y_test)


# In[20]:


#calling with pca
#Third CNN
CNN_train(X_train_pca,y_train,X_test_pca,y_test)


# In[ ]:




