Machine learning project. Group Project made by Priyansh and Prannet.

Topic chosen is Natural image processing. The dataset chosen for the same is Cifar10.
We have trained three different machine learning models. Which are Random forest,
Multi layer perceptron and Convolutional neutral network.
All these models are first trained normally and then are trained by using a
dimensionality reduction technique known as Principal component Analysis which
basically reduces the dimesionality of our dataset by projecting it in the direction
of maximum variance. 6 fold cross validation was done for all the models and each fold
accuracy was calulated and analysed.
Structure of the code. The is basically made of four functions namely in order:
1. MLP_train: This is the function which uses keras library. It has all the
prospects that the code performs i.e it contains model defining, compliling,
trainging as well as cross validating.There are three hidden layers in the
model one with 2050,1030,510 neurons each layer has a dropout layer of 0.2 in
between.The optimisise used is adam and relu is used for activation.
2. RF_train:It uses sklern library. It also has all the prospects that the code
performs i.e it contains model defining, trainging as well as cross
validating.The number of tress taken are 300.
3. CNN_train: This function which uses keras library. It has all the prospects
that the code performs i.e it contains model defining, compliling, trainging as
well as cross validating.Firstly there is a convolution network with two (4
cross 4) conv layers each layer has a maxpool (2*2) There are three hidden
layers in the model one with 2050,1030,510 neurons each layer has a dropout
layer of 0.2 in between.The optimisise used is adam and relu is used for
activation.
4. get_PCA_data: This functions returns the data after applying PCA on it.
For getting traing any model of the above three all we have to do is to run that model
with the parameters being X_train, y_train, X_test, y_test.
Process of execution: For each of the following function the exution happens like:
1.We split the data into folds using kfold 2.We initialise the model. 3.We train the
model using the training split then we test the model and add it to the fold accuracy.
4.This process repeats for all the folds. Then we find the average accuracies and also
average losses in case of mlp and cnn.
