from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold, StratifiedKFold
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten

# Load data step
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)) #60000 set of images
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))    #10000 set of  images
Ytrain = Y_train
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

# Normalization step
Xtrain_norm = X_train.astype('float32')
Xtest_norm = X_test.astype('float32')
X_train = Xtrain_norm / 255.0
X_test = Xtest_norm / 255.0

# Build model step
arch = 4
model = [0] * arch

for j in range(arch):
    #For all model (model_1/model_2/model_3/model_4)
    model[j] = Sequential()
    model[j].add(Conv2D(24,kernel_size=5,padding='same',activation='relu',
            input_shape=(28,28,1)))
    model[j].add(MaxPooling2D((2, 2)))
    if j>0:
        #For (model_2/model_3/model_4)
        model[j].add(Conv2D(32,kernel_size=5,padding='same',activation='relu'))
        model[j].add(MaxPooling2D((2, 2)))
    if j>1:
        #For (model_3/model_4)
        model[j].add(Conv2D(48,kernel_size=5,padding='same',activation='relu'))
        model[j].add(MaxPooling2D((2, 2)))
    if j>2:
        #For(model_4)
        model[j].add(Conv2D(64,kernel_size=5,padding='same',activation='relu'))
        model[j].add(MaxPooling2D((2, 2)))
    #For all model (model_1/model_2/model_3/model_4)
    model[j].add(Flatten())
    model[j].add(Dense(256, activation='relu'))
    model[j].add(Dense(10, activation='softmax'))
    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    

k_fold = StratifiedKFold(n_splits = 3, shuffle = True)  

for i in range(arch):
  print("Model number=",i+1)
  for train, test in k_fold.split(X_train, Ytrain):
    model[i].fit(X_train[train], Y_train[train], validation_data=(X_train[test], Y_train[test]), epochs=3)
  arr = model[i].evaluate(X_test, Y_test) # Evaluate the model
  print('Cost of model ', i+1 , ' = ',arr[0])
  print('Accuracy of test data using model ', i+1 , ' = ', arr[1])
  
