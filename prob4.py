#!/usr/bin/env python
# coding: utf-8

# In[1]:


## (가) (10점) Tensorflow를 사용하여 위에 주어진 feed-forward 네트워크 구조를 구현하여 
# 50 epoch 동안 학습한 후, training data와 validation data의 training curve를 그리시오. (Hint: 수업 자료 마지막 실험)
# Load keras modules and packages

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers
from sklearn.preprocessing import minmax_scale 

import matplotlib.pyplot as plt
import os
import numpy as np


# In[2]:


# Read MNIST dataset and reshaping

# Train, Test 데이터 Load 
(X_train, Y_train), (X_validation, Y_validation) = mnist.load_data()

# Train 데이터 포맷 변환 
# 60000(Train Sample 수) * 28(가로) * 28(세로) 포맷
# X_train.shape[0] - Train 샘플 수 
# Feature Scaling 
# X_train의 각 원소는 0-255 사이의 값을 가지고 있다 
# Overfitting 방지 및 Cost 함수의 빠른 수렴을 위해서 
# Feature Scaling 작업을 한다. 
# 예제에서는 0-255 범위를 0-1 범위로 Scaling

num_of_train_samples = X_train.shape[0] # Train Sample 수 
width = X_train.shape[1] # 가로 길이 
height = X_train.shape[2] # 세로 길이 
X_train = X_train.reshape(num_of_train_samples, width * height)

# Test 데이터 포맷 변환 
# width, height는 Train 데이터와 같으므로 재사용 
# 10000(Test Sample 수) * 28(가로) * 28(세로) 포맷을
# 10000(Test Sample 수) * 784(= 28 * 28) 포맷으로 수정 

num_of_test_samples = X_validation.shape[0] # Sample 수 
X_validation = X_validation.reshape(num_of_test_samples, width * height)


# 나누기 연산이 들어가므로 uint8을 float64로 변환한다 
X_train = X_train.astype(np.float64) 
X_validation = X_validation.astype(np.float64)

X_train = minmax_scale(X_train, feature_range=(0, 1), axis=0) 
X_validation = minmax_scale(X_validation, feature_range=(0, 1), axis=0)


# Lable의 categorical 값을 One-hot 형태로 변환 
# 예를 들어 [1, 3, 2, 0] 를 
# [[ 0., 1., 0., 0.], 
# [ 0., 0., 0., 1.], 
# [ 0., 0., 1., 0.], 
# [ 1., 0., 0., 0.]] 
# 로 변환하는 것을 One-hot 형태라고 함 
# MNIST Label인 0 ~ 9사이의 10가지 값을 변환한다

Y_train = np_utils.to_categorical(Y_train, 10)
Y_validation = np_utils.to_categorical(Y_validation, 10)


# In[3]:


# Construct model
# Multilayer Perceptron (MLP) 생성

model = Sequential()

# glorot_uniform == Xavier Initialization

# 첫 번째 Layer ( Input Lavyer )
model.add(Dense(64, input_dim=width * height, init='glorot_uniform', activation='relu'))
model.add(Dropout(0.3))

# 두 번째 Layer (Hidden layer 1)
model.add(Dense(64,  init='glorot_uniform', activation='relu'))
model.add(Dropout(0.3))

# 세 번째 Layer (Hidden layer 2)
model.add(Dense(64,  init='glorot_uniform', activation='relu'))
model.add(Dropout(0.3))

# 네 번째 Layer (Hidden layer 3)
model.add(Dense(64,  init='glorot_uniform', activation='relu'))
model.add(Dropout(0.3))

# 다섯 번째 Layer ( Output Layer )
model.add(Dense(10, activation='softmax'))

# Cost function 및 Optimizer 설정 
# Multiclass 분류이므로 Cross-entropy 사용 
# SGD optimizer 사용
# optimezer : SGD ( learning rate = 0.01 )

sgd = optimizers.SGD(lr=0.01) 
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


# In[5]:


# Model training 
# batch size = 200 
# epochs = 50 

history = model.fit(X_train, Y_train, 
                    validation_data=(X_validation, Y_validation),
                    epochs=50, batch_size=200)

print('\nAccuracy:{:.4f}'.format(model.evaluate(X_validation,
                                                    Y_validation)[1]))

y_vloss = history.history['val_loss']
y_loss = history.history['loss']

y_vloss = history.history['val_loss']
y_loss = history.history['loss']


# In[6]:


x_len = np.arange(len(y_loss))
plt.plot(x_len, y_loss, marker='.', c='blue', label="Train-set Loss")
plt.plot(x_len, y_vloss, marker='.', c='red',

label="Validation-set Loss")

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.show()


# In[7]:


## (나) (10점) 모든 hidden layer의 node 수를 512로 변환한 후, (가)와 동일한 실험을 진행하고 그 결과를 비교하시오. 

# Construct model
# Multilayer Perceptron (MLP) 생성

model = Sequential()

# glorot_uniform == Xavier Initialization

# 첫 번째 Layer ( Input Lavyer )
model.add(Dense(64, input_dim=width * height, init='glorot_uniform', activation='relu'))
model.add(Dropout(0.3))

# 두 번째 Layer (Hidden layer 1)
model.add(Dense(512,  init='glorot_uniform', activation='relu'))
model.add(Dropout(0.3))

# 세 번째 Layer (Hidden layer 2)
model.add(Dense(512,  init='glorot_uniform', activation='relu'))
model.add(Dropout(0.3))

# 네 번째 Layer (Hidden layer 3)
model.add(Dense(512,  init='glorot_uniform', activation='relu'))
model.add(Dropout(0.3))

# 다섯 번째 Layer ( Output Layer )
model.add(Dense(10, activation='softmax'))

# Cost function 및 Optimizer 설정 
# Multiclass 분류이므로 Cross-entropy 사용 
# SGD optimizer 사용
# optimezer : SGD ( learning rate = 0.01 )

sgd = optimizers.SGD(lr=0.01) 
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Model training 
# batch size = 200 
# epochs = 50 

history = model.fit(X_train, Y_train, 
                    validation_data=(X_validation, Y_validation),
                    epochs=50, batch_size=200)

print('\nAccuracy:{:.4f}'.format(model.evaluate(X_validation,
                                                    Y_validation)[1]))

y_vloss = history.history['val_loss']
y_loss = history.history['loss']

y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = np.arange(len(y_loss))
plt.plot(x_len, y_loss, marker='.', c='blue', label="Train-set Loss")
plt.plot(x_len, y_vloss, marker='.', c='red',

label="Validation-set Loss")

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.show()






# In[8]:


### (다) (5점) optimizer를 Adam으로 변환한 후, (가) (나) 의 결과와 비교하시오.


model = Sequential()

# glorot_uniform == Xavier Initialization

# 첫 번째 Layer ( Input Lavyer )
model.add(Dense(64, input_dim=width * height, init='glorot_uniform', activation='relu'))
model.add(Dropout(0.3))

# 두 번째 Layer (Hidden layer 1)
model.add(Dense(512,  init='glorot_uniform', activation='relu'))
model.add(Dropout(0.3))

# 세 번째 Layer (Hidden layer 2)
model.add(Dense(512,  init='glorot_uniform', activation='relu'))
model.add(Dropout(0.3))

# 네 번째 Layer (Hidden layer 3)
model.add(Dense(512,  init='glorot_uniform', activation='relu'))
model.add(Dropout(0.3))

# 다섯 번째 Layer ( Output Layer )
model.add(Dense(10, activation='softmax'))

# Cost function 및 Optimizer 설정 
# Multiclass 분류이므로 Cross-entropy 사용 
# adam optimizer 사용

#sgd = optimizers.SGD(lr=0.01) 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model training 
# batch size = 200 
# epochs = 50 

history = model.fit(X_train, Y_train, 
                    validation_data=(X_validation, Y_validation),
                    epochs=50, batch_size=200)

print('\nAccuracy:{:.4f}'.format(model.evaluate(X_validation,
                                                    Y_validation)[1]))

y_vloss = history.history['val_loss']
y_loss = history.history['loss']

y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = np.arange(len(y_loss))
plt.plot(x_len, y_loss, marker='.', c='blue', label="Train-set Loss")
plt.plot(x_len, y_vloss, marker='.', c='red',

label="Validation-set Loss")

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.show()







# In[ ]:




