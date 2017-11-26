#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

import os
import glob
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop, SGD
from keras.utils import np_utils

Y = []
R = []
R_TEST = []
Y_TEST = []
ct = 0
num_classes = 5
epochs = 100
batch_size = 1

for r in glob.glob("./TRAIN/*/"):
	label = r.split("/")[-2]
	if ct==num_classes:
		break	
	for fil in glob.glob(r+"*.npy"):
		Y.append(ct)
		X = []
		ceps = np.load(fil)
		num_ceps = len(ceps)
		X = (np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0, dtype=np.float32))
		# X = ceps
		X = np.array(X)
		R.append(X)
	ct += 1

ct = 0
for r in glob.glob("./TEST/*/"):
	label = r.split("/")[-2]
	if ct == num_classes:
		break
	for fil in glob.glob(r+"*.npy"):
		Y_TEST.append(ct)
		X = []
		ceps = np.load(fil)
		num_ceps = len(ceps)
		X = (np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0, dtype=np.float32))
		# X = ceps
		X = np.array(X)
		R_TEST.append(X)
	ct += 1	


R = np.array(R)
R_TEST = np.array(R_TEST)
Y=np.array(Y)
Y_TEST=np.array(Y_TEST)
# R = R.reshape((-1, 1))
# R_TEST = R_TEST.reshape((-1,1))
Y = Y.reshape((-1, 1))
# Y_TEST = Y_TEST.reshape((-1,1))
print R.shape

# Y_train = keras.utils.to_categorical(Y, num_classes)
# Y_valid = keras.utils.to_categorical(Y_TEST, num_classes)

model = Sequential()

model.add(Dense(64, input_dim=R.shape[1]))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('relu'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='sparse_categorical_crossentropy',
			  optimizer=sgd,
			  metrics=['accuracy'])

model.fit(R, Y, epochs=epochs, validation_data = (R_TEST,Y_TEST), batch_size=32)
