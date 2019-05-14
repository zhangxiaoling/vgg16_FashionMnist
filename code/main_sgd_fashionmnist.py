#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 09:58:25 2019

@author: zxl
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:53:58 2019

@author: zhangxiaoling
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:14:07 2019

@author: zhangxiaoling
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:46:42 2019

@author: zhangxiaoling
"""

import numpy as np
import time
from datetime import timedelta
from de import DifferentialEvolution
from es_sgd import EvolutionStrategy

import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten,Input,Conv2D,MaxPooling2D
from keras.optimizers import SGD  # not important as there's no training here, but required by Keras.
import tensorflow as tf
from keras import backend as K

import cv2
from keras import datasets
from vgg16 import VGG16
import gzip


start_time = time.time()


# to run model evluation on 1 core
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


#transfer learning
ishape=100
model_vgg=VGG16(include_top=False,weights='imagenet',
               input_shape=(ishape,ishape,3))

model=Model(inputs=model_vgg.input,outputs=model_vgg.get_layer('block4_pool').output)

# load data
data_dir="../Fashion_mnist/"

def extract_data(filename, num_data, head_size, data_size):
    with gzip.open(filename) as bytestream:
        bytestream.read(head_size)
        buf = bytestream.read(data_size * num_data)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data


data = extract_data(data_dir + 'train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
X_train_pre = data.reshape((60000, 28, 28))
 
data = extract_data(data_dir + 'train-labels-idx1-ubyte.gz', 60000, 8, 1)
y_train = data.reshape((60000))
 
data = extract_data(data_dir + 't10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
X_test_pre = data.reshape((10000, 28, 28))
 
data = extract_data(data_dir + 't10k-labels-idx1-ubyte.gz', 10000, 8, 1)
y_test = data.reshape((10000))

X_train_pre=X_train_pre.astype(np.uint8)
X_test_pre=X_test_pre.astype(np.uint8)
y_train=y_train.astype(np.uint8)
y_test=y_test.astype(np.uint8)

print("finished")
print(X_train_pre.shape)
print(X_test_pre.shape)
print(y_train.shape)
print(y_test.shape)

X_train_pre=[cv2.cvtColor(cv2.resize(i,(ishape,ishape)),cv2.COLOR_GRAY2BGR) for i in X_train_pre]
X_train=np.concatenate([arr[np.newaxis] for arr in X_train_pre]).astype('float32')

X_test_pre=[cv2.cvtColor(cv2.resize(i,(ishape,ishape)),cv2.COLOR_GRAY2BGR) for i in X_test_pre]
X_test=np.concatenate([arr[np.newaxis] for arr in X_test_pre]).astype('float32')

print(X_train.shape)
print(X_test.shape)

#X_train/=255
#X_test/=255

def train_y(y):
    y_one=np.zeros(10)
    y_one[y]=1
    return y_one
    
y_train_one=np.array([train_y(y_train[i]) for i in range(len(y_train))])
y_test_one=np.array([train_y(y_test[i]) for i in range(len(y_test))])



batch_size=128
num_classes=10

print(y_train_one[0])
print(y_test_one[0])

#np.savetxt("features.txt", flatten_features)

features_train=[]
for i in range(3000):
    features_train.extend(model.predict(X_train[20*i:20*i+20]))
    print(np.array(features_train).shape)
features_train=np.array(features_train)

features_test=[]
for i in range(500):
    features_test.extend(model.predict(X_test[20*i:20*i+20]))
    print(np.array(features_test).shape)
features_test=np.array(features_test)


input_layer_sgd=Input(shape=(6,6,512,))
Conv1_sgd = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1_sgd')(input_layer_sgd)
Conv2_sgd = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2_sgd')(Conv1_sgd)
Conv3_sgd = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3_sgd')(Conv2_sgd)
Maxpooling_sgd = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool_sgd')(Conv3_sgd)
Flatten_sgd = Flatten(name='flatten_sgd')(Maxpooling_sgd)
hidden_layer1_sgd=Dense(1024, activation='relu')(Flatten_sgd)
dropout1_sgd=Dropout(0.2, noise_shape=None, seed=None)(hidden_layer1_sgd)
hidden_layer2_sgd=Dense(512, activation='relu')(dropout1_sgd)
dropout2_sgd=Dropout(0.2, noise_shape=None, seed=None)(hidden_layer2_sgd)
model_pretrain_sgd=Dense(10,activation='softmax')(dropout2_sgd)
model_vgg_mnist_pretrain_sgd=Model(inputs=input_layer_sgd,outputs=model_pretrain_sgd,name='vgg16_pretrain_sgd')
model_vgg_mnist_pretrain_sgd.summary()
    
sgd=SGD(lr=0.001,decay=1e-5)
model_vgg_mnist_pretrain_sgd.compile(optimizer=sgd,loss='mean_squared_error',
                                 metrics=['accuracy'])
model_vgg_mnist_pretrain_sgd.fit(features_train,y_train_one,batch_size=128,epochs=500,verbose=1,validation_data=(features_test,y_test_one))
score=model_vgg_mnist_pretrain_sgd.evaluate(features_test,y_test_one,verbose=1)
print('Test accuracy:',score[1])
    
