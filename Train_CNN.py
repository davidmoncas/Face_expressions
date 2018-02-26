import os,cv2
import numpy as np
import matplotlib.pyplot as plt
import h5py

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras import backend as K
K.set_image_dim_ordering('tf')

from keras.utils import np_utils
from keras.models import Sequential,load_model
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras.optimizers import SGD,RMSprop,adam

# Size of the resized images
size=128

img_rows=size
img_cols=size
num_channel=1

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#Cargando las imágenes en una matriz de numpy
img_data_list=[]

data_dir_list=['Angry_chopped','Happy_chopped','Neutral_chopped','Sad_chopped','Very_Happy_chopped']
data_path='C:/Users/David/Desktop/proyecto_CNN'

for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+dataset)
    print ('loaded the images of dataset-'+ dataset)
    for img in img_list:
        input_img=cv2.imread(data_path+'/'+dataset+'/'+img)
        input_img=cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)
        input_img_resize=cv2.resize(input_img,(size,size))
        img_data_list.append(input_img_resize)

img_data=np.array(img_data_list)
img_data=img_data.astype('float32')
img_data/=255

# el formato de entrada para Keras es (datos,columnas, filas,canales )
img_data=np.expand_dims(img_data,axis=4)

print ('data format='+str(img_data.shape))



#putting Labels to the data 
num_classes=5
num_of_samples=img_data.shape[0]
labels=np.ones((num_of_samples),dtype='int64')
labels[0:113]=0
labels[113:218]=1
labels[218:362]=2
labels[362:461]=3
labels[461:]=4

names=['Angry','Happy','Neutral','Sad','Very Happy']

#convirtiendo el vector labels a una matriz categórica
Y=np_utils.to_categorical(labels,num_classes)

#taking random data from the dataset
x,y=shuffle(img_data,Y,random_state=2)

#spliting the data into test and train
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)

#building and training the CNN
imput_shape=img_data[0].shape

model=Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=imput_shape))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(size))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])

hist=model.fit(X_train,y_train,batch_size=24,epochs=20,verbose=1,validation_data=(X_test,y_test))

model.save('Expressions_CNN.p5')

