# Face expressions
## Convolutional neural networks proyect
Create a program in python capable of recognize different face expressions, such as Happy, sad, angry, neutral. Using Keras and openCv

![Image samples](https://github.com/davidmoncas/Face_expressions/blob/master/samples/faces.jpg)

I want the program to change the color of the image depending of the facial expression, but also to put a text above the face saying what expression is it.

The project consists in three parts:
1) prepare the data
2) create and train the neural network
3) Predicting the expressions live with the webcam

### 1) Preparing the data
I took around 500 photos of myself making different facial expressions, the size of the images was 1280x720. 
In a first attempt, I used the whole images for training the neural network, but I got pretty bad results, so I decieded to use only the faces as the training data.
The first step was to crop the area in which the face is, I used haarcascade_frontalface_default.xml file and used a cascade classifier in openCv to find the faces in the dataset, to then save that crop face in another directory. Some of the images were worg so I have to delete it manually. 

```
		img=cv2.imread(folder + "/" + image)
		gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		#detecting the faces using haar cascades
		faces=face_cascade.detectMultiScale(gray,1.3,5)

		for (x,y,w,h) in faces:
			# In order to be a square image, I choose the maximum between h and w
			maxo=max([w,h])
			sub_face = img[y:y+maxo, x:x+maxo]
		
		#It will only save images larger than 50x50
		if maxo>50:
			i+=1
			# Saving the image in the specified folder
			face_file_name = folder + "_chopped/" + folder + "_" + str(i) + ".jpg"
			cv2.imwrite(face_file_name, sub_face)
```

![dataset](https://github.com/davidmoncas/Face_expressions/blob/master/samples/caras.jpg)

At the end of this first step I had 500 choped faces in different expressions organized in folders.



### 2) create and train the neural network

I used Keras and tensorflow to build the convolutional neural network, first, the dataset has to be resized and converted to grayscale in order to improve the performance of the training. I resized the images to 128x128. I used this CNN:

```
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

```

### 3) Predicting the expressions live with the webcam

Loading the trained model, I was able to predict the expression of the face of an image, the next step wast to feed the model with images from a webcam, to make the prediction live and to change the color of the image.

```
      expression=predecir2(sub_face)
        
        if expression=="Sad":
        	cv2.imshow('frame',cv2.add(frame,blue))
        elif expression=="Happy":
        	cv2.imshow('frame',cv2.add(frame,yellow))
        elif expression=="Very Happy":
        	cv2.imshow('frame',cv2.add(frame,yellow2))
        elif expression=="Angry":
        	cv2.imshow('frame',cv2.add(frame,red))
```

The final proyect is able to recognize the following expressions: Happy, Very Happy, Sad, Neutral, and Angry. Although it isn't ver accurate with the sad and neutral expressions. Maybe using a larger dataset and different sources of lights for the pictures could bring better results. 
