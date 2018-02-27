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
The first step was to crop the area in which the face is, I used haarcascade_frontalface_default.xml file and used a cascade classifier to find the faces in the dataset, then I manually delete those which wasn't right. 

'''
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

'''
