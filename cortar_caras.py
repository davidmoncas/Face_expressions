import numpy as np
import cv2
import os

#this file has to be in your folder
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#theese are the folders in which you'll search for the images tho chop the faces
folders=['Sad','Happy','Neutral','Angry','Very_Happy']

for folder in folders:
	os.makedirs(folder+"_chopped")
	files=os.listdir(folder)
	i=0
	for image in files:
		#reading and converting the images to grayscale
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


