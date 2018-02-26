import numpy as np
import cv2
from keras.models import load_model

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model=load_model('Expressions_CNN.p5')
names=['Angry','Happy','Neutral','Sad','Very Happy']
size=128


#function that returns the prediction of the image
def predecir2(imagen): 

    img=cv2.resize(imagen,(size,size))
    img_data=np.array(img)
    img_data=img_data.astype('float32')
    img_data/=255
    img_data=np.expand_dims(img_data,0)
    img_data=np.expand_dims(img_data,3)
    idx=np.argmax(model.predict(img_data))
    return names[int(idx)]

#capturing the image in the webcam
cap = cv2.VideoCapture(0)

ret1,frame1=cap.read()

#Creating color effects to show depending of the facial expression
yellow = np.zeros((frame1.shape[0], frame1.shape[1], 3), np.uint8)
blue=np.zeros((frame1.shape[0], frame1.shape[1], 3), np.uint8)
yellow2 = np.zeros((frame1.shape[0], frame1.shape[1], 3), np.uint8)
red=np.zeros((frame1.shape[0], frame1.shape[1], 3), np.uint8)

yellow[:]=(0,50,50)
blue[:]=(60,0,0)
yellow2[:]=(0,85,85)
red[:]=(0,0,60)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    #Converting to gray and finding the face in the video
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)

    #predicting the expression in the image
    for (x,y,w,h) in faces:
        sub_face = gray[y:y+w, x:x+h]
        expression=predecir2(sub_face)
        if min([w,h])>100:
            cv2.putText(frame,expression,(x,y+20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(frame,expression,(x,y+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1,cv2.LINE_AA)

        #showing the color effect for every expression

        if expression=="Sad":
        	cv2.imshow('frame',cv2.add(frame,blue))
        elif expression=="Happy":
        	cv2.imshow('frame',cv2.add(frame,yellow))
        elif expression=="Very Happy":
        	cv2.imshow('frame',cv2.add(frame,yellow2))
        elif expression=="Angry":
        	cv2.imshow('frame',cv2.add(frame,red))

        else:
        	cv2.imshow('frame',frame)
    
    #wating for the user to press "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()