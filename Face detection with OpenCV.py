
# Import the necessary libraries
import numpy as np
import cv2 
import matplotlib.pyplot as plt


# loading the classifiers 

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


cap = cv2.VideoCapture('face.mp4')
#out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','P','E','G'), 10, (768,432)) 

    
while 1:
    ret, img = cap.read()
    if ret == True:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    
        for (x,y,w,h) in faces:
            
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            
            
            #to play the full video with bounding box uncomment the next line
            #cv2.imshow('img',img)
            
            #this line displays only the part of the image that has a face 
            cv2.imshow('img',roi_color)
            #joining all frames that have a face in them 
            #out.write(img)
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()






















