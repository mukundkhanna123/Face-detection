
# Import the necessary libraries
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import os 

# loading the classifiers 

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
nose_cascade = cv2.CascadeClassifier('nose.xml')
rear_cascade = cv2.CascadeClassifier('right_ear.xml')
lear_cascade = cv2.CascadeClassifier('left_ear.xml')
fullbody_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

cap = cv2.VideoCapture('sample1.mp4')

while 1:
    ret, img = cap.read()
    
    if ret == True:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    
        for (x,y,w,h) in faces:
            
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            
            
            #this line displays only the part of the image that has a face 
            #cv2.imshow('img',roi_color)

            fb = fullbody_cascade.detectMultiScale(roi_gray)
            for (fx,fy,fw,fh) in fb:
                cv2.rectangle(roi_color,(fx,fy),(fx+fw,fy+fh),(0,255,0),2)  

            nose = nose_cascade.detectMultiScale(roi_gray)
            for (nx,ny,nw,nh) in nose:
                cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(0,255,0),2) 

            smile = smile_cascade.detectMultiScale(roi_gray)
            for (mx,my,mw,mh) in smile:
                cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(0,255,0),2)   

            lear = lear_cascade.detectMultiScale(roi_gray)
            for (lx,ly,lw,lh) in lear:
                cv2.rectangle(roi_color,(lx,ly),(lx+lw,ly+lh),(0,255,0),2)

            rear = rear_cascade.detectMultiScale(roi_gray)
            for (rx,ry,rw,rh) in rear:
                cv2.rectangle(roi_color,(rx,ry),(rx+rw,ry+rh),(0,255,0),2)
                
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
            #to play the full video with bounding box uncomment the next line
        cv2.imshow('img',img)
            
            
            
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()






















