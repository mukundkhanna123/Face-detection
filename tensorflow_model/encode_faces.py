"""
This program takes as input closely cropped images and calculates their face embeddings
It uniquely identifies and stores the face and deletes similar faces. 
the path to the unique and raw images should be changed accordingly 


"""
import numpy as np
import cv2
import os
from imutils import paths
import face_recognition
import pickle
from datetime import datetime
import math
import matplotlib.pyplot as plt


def check_person(image,param,encod,encodings,data):
    mini = param
    prev_person = 0
    code = 0
    for no,i in enumerate(encodings):
        x = np.linalg.norm(i - encod)
        if x< mini:
            mini = x
            code = 1
            prev_person = no
    if code == 1:
        if mini > 0.42:
            plt.imshow(image)
            plt.show()
        print("person already exists with unique id :", data[no]["UniqueId"])
    return code
def adjust_gamma(image):
    intensity = np.mean(image)
    if intensity >= 127.5:
        r = -0.1
    else:
        r = 0.1
    invGamma = intensity/127.5 + r
    table = np.array([((i/255.0) ** invGamma) * 255 for i in np.arange(0,256)]).astype("uint8")
    return cv2.LUT(image,table)


def edge_enhance(img):
    
    kernel = np.array([[-1,-1,-1,-1,-1],
                               [-1,2,2,2,-1],
                               [-1,2,8,2,-1],
                               [-2,2,2,2,-1],
                               [-1,-1,-1,-1,-1]])/8.0
    output = cv2.filter2D(output, -1, kernel)
    output = adjust_gamma(img)
    return output


def check_hours(image,img_no,iter_no,d,output_dir):

    curr_hour = math.ceil(d[0]["time"])

    if os.path.exists(output_dir + "/hour_no-" + str(curr_hour)):

        cv2.imwrite(output_dir +"/hour_no-" + str(curr_hour) + "/image" + str(iter_no)+ "_" + str(img_no) + ".jpg",image) 

    else:

        os.makedirs(output_dir +"/hour_no-" + str(curr_hour))
        cv2.imwrite(output_dir +"/hour_no-" + str(curr_hour) + "/image" + str(iter_no)+ "_" + str(img_no) + ".jpg",image)     


    return output_dir +str(curr_hour)+ "/image" + str(iter_no)+ "_" + str(img_no) + ".jpg"




def check_encodings(image_dir, output_dir, iter_no,time_data,param= 0.485):
    img_no = 0
    data = []
    filenames = os.listdir(image_dir)
    imagePaths = list(paths.list_images(image_dir)) 
    if os.path.exists("encodings.pickle"):
        with open("encodings.pickle","rb")as f:
            data = pickle.load(f)

    check = False
    
    for  (file,imagepath) in zip(filenames, imagePaths):

        if os.stat(image_dir + "/"+ file).st_size == 0:
            os.remove(imagepath)
            continue

        i = int(file[0:-4])
        image = cv2.imread(imagepath)

        #image = edge_enhance(image)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #remove extremely blurry images 
        fm = cv2.Laplacian(image, cv2.CV_64F).var()
        if fm < 80:
            continue
        start_time = time_data[i]  
        os.remove(imagepath)
        
        boxes = face_recognition.face_locations(rgb,model="cnn")

        if len(boxes) == 0:
            print("Unable to find Facial features. Image Deleted")
            continue

        else:

            encod = np.array(face_recognition.face_encodings(rgb, boxes,num_jitters = 20))
            # low param means more faces detected
            if encod.shape[0] == 1:
                encod = np.squeeze(encod,axis = 0)
            if len(data) != 0:

                encodings = [d["encoding"] for d in data]

                code= check_person(rgb,param,encod,encodings,data)

                if code == 1:
                    continue
            
                d = [{"imagePath": output_dir + "/image" + str(iter_no)+ "_" + str(img_no) + ".jpg", "loc": boxes, "encoding": encod,"UniqueId":str(iter_no) + " " + str(img_no),"time":start_time}]
                print(img_no," person of ", iter_no ,"  video found")
                d[0]["imagePath"] = check_hours(image,img_no,iter_no,d,output_dir)
                data.extend(d)
                img_no += 1

            else:
                
                d = [{"imagePath": output_dir + "/image" + str(iter_no)+ "_" + str(img_no) + ".jpg", "loc": boxes  , "encoding": encod,"UniqueId":str(iter_no) + " " + str(img_no),"time":start_time}]
                print("0th person of 1st video found")
                d[0]["imagePath"] = check_hours(image,img_no,iter_no,d,output_dir)
                data.extend(d)
                img_no += 1
    with open("encodings.pickle", "wb") as f:
        pickle.dump(data,f)
            
    return img_no

