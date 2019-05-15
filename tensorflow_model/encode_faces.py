import numpy as np
import cv2
import os
from imutils import paths
import face_recognition
import pickle

def check_encodings(iter_no,param= 0.5):
    imagePaths = list(paths.list_images("C:/Users/Mukund/Downloads/Models/tensorflow_model/raw_data"))
    img_no = 0
    code = 0
    data = []
    if os.path.exists("encodings.pickle"):
        with open("encodings.pickle","rb")as f:
            data = pickle.load(f)
    check = False
    for  (i,imagePath) in enumerate(imagePaths):
        image = cv2.imread(imagePath) 
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        os.remove(imagePath)
        boxes = face_recognition.face_locations(rgb,number_of_times_to_upsample=2,model="cnn")
        
        if len(boxes) == 0:
            print("Unable to find Facial features. Image Deleted")
            continue
        else:
            encod = face_recognition.face_encodings(rgb, boxes)
            # low param means more faces detected
            if len(data) != 0:
                encodings = [d["encoding"] for d in data]
                encod = np.squeeze(np.array(encod),axis = 0)
                for no,i in enumerate(encodings):
                    x = np.linalg.norm(i - encod)
                    if x< param:
                        print(x)    
                        code = 1
                        print("person already exists with unique id :", data[no]["UniqueId"])
                        break
                if code == 1:
                    continue
                print(code)
                d = [{"imagePath": imagePath, "loc": boxes, "encoding": encod,"UniqueId":str(iter_no) + " " + str(img_no)}]
                print(img_no," person of ", iter_no ,"  iteration found")
                img_no += 1
                data.extend(d)
                cv2.imwrite("C:/Users/Mukund/Downloads/Models/tensorflow_model/unique_data/image" + str(iter_no)+ "_" + str(img_no) + ".jpg",image) 
            else:
                print("first iteration")
                d = [{"imagePath": imagePath, "loc": boxes  , "encoding": encod,"UniqueId":str(iter_no) + " " + str(img_no)}]
                print("0th person of 1st  iteration found")
                cv2.imwrite("C:/Users/Mukund/Downloads/Models/tensorflow_model/unique_data/image" + str(iter_no)+ "_" + str(img_no) + ".jpg",image)
                img_no += 1
                data.extend(d)
    
    with open("encodings.pickle", "wb") as f:
        pickle.dump(data,f)
            
    return img_no
