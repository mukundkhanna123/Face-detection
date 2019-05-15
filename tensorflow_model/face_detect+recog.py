"""
This program  detects faces in a video from a live stream using a pre trained ssd then simulataneously calculates
the 128d face embeddings from it and saves 
the different faces. Some parameter tuning is required based on the camera you are using. 


Also create this directories where this program is stored-
1.   raw_data

"""
#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101
import math
import time
import numpy as np
import tensorflow as tf
import cv2
import face_recognition
import os
import pickle
from imutils import paths
import dlib
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import matplotlib.pyplot as plt


from utils import label_map_util
from utils import visualization_utils_color as vis_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './protos/face_label_map.pbtxt'

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

class TensoflowFaceDector(object):

    def __init__(self, PATH_TO_CKPT):
        """Tensorflow detector
        """

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')


        with self.detection_graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=self.detection_graph, config=config)
            self.windowNotSet = True


    def run(self, image):
        """image: bgr image
        return (boxes, scores, classes, num_detections)
        """

        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        elapsed_time = time.time() - start_time
        print('inference time cost: {}'.format(elapsed_time))

        return (boxes, scores, classes, num_detections)


if __name__ == "__main__":

    data = []
    imagePaths = list(paths.list_images("C:/Users/Mukund/Downloads/Models/tensorflow_model/raw_data"))
    img_no = 0
    predictor = dlib.shape_predictor("C:/Users/Mukund/Downloads/Models/tensorflow_model/shape_predictor_68_face_landmarks.dat")
    fa = FaceAligner(predictor, desiredFaceWidth=256)
    
    def  check(encod,imagePath, box,data,Id,param=0.6):
        # low param means more faces detected
        if os.path.exists("encodings.pickle"):
            with open("encodings.pickle","rb") as f:
                data = pickle.load(f)

            encodings = np.array([d["encoding"] for d in data])

            encod = np.squeeze(np.array(encod),axis = 0)
            for i in encodings:
                if np.linalg.norm(i - encod)< param:
                   return False
            print("new person found")
            d = [{"imagePath": imagePath, "loc": box, "encoding": encod,"UniqueId":Id}]
            data.extend(d)
            with open("encodings.pickle", "wb") as f:
                pickle.dump(data,f)
            return True
        else:
            d = [{"imagePath": imagePath, "loc": box, "encoding": encod,"UniqueId":Id}]
            data.extend(d)
            with open("encodings.pickle", "wb") as f:
                pickle.dump(data,f)    
            return True

    while (True):
                
        tDetector = TensoflowFaceDector(PATH_TO_CKPT)
        #rtsp://admin:Flora123@192.168.1.184:554/Streaming/Channels/401
        cap = cv2.VideoCapture('rtsp://admin:Flora123@192.168.1.184:554/Streaming/Channels/401')
        
        windowNotSet = True
        
        #set capture duration (seconds)
        cap_duration = 5000

        start_time = time.time()
        cap_times = [ i for i in range(0,cap_duration)] 
        while (int(time.time() - start_time) < cap_duration ):
        
            ret, image = cap.read()
            if ret == False:
                break
            a= time.time()
            copy_image = image
            if round(a - start_time,1) in cap_times :
                
                [h, w] = image.shape[:2]
                (boxes, scores, classes, num_detections) = tDetector.run(image)
                
                total_boxes=vis_util.visualize_boxes_and_labels_on_image_array(image,np.squeeze(boxes),np.squeeze(classes).astype(np.int32),np.squeeze(scores),category_index,use_normalized_coordinates=True,line_thickness=4)
            
                for box in total_boxes:
                    ymin = int(box[0] * h)
                    xmin = int(box[1] * w)
                    ymax = int(box[2] * h)
                    xmax = int(box[3] * w)
                    cropped = copy_image[ymin:ymax, xmin:xmax]

                    rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                    
                    box_new = face_recognition.face_locations(rgb,number_of_times_to_upsample=2,model = "hog")                                        
                    #box_new = [[ymin,xmax,ymax,xmin]]
                    
                    if len(box_new) == 0:
                        print("box not found")
                        continue
                    else:
                        #rgb_aligned = fa.align(cropped,cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) , dlib.rectangle(box_new[0][3],box_new[0][0],box_new[0][1],box_new[0][2]))
                        #pas= [rgb,rgb_aligned]
                        #for i in range(0,2):
                        encod = face_recognition.face_encodings(rgb, box_new)
                        imagePath = 'C:/Users/Mukund/Downloads/Models/tensorflow_model/raw_data/image' + str(img_no) + '.jpg'
                        bol_val = check(encod,imagePath,box,data,img_no)
                        if bol_val == True:
                            img_no += 1
                            cv2.imwrite(imagePath,cropped)
                        else:
                            print("face exists!")
                    
                if windowNotSet is True:
                    cv2.namedWindow("tensorflow based (%d, %d)" % (w, h), cv2.WINDOW_NORMAL)
                    windowNotSet = False
                    
                cv2.imshow("tensorflow based (%d, %d)" % (w, h), image)
                k = cv2.waitKey(1) & 0xff
                if k == ord('q') or k == 27:
                    break
        cap.release()
        cv2.destroyAllWindows()
        

    
    
