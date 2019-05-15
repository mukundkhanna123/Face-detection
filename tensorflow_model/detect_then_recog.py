"""
This program first detects faces in a video from a live stream using a pre trained ssd then calculates the 128d face embeddings from it and saves
the different faces. Some parameter tuning is required based on the camera you are using. 


Also create two directories where this program is stored-
1.   raw_data
2.   Unique_data
""""



#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101
import time
import numpy as np
import tensorflow as tf
import cv2
import os
import pickle
from imutils import paths
import face_recognition
from encode_faces import check_encodings

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
    iter_no = 1
    while (True):
        img_no = 0
        print("encodings and clustering done")        
        tDetector = TensoflowFaceDector(PATH_TO_CKPT)
        #rtsp://admin:Flora123@192.168.1.184:554/Streaming/Channels/101
        cap = cv2.VideoCapture('rtsp://admin:Flora123@192.168.1.184:554/Streaming/Channels/401')
        windowNotSet = True
        
        #set capture duration (seconds)
        cap_duration = 30

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
                    cropped = copy_image[ymin-10:ymax+10, xmin-10:xmax+10]
                    
                    cv2.imwrite('C:/Users/Mukund/Downloads/Models/tensorflow_model/raw_data/image' + str(img_no) + '.jpg', cropped)
                    img_no += 1     
                
                    
                if windowNotSet is True:
                    cv2.namedWindow("tensorflow based (%d, %d)" % (w, h), cv2.WINDOW_NORMAL)
                    windowNotSet = False
                    
                cv2.imshow("tensorflow based (%d, %d)" % (w, h), image)
                k = cv2.waitKey(1) & 0xff
                if k == ord('q') or k == 27:
                    break
        cap.release()
        cv2.destroyAllWindows()
        print("total images stored ",img_no) 
        im_saved = check_encodings(iter_no)
        iter_no += 1
        
        

    
    
