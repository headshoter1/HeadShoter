# coding: utf-8
# # Object Detection Demo with modification from Sentdex and HeadShoter
# License: Apache License 2.0 (https://github.com/tensorflow/models/blob/master/LICENSE)
# source: https://github.com/tensorflow/models
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import ctypes
import win32api, win32con
import time
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from grabscreen import grab_screen
import cv2



# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


# ## Object detection imports
# Here are the imports from the object detection module.

from utils import label_map_util
from utils import visualization_utils as vis_util

# This is defenition of clicking in exact point after moving
def click(x,y):
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(x), int(y))
    time.sleep(.01)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    time.sleep(.01)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)
    
def get_position():
    """ Returns the (x, y) mouse position. """
    return win32api.GetCursorPos()



    

# # Model preparation 
# What model to download. You can use different models from 
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
# Just change model name
MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# ## Download Model
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.70)

with detection_graph.as_default():
  with tf.Session(graph=detection_graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    while True:
      #screen = cv2.resize(grab_screen(region=(0,40,1280,745)), (WIDTH,HEIGHT))
      screen = grab_screen(region=(0,40,1280,780))
      image_np = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=4)
      for i,b in enumerate(boxes[0]):
        if classes[0][i] == 1:
         if scores[0][i] >= 0.7:
           position_x = 640
           position_y = 360  
           mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
           mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
           move_x = int((boxes[0][i][1])* 1280) - position_x
           move_y = int((boxes[0][i][0])* 720) - position_y
           pos_x1 = int((boxes[0][i][1])* 1280)
           pos_y1 = int((boxes[0][i][0])* 720)
           bias_x = int(((boxes[0][i][3])-(boxes[0][i][1]))*1280/2)
           bias_y = int(((boxes[0][i][2])-(boxes[0][i][0]))*720/15)
           apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
           ##cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*600)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
           if apx_distance >= 0:
               ##move (int((boxes[0][i][1])*800)-position_x, int((boxes[0][i][0])*600)-position_y ) 
               click(int((move_x + bias_x)*0.4), int((move_y + bias_y)*0.4))
             ##  click(int((move_x + bias_x)*0.5), int((move_y + bias_y)*0.5))
               cv2.putText(image_np, 'HEADSHOT!!!1', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
            
             
             
               
      cv2.imshow('window',image_np)
      if cv2.waitKey(25) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          break
