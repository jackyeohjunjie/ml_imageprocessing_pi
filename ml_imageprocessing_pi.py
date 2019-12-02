# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import argparse
import tensorflow as tf
import sys
from matplotlib import pyplot as plt

FLAGS = None

#Title: 1000 image classifier
#Author: TensorFlow Team
#Date: Aug 26, 2017
#Availability: https://github.com/tensorflow/models/blob/master/tutorials/image/imagenet/classify_image.py

#Title: Retraining Image Classifier 
#Author: TensorFlow Team
#Date: Aug 26, 2017
#Availability: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py

#Title: Retrained Image Classifer 
#Author: Siraj
#Date: Aug 26, 2017
#Availability: https://github.com/llSourcell/tensorflow_image_classifier/blob/master/src/label_image.py

#Takes in image and returns predictions and scores for object in image
def classify_image(image):

    save_result = 0
    
    # Read in the image_data
    image_data = tf.gfile.FastGFile(image, 'rb').read()
    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line 
                       in tf.gfile.GFile("/home/pi/tf_files/retrained_labels.txt")]
    # Unpersists graph from file
    with tf.gfile.FastGFile("/home/pi/tf_files/retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        
        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})
        
        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            if save_result == 0:
                highest_string = label_lines[node_id]
                highest_score = predictions[0][node_id]
                save_result = 1
            print('%s (score = %.5f)' % (human_string, score))

        return highest_string, highest_score
    
#Title: Camera 
#Author: Dr. Adrian Rosebrock
#Date: Aug 15, 2017
#Availability: https://www.pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python/	
#Intializes Pi Cam and takes a picture when character 'c' is pressed
def run_camera():
    #initialize the camera 
    camera = PiCamera()

    #set resolution for camera
    camera.resolution = (640, 480)
    camera.framerate = 32
    rawCapture = PiRGBArray(camera,
                            size=(640, 480))
     
    # set camera to warmup
    time.sleep(0.1)
     
    #set to run in loop until c is pressed
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

            image = frame.array

            key = cv2.waitKey(1) & 0xFF

            rawCapture.truncate(0)
            cv2.imshow("Video", image)
            
            # if the `c` key was pressed, break from the loop
            if key == ord("c"):
                cv2.imwrite("captured.jpg",image)
                break

# Generates a mask for the crop by setting appropriate
# theshold values depending on crop type, biggest contour is then identified
# as crop
def process_crop(image, human_string,image_flag):

    #set flag to default
    red_flag = 0

    #set appropriate values threshold for crops
    if human_string == "broccoli":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        if image_flag == 1:
            threshold_low = np.array([30, 30, 25], dtype = "uint8")
            threshold_high = np.array([70, 255, 255], dtype = "uint8")
        else :
            threshold_low = np.array([20, 30, 30], dtype = "uint8")
            threshold_high = np.array([80, 255, 255], dtype = "uint8")

    elif human_string == "cabbage":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        if image_flag == 1:
            threshold_low = np.array([20, 40, 30], dtype = "uint8")
            threshold_high = np.array([70, 255, 255], dtype = "uint8")
        else:
            threshold_low = np.array([20, 55, 40], dtype = "uint8")
            threshold_high = np.array([70, 255, 255], dtype = "uint8")

    elif human_string == "cauliflower":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        if image_flag == 1:
            threshold_low = np.array([15, 0, 30], dtype = "uint8")
            threshold_high = np.array([30, 170, 255], dtype = "uint8")
        else:
            threshold_low = np.array([0, 0, 30], dtype = "uint8")
            threshold_high = np.array([180, 55, 255], dtype = "uint8")

    elif human_string == "red capsicum":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        red_flag = 1
        if image_flag == 1:
            threshold_low = np.array([0, 60, 75], dtype = "uint8")
            threshold_high = np.array([10, 255, 255], dtype = "uint8")
        else:
            threshold_low = np.array([0, 60, 50], dtype = "uint8")
            threshold_high = np.array([20, 255, 255], dtype = "uint8")

        
    #blur and convert to hsv
    blur = cv2.GaussianBlur(image, (15, 15), 2)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)    

    #apply threshold values
    if red_flag == 1:
        mask1 = cv2.inRange(hsv, threshold_low, threshold_high)

        #set second set of threshold values
        if image_flag == 1:
            threshold_low = np.array([170, 60, 75], dtype = "uint8")
            threshold_high = np.array([180, 255, 255], dtype = "uint8")
        else:
            threshold_low = np.array([160, 60, 50], dtype = "uint8")
            threshold_high = np.array([180, 255, 255], dtype = "uint8")

        #get final mask
        mask2 = cv2.inRange(hsv, threshold_low, threshold_high)
        mask = cv2.bitwise_or(mask1, mask2, mask = None)
        
    else:
        mask = cv2.inRange(hsv, threshold_low, threshold_high)
        
    #do some erosion
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    (_, cnts, _) = cv2.findContours(opened_mask.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)

    #create a blank image to draw on
    draw_mask = np.zeros(image.shape[:2],dtype = "uint8")

    #when we find the crop
    if len(cnts) > 0:

        #find the biggest area of the crop
        cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

        #find the area of the crop
        c = max(cnts,key = cv2.contourArea)
        ((x,y),radius) = cv2.minEnclosingCircle(c)

        
        #highlight crop on image
        cv2.drawContours(image, [cnt], -1, (255, 255, 255), 2)

        #generate mask of crop
        cv2.drawContours(draw_mask, [cnt], -1, 255, -1)

        #apply mask to image
        masked_img = cv2.bitwise_and(image, image, mask=draw_mask)
    

    return radius, masked_img, image, draw_mask, 

#Appends all results onto picture     
def append_text(image,human_string,size,crop_score,health_state,correl_score):

    if human_string == "broccoli":
        cv2.putText(image,"Crop: Broccoli",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)

    elif human_string == "cabbage":
        cv2.putText(image,"Crop: Cabbage",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)

    elif human_string == "cauliflower":
        cv2.putText(image,"Crop: Cauliflower",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)

    else:
        cv2.putText(image,"Crop: Red Capsicum",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)

    cv2.putText(image,"Confidence: " + str(int(crop_score * 100)) + "/100",(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)

    cv2.putText(image,"Size: " + str(int(size)),(10,70),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)

    if health_state == 1:
        cv2.putText(image,"Health - PASS ",(10,90),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
    elif health_state == 2:
        cv2.putText(image,"Health - FAIL ",(10,90),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)

    cv2.putText(image,"Correl: " + str(int(correl_score * 100)) + "/100",(10,110),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),2)

    print (human_string, crop_score*100,size,health_state,correl_score*100) 
    return image

#Generates mask for diseased pixels and performs an AND operation with the crop mask
def health_check(masked_img,human_string,draw_mask,image_flag,image):

    health_state = 0

    # set threshold values for things we are going to look for in the crop
    if human_string == "broccoli":
        if image_flag == 1:
            threshold_low = np.array([0, 0, 0], dtype = "uint8")
            threshold_high = np.array([26, 150, 150], dtype = "uint8")
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        else:
            threshold_low = np.array([0, 0, 0], dtype = "uint8")
            threshold_high = np.array([20, 150, 150], dtype = "uint8")
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    elif human_string == "cauliflower":
        if image_flag == 1:
            threshold_low = np.array([0, 0, 0], dtype = "uint8")
            threshold_high = np.array([20, 150, 150], dtype = "uint8")
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        else:
            threshold_low = np.array([0, 0, 0], dtype = "uint8")
            threshold_high = np.array([20, 150, 150], dtype = "uint8")
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    elif human_string == "cabbage":
        if image_flag == 1:
            threshold_low = np.array([0, 0, 0], dtype = "uint8")
            threshold_high = np.array([20, 255, 150], dtype = "uint8")
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))         
        else: 
            threshold_low = np.array([0, 0, 0], dtype = "uint8")
            threshold_high = np.array([20, 150, 150], dtype = "uint8")
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))

    elif human_string == "red capsicum":
        if image_flag == 1:
            threshold_low = np.array([0, 0, 0], dtype = "uint8")
            threshold_high = np.array([180, 255, 80], dtype = "uint8")
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        else:
            threshold_low = np.array([50, 0, 0], dtype = "uint8")
            threshold_high = np.array([70, 255, 150], dtype = "uint8")
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)) 

    blur_masked = cv2.GaussianBlur(masked_img, (15, 15), 2)
    hsv_masked = cv2.cvtColor(blur_masked, cv2.COLOR_BGR2HSV)
    detect_mask = cv2.inRange(hsv_masked, threshold_low, threshold_high)


    # do the AND operation to check for black spots within the crop
    blackDetect = cv2.bitwise_and(detect_mask, draw_mask)
    blackDetect = cv2.morphologyEx(blackDetect, cv2.MORPH_OPEN, kernel)

    #check if we can find any contours
    (_, cnts2, _) = cv2.findContours(blackDetect.copy(), cv2.RETR_EXTERNAL,
              cv2.CHAIN_APPROX_SIMPLE)

    #if yes, then draw a circle around it and state unhealthy
    if len(cnts2) > 0:
        cv2.drawContours(image, cnts2, -1, (0, 0, 255), 2)

        health_state = 2
    else:
        health_state = 1

    return image, health_state

#Title: Plotting histogram source code
#Author: Dr. Adrian Rosebrock
#Date: Oct 17, 2017
#Availability: Practical Python and OpenCV, 3rd Edition, Page 75
def plot_histogram(image, title, mask = None):

	chans = cv2.split(image)
	colors = ("b", "g", "r")
	plt.figure()
	plt.title(title)
	plt.xlabel("Bins")
	plt.ylabel("# of Pixels")

	# Loop over the image channels
	for (chan, color) in zip(chans, colors):
		hist = cv2.calcHist([chan], [0], mask, [256], [0, 256])
		plt.plot(hist, color = color)
		plt.xlim([0, 256])

#Compares Histogram of input image and reference Image	
def mature_check(image,human_string,draw_mask):

    red_flag = 0

    if human_string == "broccoli":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        threshold_low = np.array([20, 30, 20], dtype = "uint8")
        threshold_high = np.array([80, 255, 255], dtype = "uint8")
        reference_img = cv2.imread("reference/reference1.jpg")
    elif human_string == "cabbage":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        threshold_low = np.array([20, 40, 30], dtype = "uint8")
        threshold_high = np.array([70, 255, 255], dtype = "uint8")
        reference_img = cv2.imread("reference/reference2.jpg")
    elif human_string == "cauliflower":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        threshold_low = np.array([0, 0, 30], dtype = "uint8")
        threshold_high = np.array([180, 50, 255], dtype = "uint8")
        reference_img = cv2.imread("reference/reference3.jpg")
    elif human_string == "red capsicum":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        threshold_low = np.array([0, 60, 75], dtype = "uint8")
        threshold_high = np.array([10, 255, 255], dtype = "uint8")
        reference_img = cv2.imread("reference/reference4.jpg")
        red_flag = 1
        
    #process the reference image
    reference_blur = cv2.GaussianBlur(reference_img, (15, 15), 2)
    reference_hsv = cv2.cvtColor(reference_blur, cv2.COLOR_BGR2HSV)

    if red_flag == 1:
        mask1 = cv2.inRange(reference_hsv, threshold_low, threshold_high)

        #set second set of threshold values 
        threshold_low = np.array([170, 60, 75], dtype = "uint8")
        threshold_high = np.array([180, 255, 255], dtype = "uint8")

        #get final mask
        mask2 = cv2.inRange(reference_hsv, threshold_low, threshold_high)
        reference_mask = cv2.bitwise_or(mask1, mask2, mask = None) 
        
    else:    
        reference_mask = cv2.inRange(reference_hsv, threshold_low, threshold_high)

    reference_mask = cv2.morphologyEx(reference_mask, cv2.MORPH_OPEN, kernel)
    reference_masked = cv2.bitwise_and(reference_img,reference_img, mask = reference_mask)

    #set the mask for input image
    input_masked = cv2.bitwise_and(image,image, mask = draw_mask)

    hist1 = cv2.calcHist([image], [0,1], draw_mask, [180,256], [0,180, 0,256])
    hist2 = cv2.calcHist([reference_img], [0,1], reference_mask, [180,256], [0,180, 0,256])

    val = cv2.compareHist(hist1,hist2,cv2.HISTCMP_CORREL)

    plot_histogram(reference_img, "Histogram for Reference Image", mask = reference_mask)
    plot_histogram(image, "Histogram for Input Image", mask = draw_mask)
 
    if val < 0:
        val = 0

    return val

    cv2.waitKey(0) 
        
    
def main(_):

    image_flag = 0

    if FLAGS.image_file:
        image = (FLAGS.image_file)
        string, score = classify_image(image)
        cv_image = cv2.imread(image)
        unedited_img = cv2.imread(image)
        image_flag = 1
    else:
        run_camera()
        string, score = classify_image("captured.jpg")
        cv_image = cv2.imread("captured.jpg")
        unedited_img = cv2.imread("captured.jpg")
    
    size, masked_img, contour_img, draw_mask = process_crop(cv_image, string,image_flag)

    marked_img,health_state = health_check(masked_img,string,draw_mask,image_flag,contour_img)

    correl_score = mature_check(unedited_img,string,draw_mask)

    final_image = append_text(marked_img,string, size, score, health_state,correl_score)


    cv2.destroyAllWindows()
    cv2.imshow("Final Image",final_image)
    cv2.imwrite("results.jpg",final_image)

    plt.show()
    cv2.waitKey(0) 

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # indicate if argument
  parser.add_argument(
      '--image_file',
      type=str,
      default='',
      help='Absolute path to image file.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    

