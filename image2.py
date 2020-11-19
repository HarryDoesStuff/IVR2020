#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError


class image_converter:

  # Defines publisher and subscriber
  def __init__(self):
    # initialize the node named image_processing
    rospy.init_node('image_processing', anonymous=True)
    # initialize a publisher to send images from camera2 to a topic named image_topic2
    self.image_pub2 = rospy.Publisher("image_topic2",Image, queue_size = 1)
    # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
    self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw",Image,self.callback2)
    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()
    self.prev_yellow = np.array([0,0])
    self.prev_blue = np.array([0,0])
    self.prev_green = np.array([0,0])
    self.prev_red = np.array([0,0])
    self.prev_joint3_estimate = 0
    self.prev_target_x_estimate = [0,0]
    self.prev_target_z_estimate = 0
    self.circle_chamfer = cv2.imread("chamfer_template.png", 0)
    self.joint3_estimate_pub = rospy.Publisher("/robot/joint3_position_estimate", Float64, queue_size=10)
    self.target_prediction_x_pub = rospy.Publisher("/target/x_position_estimate", Float64, queue_size=10)
    self.target_prediction_z_pub = rospy.Publisher("/target/z_position_estimate2", Float64, queue_size=10)

  def detect_colour(self, image, low, high):
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(cv2.inRange(image, low, high), kernel, iterations=3)
    moments = cv2.moments(mask)
    if(moments['m00'] != 0):
      centreX = int(moments['m10'] / moments['m00'])
      centreZ = int(moments['m01'] / moments['m00'])
    else:
      if(low == (0,100,100)):
        centreX = self.prev_yellow[0]
        centreZ = self.prev_yellow[1]
      elif(low == (100, 0, 0)):
        centreX = self.prev_blue[0]
        centreZ = self.prev_blue[1]
      elif(low == (0, 100, 0)):
        centreX = self.prev_green[0]
        centreZ = self.prev_green[1]
      else:
        centreX = self.prev_red[0]
        centreZ = self.prev_red[1]
    return np.array([centreX, centreZ])


  def getJointAngles(self, blue, green):
    joint3Angle = -np.arctan2(blue[0] - green[0], blue[1] - green[1])
    if joint3Angle > np.pi/2:
      joint3Angle = np.pi/2
    elif joint3Angle < -np.pi/2:
      joint3Angle = -np.pi/2

    if (np.abs(joint3Angle - self.prev_joint3_estimate) > 1):
      if (joint3Angle > self.prev_joint3_estimate):
        joint3Angle = self.prev_joint3_estimate + 0.08
      else:
        joint3Angle = self.prev_joint3_estimate - 0.08

    return joint3Angle

  def pix2Metres(self, yellow, blue):
    return 2.5/np.linalg.norm(yellow-blue)

  def getSpherePos(self, image, yellow, blue):
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(cv2.inRange(image, (5, 50, 50), (15, 255, 255)), kernel, iterations=3)
    result = cv2.matchTemplate(mask, self.circle_chamfer, cv2.TM_CCOEFF_NORMED)
    valMin, valMax, posMin, posMax = cv2.minMaxLoc(result)
    if(valMax > 0.78):
      width, height = self.circle_chamfer.shape[::-1]
      (x,z) = (posMax[0] + int(width/2), posMax[1] + int(height/2))
      x = (x - yellow[0]) * self.pix2Metres(yellow, blue)
      z = (yellow[1] - z) * self.pix2Metres(yellow, blue)
    else:
      x = self.prev_target_x_estimate[0] + (self.prev_target_x_estimate[0] - self.prev_target_x_estimate[1])
      z = self.prev_target_z_estimate

    if(x < -2):
      x = -2
    elif(x > 3):
      x = 3

    return (x,z)

  # Recieve data, process it, and publish
  def callback2(self,data):
    # Recieve the image
    try:
      self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    # Uncomment if you want to save the image
    #cv2.imwrite('image_copy.png', cv_image)
    im2=cv2.imshow('window2', self.cv_image2)
    cv2.waitKey(1)

    yellow = self.detect_colour(self.cv_image2, (0,100,100), (0,255,255))
    blue = self.detect_colour(self.cv_image2, (100, 0, 0), (255, 0, 0))
    green = self.detect_colour(self.cv_image2, (0, 100, 0), (0, 255, 0))
    red = self.detect_colour(self.cv_image2, (0, 0, 100), (0, 0, 255))

    if(blue[1] < green[1]):
      green[1] = blue[1]

    self.prev_yellow = yellow
    self.prev_blue = blue
    self.prev_green = green
    self.prev_red = red
    # print(yellow, blue, green, red)

    joint3Angle = self.getJointAngles(blue, green)
    self.prev_joint3_estimate = joint3Angle

    (x,z) = self.getSpherePos(self.cv_image2, yellow, blue)
    self.prev_target_x_estimate[1] = self.prev_target_x_estimate[0]
    self.prev_target_x_estimate[0] = x
    self.prev_target_z_estimate = z

    # Publish the results
    try: 
      self.image_pub2.publish(self.bridge.cv2_to_imgmsg(self.cv_image2, "bgr8"))
      self.joint3_estimate_pub.publish(joint3Angle)
      self.target_prediction_x_pub.publish(x)
      self.target_prediction_z_pub.publish(z)
    except CvBridgeError as e:
      print(e)

# call the class
def main(args):
  ic = image_converter()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)


