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
    self.prev_joint3_estimate = 0
    self.prev_target_x_estimate = [0,0]
    self.prev_target_z_estimate = 0
    self.circle_chamfer = cv2.imread("chamfer_template.png", 0)
    self.joint3_estimate_pub = rospy.Publisher("/robot/joint3_position_estimate", Float64, queue_size=10)
    self.target_prediction_x_pub = rospy.Publisher("/target/x_position_estimate", Float64, queue_size=10)
    self.target_prediction_z_pub = rospy.Publisher("/target/z_position_estimate2", Float64, queue_size=10)
    self.yellow_centre = [0,0]
    self.blue_centre = [0,0]
    self.green_centre = [0,0]
    self.red_centre = [0,0]
    self.prev_joint_ys = [0,0,0,0]
    self.prev_joint_zs = [0, 0, 0, 0]

  def detect_joint(self, image, tmp, jointNo):
    # kernel = np.ones((5,5), np.uint8)
    mask = cv2.inRange(image, (0, 0, 0), (60, 255, 20))
    template = cv2.imread(tmp, 0)
    rows, cols = template.shape
    maxList = []
    posList = []

    for angle in range(0, 360, 45):
      rotMatrix = cv2.getRotationMatrix2D((int(cols/2),int(rows/2)), angle, 1)
      imgRotated = cv2.warpAffine(template, rotMatrix, (cols, rows))
      result = cv2.matchTemplate(mask, imgRotated, cv2.TM_CCOEFF_NORMED)
      valMin, valMax, posMin, posMax = cv2.minMaxLoc(result)
      maxList.append(valMax)
      posList.append(posMax)
    maxIndex = maxList.index(max(maxList))
    (y,z) = (posList[maxIndex][0] + int(rows/2), posList[maxIndex][1] + int(cols/2))

    if((y > self.prev_joint_ys[jointNo] + 40 or y < self.prev_joint_ys[jointNo] - 40) & (self.prev_joint_ys[jointNo] != 0)):
      y = self.prev_joint_ys[jointNo]
    if((z > self.prev_joint_zs[jointNo] + 40 or z < self.prev_joint_zs[jointNo] - 40) & (self.prev_joint_zs[jointNo] != 0)):
      z = self.prev_joint_zs[jointNo]

    return (y,z)


  def getJointAngles(self):
    joint3Angle = -np.arctan2(self.blue_centre[0] - self.green_centre[0], self.blue_centre[1] - self.green_centre[1])
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

  def pix2Metres(self):
    return 2.5/np.sqrt((self.yellow_centre[0] - self.blue_centre[0])**2 + (self.yellow_centre[1] - self.blue_centre[1])**2)

  def getSpherePos(self, image):
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(cv2.inRange(image, (5, 50, 50), (15, 255, 255)), kernel, iterations=3)
    result = cv2.matchTemplate(mask, self.circle_chamfer, cv2.TM_CCOEFF_NORMED)
    valMin, valMax, posMin, posMax = cv2.minMaxLoc(result)
    if(valMax > 0.78):
      width, height = self.circle_chamfer.shape[::-1]
      (x,z) = (posMax[0] + int(width/2), posMax[1] + int(height/2))
      x = (x - self.yellow_centre[0]) * self.pix2Metres()
      z = (self.yellow_centre[1] - z) * self.pix2Metres()
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

    if(self.yellow_centre[0] == 0):
      self.yellow_centre = self.detect_joint(self.cv_image2, "joint1template.png", 0)

    if(self.blue_centre[0] == 0):
      self.blue_centre = self.detect_joint(self.cv_image2, "joint2template.png", 1)

    self.green_centre = self.detect_joint(self.cv_image2, "joint3template.png", 2)
    self.red_centre = self.detect_joint(self.cv_image2, "joint4template.png", 3)

    if(self.green_centre[1] > self.blue_centre[1]):
      self.green_centre = (self.green_centre[0], self.blue_centre[1])

    joint3Angle = self.getJointAngles()
    self.prev_joint3_estimate = joint3Angle

    (x,z) = self.getSpherePos(self.cv_image2)
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


