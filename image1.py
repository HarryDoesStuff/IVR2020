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
    # initialize a publisher to send images from camera1 to a topic named image_topic1
    self.image_pub1 = rospy.Publisher("image_topic1",Image, queue_size = 1)
    # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
    self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw",Image,self.callback1)
    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()
    self.start_time = rospy.get_time()
    self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size = 10)
    self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
    self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)
    self.prev_yellow = np.array([0,0])
    self.prev_blue = np.array([0,0])
    self.prev_green = np.array([0,0])
    self.prev_red = np.array([0,0])
    self.prev_joint2_estimate = 0
    self.prev_joint4_estimate = 0
    self.joint2_estimate_pub = rospy.Publisher("/robot/joint2_position_estimate", Float64, queue_size=10)
    self.joint4_estimate_pub = rospy.Publisher("/robot/joint4_position_estimate", Float64, queue_size=10)

  def detect_colour(self, image, low, high):
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(cv2.inRange(image, low, high), kernel, iterations=3)
    moments = cv2.moments(mask)
    if(moments['m00'] != 0):
      centreY = int(moments['m10'] / moments['m00'])
      centreZ = int(moments['m01'] / moments['m00'])
    else:
      if(low == (0,100,100)):
        centreY = self.prev_yellow[0]
        centreZ = self.prev_yellow[1]
      elif(low == (100, 0, 0)):
        centreY = self.prev_blue[0]
        centreZ = self.prev_blue[1]
      elif(low == (0, 100, 0)):
        centreY = self.prev_green[0]
        centreZ = self.prev_green[1]
      else:
        centreY = self.prev_red[0]
        centreZ = self.prev_red[1]
    return np.array([centreY, centreZ])

  def getJointAngles(self, blue, green, red):
    joint2Angle = np.arctan2(blue[0] - green[0], blue[1] - green[1])
    joint4Angle = np.arctan2(green[0] - red[0], green[1] - red[1]) - joint2Angle
    if joint2Angle > np.pi/2:
      joint2Angle = np.pi/2
    elif joint2Angle < -np.pi/2:
      joint2Angle = -np.pi/2

    if joint4Angle > np.pi/2:
      joint4Angle = np.pi/2
    elif joint4Angle < -np.pi/2:
      joint4Angle = -np.pi/2

    if(np.abs(joint2Angle - self.prev_joint2_estimate) > 0.5):
      if(joint2Angle > self.prev_joint2_estimate):
        joint2Angle = self.prev_joint2_estimate + 0.05
      else:
        joint2Angle = self.prev_joint2_estimate - 0.05
    if(np.abs(joint4Angle - self.prev_joint4_estimate) > 0.5):
      if (joint4Angle > self.prev_joint4_estimate):
        joint4Angle = self.prev_joint4_estimate + 0.05
      else:
        joint4Angle = self.prev_joint4_estimate - 0.05

    return joint2Angle, joint4Angle
  # Recieve data from camera 1, process it, and publish
  def callback1(self,data):
    # Recieve the image
    try:
      self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
      # joint3Estimate = data[1]
    except CvBridgeError as e:
      print(e)
    
    # Uncomment if you want to save the image
    #cv2.imwrite('image_copy.png', cv_image)

    im1=cv2.imshow('window1', self.cv_image1)
    cv2.waitKey(1)

    # Move the joints
    time = rospy.get_time() - self.start_time
    joint2Move = (np.pi/2) * np.sin((np.pi/15) * time)
    joint3Move = (np.pi / 2) * np.sin((np.pi / 18) * time)
    joint4Move = (np.pi / 2) * np.sin((np.pi / 20) * time)

    yellow = self.detect_colour(self.cv_image1, (0,100,100), (0,255,255))
    blue = self.detect_colour(self.cv_image1, (100, 0, 0), (255, 0, 0))
    green = self.detect_colour(self.cv_image1, (0, 100, 0), (0, 255, 0))
    red = self.detect_colour(self.cv_image1, (0, 0, 100), (0, 0, 255))

    if(blue[1] < green[1]):
      green[1] = blue[1]

    self.prev_yellow = yellow
    self.prev_blue = blue
    self.prev_green = green
    self.prev_red = red
    # print(yellow, blue, green, red)

    # if joint3Estimate > np.pi/4:
    #   joint3Angle = self.getJointAngles(blue, green, red, 'y')
    #   self.joint3_estimate_pub.publish(joint3Angle)
    # else:
    joint2Angle, joint4Angle = self.getJointAngles(blue, green, red)
    self.prev_joint2_estimate = joint2Angle
    self.prev_joint4_estimate = joint4Angle


    # print(joint2Angle, joint4Angle)

    # Publish the results
    try: 
      self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image1, "bgr8"))
      self.robot_joint2_pub.publish(joint2Move)
      self.robot_joint3_pub.publish(joint3Move)
      self.robot_joint4_pub.publish(joint4Move)
      self.joint2_estimate_pub.publish(joint2Angle)
      self.joint4_estimate_pub.publish(joint4Angle)
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


