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
    self.prev_joint2_estimate = 0
    self.prev_joint4_estimate = 0
    self.prev_target_y_estimate = [0,0]
    self.prev_target_z_estimate = 0
    self.joint2_estimate_pub = rospy.Publisher("/robot/joint2_position_estimate", Float64, queue_size=10)
    self.joint4_estimate_pub = rospy.Publisher("/robot/joint4_position_estimate", Float64, queue_size=10)
    self.circle_chamfer = cv2.imread("chamfer_template.png", 0)
    self.target_prediction_y_pub = rospy.Publisher("/target/y_position_estimate", Float64, queue_size=10)
    self.target_prediction_z_pub = rospy.Publisher("/target/z_position_estimate", Float64, queue_size=10)
    self.target_prediction_z_sub = rospy.Subscriber("/target/z_position_estimate2", Float64, self.callbackZPos)
    self.camera2_z_estimate = 0
    self.yellow_centre = [0,0]
    self.blue_centre = [0,0]
    self.green_centre = [0,0]
    self.red_centre = [0,0]
    self.prev_joint_ys = [0,0,0,0]
    self.prev_joint_zs = [0, 0, 0, 0]
    self.joint3_sub = rospy.Subscriber("/robot/joint3_position_estimate", Float64, self.callbackJoint3)
    self.joint3_estimate = 0
    self.joint3_sub = rospy.Subscriber("/target/x_position_estimate", Float64, self.callbackTargetX)
    self.target_x_estimate = 0
    self.time_previous_step = np.array([rospy.get_time()])
    self.error = np.zeros((1, 3))
    self.end_effector_x_pub = rospy.Publisher("/robot/end_effector_x", Float64, queue_size=10)
    self.end_effector_y_pub = rospy.Publisher("/robot/end_effector_y", Float64, queue_size=10)

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
    joint2Angle = np.arctan2(self.blue_centre[0] - self.green_centre[0], self.blue_centre[1] - self.green_centre[1])
    joint4Angle = np.arctan2(self.green_centre[0] - self.red_centre[0], self.green_centre[1] - self.red_centre[1]) - joint2Angle
    if joint2Angle > np.pi/2:
      joint2Angle = np.pi/2
    elif joint2Angle < -np.pi/2:
      joint2Angle = -np.pi/2

    if joint4Angle > np.pi/2:
      joint4Angle = np.pi/2
    elif joint4Angle < -np.pi/2:
      joint4Angle = -np.pi/2

    if(np.abs(joint2Angle - self.prev_joint2_estimate) > 0.8):
      if(joint2Angle > self.prev_joint2_estimate):
        joint2Angle = self.prev_joint2_estimate + 0.1
      else:
        joint2Angle = self.prev_joint2_estimate - 0.1
    if(np.abs(joint4Angle - self.prev_joint4_estimate) > 0.8):
      if (joint4Angle > self.prev_joint4_estimate):
        joint4Angle = self.prev_joint4_estimate + 0.1
      else:
        joint4Angle = self.prev_joint4_estimate - 0.1

    return joint2Angle, joint4Angle

  def pix2Metres(self):
    return 2.5/np.sqrt((self.yellow_centre[0] - self.blue_centre[0])**2 + (self.yellow_centre[1] - self.blue_centre[1])**2)

  def getSpherePos(self, image):
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(cv2.inRange(image, (5, 50, 50), (15, 255, 255)), kernel, iterations=3)
    result = cv2.matchTemplate(mask, self.circle_chamfer, cv2.TM_CCOEFF_NORMED)
    valMin, valMax, posMin, posMax = cv2.minMaxLoc(result)
    if(valMax > 0.8):
      width, height = self.circle_chamfer.shape[::-1]
      (y,z) = (posMax[0] + int(width/2), posMax[1] + int(height/2))
      y = (y - self.yellow_centre[0]) * self.pix2Metres()
      z = (self.yellow_centre[1] - z) * self.pix2Metres()
    else:
      y = self.prev_target_y_estimate[0] + (self.prev_target_y_estimate[0] - self.prev_target_y_estimate[1])
      z = self.prev_target_z_estimate

    if(self.camera2_z_estimate == 0):
      self.camera2_z_estimate = z

    z = ((z + self.camera2_z_estimate)/2) - 0.25

    if(y < -2.5):
      y = -2.5
    elif(y > 2.5):
      y = 2.5
    if(z < 6):
      z = 6
    elif(z > 8):
      z = 8
    return (y,z)

  ################################################################################## Rongxuan's Section 3 Code
  def get_jacobian(self, j2, j3, j4):
    x = np.array([6.5 * np.sin(j3), 6.5 * np.cos(j3), 6.5 * np.sin(j3)])
    y = np.array([np.cos(j3) * (3.5 * np.cos(j2) + 3 * np.cos(j2 + j4)),
                  -np.sin(j3) * (3.5 * np.sin(j2) + 3 * np.sin(j2 + j4)), 3 * np.cos(j3) * np.cos(j2 + j4)])
    z = np.array([np.cos(j3) * (-3.5 * np.sin(j2) - 3 * np.sin(j2 + j4)),
                  -np.sin(j3) * (3.5 * np.cos(j2) + 3 * np.cos(j2 + j4)), -3 * np.cos(j3) * np.sin(j2 + j4)])
    return np.array([x, y, z])

  def forward_kinematics(self, joint2, joint3, joint4):
    L1 = 2.5
    L2 = 3.5
    L3 = 3
    bottom = np.array([0, 0, L1])
    middle = np.array(
      [L2 * np.sin(joint3), L2 * np.sin(joint2) * np.cos(joint3), L2 * np.cos(joint3) * np.cos(joint2)])
    top = np.array([L3 * np.sin(joint3), L3 * np.sin(joint2 + joint4) * np.cos(joint3),
                    L3 * np.cos(joint3) * np.cos(joint2 + joint4)])
    end_effector = bottom + middle + top
    return end_effector

  ##############################################################################################

  def callbackJoint3(self, data):
    self.joint3_estimate = data.data

  def callbackTargetX(self, data):
    self.target_x_estimate = data.data

  def callbackZPos(self, data):
    self.camera2_z_estimate = data.data

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

    if(self.yellow_centre[0] == 0):
      self.yellow_centre = self.detect_joint(self.cv_image1, "joint1template.png", 0)

    if(self.blue_centre[0] == 0):
      self.blue_centre = self.detect_joint(self.cv_image1, "joint2template.png", 1)

    self.green_centre = self.detect_joint(self.cv_image1, "joint3template.png", 2)
    self.red_centre = self.detect_joint(self.cv_image1, "joint4template.png", 3)


    if(self.green_centre[1] > self.blue_centre[1]):
      self.green_centre = (self.green_centre[0], self.blue_centre[1])

    joint2Angle, joint4Angle = self.getJointAngles()
    (y,z) = self.getSpherePos(self.cv_image1)
    self.prev_joint2_estimate = joint2Angle
    self.prev_joint4_estimate = joint4Angle
    self.prev_target_y_estimate[1] = self.prev_target_y_estimate[0]
    self.prev_target_y_estimate[0] = y
    self.prev_target_z_estimate = z

    ########################################################## Rongxuan's IK Code

    x = self.target_x_estimate
    endPos = self.forward_kinematics(joint2Angle, self.joint3_estimate, joint4Angle)

    joint3Angle = self.joint3_estimate
    jacobian = self.get_jacobian(joint2Angle, joint3Angle, joint4Angle)

    K_p = np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5]])
    K_d = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
    cur_time = np.array([rospy.get_time()])
    dt = cur_time - self.time_previous_step
    self.time_previous_step = cur_time
    pos = endPos
    pos_d = np.array([x, y, z])
    self.error_d = ((pos_d - pos) - self.error) / dt
    self.error = pos_d - pos
    q = np.array([joint2Angle, joint3Angle, joint4Angle])
    J_inv = np.linalg.pinv(jacobian)
    dq_d = np.dot(J_inv, (np.dot(K_d, self.error_d.T)) + np.dot(K_p, self.error.T))
    q_d = q + (dt * dq_d)
    # joint1 = Float64()
    # joint1 = q_d[0]
    # joint2 = Float64()
    # joint2 = q_d[1]
    # joint3 = Float64()
    # joint3 = q_d[2]
    #################################################################

    # print(joint2Angle, joint4Angle)

    # Publish the results
    try: 
      self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image1, "bgr8"))
      self.robot_joint2_pub.publish(joint2Move)
      self.robot_joint3_pub.publish(joint3Move)
      self.robot_joint4_pub.publish(joint4Move)
      # self.robot_joint2_pub.publish(q_d[0])
      # self.robot_joint3_pub.publish(q_d[1])
      # self.robot_joint4_pub.publish(q_d[2])
      self.joint2_estimate_pub.publish(joint2Angle)
      self.joint4_estimate_pub.publish(joint4Angle)
      self.target_prediction_y_pub.publish(y)
      self.target_prediction_z_pub.publish(z)
      self.end_effector_x_pub.publish(endPos[0])
      self.end_effector_y_pub.publish(endPos[1])
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


