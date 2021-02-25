#!/usr/bin/env python3

import rospy
import message_filters
import cv2
import os
import random

from sensor_msgs.msg import Image
from beleg.msg import IntWithHeader
from cv_bridge import CvBridge

cv_b = CvBridge()
current_folder = os.path.dirname(os.path.abspath(__file__))

def talker():
    pub_int = rospy.Publisher("video_stream/value_msgs", IntWithHeader, queue_size=1)
    pub_imag = rospy.Publisher('video_stream/img_msgs', Image, queue_size=1)
    rospy.init_node('cam', anonymous=True)
    rate = rospy.Rate(10)
    
    while not rospy.is_shutdown():
        a = random.randint(0, 9)
        img = cv2.imread(current_folder + f'/{a}.png')
        img_msg = cv_b.cv2_to_imgmsg(img)

        value = IntWithHeader()
        value.data = a

        pub_int.publish(value)
        pub_imag.publish(img_msg)
        rate.sleep()

if __name__ == '__main__':
   try:
       talker()
   except rospy.ROSInterruptException:
       pass        
