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
    """[summary]
        initializes a single node, able to send a custom msg (consisting of head and data; head contains the image, data contains the value)
    """
    # init of publisher that send message (IntWithHeader, Image)
    pub_int = rospy.Publisher("video_stream/value_msgs", IntWithHeader, queue_size=1)
    pub_imag = rospy.Publisher('video_stream/img_msgs', Image, queue_size=1)

    rospy.init_node('cam', anonymous=True)
    # performance option
    rate = rospy.Rate(10)
    
    while not rospy.is_shutdown():
        randomly_picked_number = random.randint(0, 9)
        img = cv2.imread(current_folder + f'/{randomly_picked_number}.png')
       
        # converting image to img_msg
        img_msg = cv_b.cv2_to_imgmsg(img)

        value = IntWithHeader()
        # assigning the value associated to image to the IntWithHeader mesage
        value.data = randomly_picked_number

        # sending the messages
        pub_int.publish(value)
        pub_imag.publish(img_msg)

        #wait for the duration of 'rate' ms for improved performance
        rate.sleep()

if __name__ == '__main__':
   try:
       talker()
   except rospy.ROSInterruptException:
       pass        
