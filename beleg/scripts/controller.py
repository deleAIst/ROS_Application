#!/usr/bin/env python3

import rospy
import message_filters

from sensor_msgs.msg import Image
from beleg.msg import IntWithHeader
from cv_bridge import CvBridge
from beleg.srv import AI

cv_b = CvBridge()

stored_input = []

def callback(img_msg, int_with_header):
    rospy.wait_for_service('ai_server')
    try:
        ai_servise = rospy.ServiceProxy('ai_server', AI)
        result =ai_servise(img_msg)
        if(result.result == int_with_header.data):
            rospy.loginfo("True")
        else:
            rospy.loginfo("False")

        #rospy.loginfo(int_with_header)
        #rospy.loginfo(red)
    except rospy.ServiceException as e:
        print("Service call failed: %s" %e)




def controller():
    rospy.init_node('controller', anonymous=True)
    image_sub=message_filters.Subscriber("video_stream/pre_img_msgs", Image)
    info_sub=message_filters.Subscriber("video_stream/value_msgs", IntWithHeader)

    ts = message_filters.TimeSynchronizer([image_sub, info_sub], 10)
    ts.registerCallback(callback)
    rospy.spin()

if __name__ == '__main__':
    controller()
