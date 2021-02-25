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
    """[summary]
        sends img_msg to the AI-Server, compares return value of AI-Server with int_with_header message.
        yields [true, false] for logging

    Args:
        img_msg ([sensor_msg/Image]): [gray-scale input image]
        int_with_header ([custom message]): [true label of image]
    """

    rospy.wait_for_service('ai_server')
    try:
        # subscription of AI-Server 
        ai_service = rospy.ServiceProxy('ai_server', AI)
        # requesting prediction for given image
        result = ai_service(img_msg)

        # evaluation of the result
        if(result.result == int_with_header.data):
            rospy.loginfo("True")
        else:
            rospy.loginfo("False")

        #rospy.loginfo(int_with_header)
        #rospy.loginfo(red)
    except rospy.ServiceException as e:
        print("Service call failed: %s" %e)

def controller():
    """[summary]
        introducing subcriber and publisher to each other
    """
    rospy.init_node('controller', anonymous = True)
    image_sub = message_filters.Subscriber("video_stream/pre_img_msgs", Image)
    info_sub = message_filters.Subscriber("video_stream/value_msgs", IntWithHeader)

    # ensure mapping matching messages based on timestamp
    ts = message_filters.TimeSynchronizer([image_sub, info_sub], 10)
    ts.registerCallback(callback)
    rospy.spin()

if __name__ == '__main__':
    controller()
