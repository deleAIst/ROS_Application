#!/usr/bin/env python3
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from PIL import Image as i

cv_bridge = CvBridge()

def callback(data):
    """[summary]
            Converts RGB image to monochrome image (reducing channels from 3 to 1)
    Args:
        data ([sensor_msg/Image]): [RGB image]
    """
    img = cv_bridge.imgmsg_to_cv2(data)
    # converting numpy array -> Python Image Library Format -> reducing channels from RGB to Black&White -> back to numpy array
    img = np.asarray(i.fromarray(img).convert('L'))
    # creating a publisher
    pub_imag = rospy.Publisher('video_stream/pre_img_msgs', Image, queue_size=1)       
    img_shape_msg = cv_bridge.cv2_to_imgmsg(img)
    # sending the message to subscribers
    pub_imag.publish(img_shape_msg)

def processor():
    """[summary]
            Init of a node "processor", linking it to the Camera (image publisher)
    """
    rospy.init_node('processor', anonymous=True)
    rospy.Subscriber("video_stream/img_msgs", Image, callback)
    rospy.spin()

if __name__ == '__main__':
    processor()