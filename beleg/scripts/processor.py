#!/usr/bin/env python3
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from PIL import Image as i

cv_bridge = CvBridge()

def callback(data):
    img = cv_bridge.imgmsg_to_cv2(data)
    img =np.asarray(i.fromarray(img).convert('L'))
    #her preprocessor from imag
    pub_imag = rospy.Publisher('video_stream/pre_img_msgs', Image, queue_size=1)       
    img_shape_msg = cv_bridge.cv2_to_imgmsg(img)
    pub_imag.publish(img_shape_msg)




def processor():
    rospy.init_node('processor', anonymous=True)
    rospy.Subscriber("video_stream/img_msgs", Image, callback)
    rospy.spin()

if __name__ == '__main__':
    processor()