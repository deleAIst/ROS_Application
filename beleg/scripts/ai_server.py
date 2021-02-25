#!/usr/bin/env python3
# license removed for brevity

import rospkg
import rospy
import torch

from beleg.srv import AI, AIResponse
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from FCNet import FCNet

cv_bridge = CvBridge()
model = FCNet(input_shape=[1, 28, 28])
rospack = rospack.RosPack()
#dynamically find path to given package in ROS
PATH = rospack.get_path('beleg')
#loading pretrained weights and biases into state_dict
state_dict =torch.load(PATH + '/scripts/mnist_simple_fc.pt')

#feed trained parameters into the model
model.load_state_dict(state_dict)

"""
This method takes an img_msg and returns a predicting what number (int32) this image represents to the Network.
"""
def  handle_AI_request(req):
    img = cv_bridge.imgmsg_to_cv2(req.image)
    #rospy.loginfo(img)
    img = torch.Tensor(img)

    a = model(img)
    predigt = torch.argmax(a, dim=1)
    #rospy.loginfo(a)
    return AIResponse(predigt[0])

def add_AI_server():
    rospy.init_node('ai_server')
    s = rospy.Service('ai_server', AI, handle_AI_request)
    rospy.spin()


if __name__ == '__main__':
   try:
       add_AI_server()
   except rospy.ROSInterruptException:
       pass        
