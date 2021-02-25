#!/usr/bin/env python3
# license removed for brevity

import rospy
import torch

from beleg.srv import AI, AIResponse
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from FCNet import FCNet

cv_bridge = CvBridge()
model = FCNet(input_shape=[1, 28, 28])
state_dict =torch.load('/home/dennis/catkin_ws/src/beleg/scripts/mnist_simple_fc.pt')
model.load_state_dict(state_dict)

def  handel_AI_request(req):
    img = cv_bridge.imgmsg_to_cv2(req.image)
    #rospy.loginfo(img)
    img = torch.Tensor(img)

    a = model(img)
    predigt = torch.argmax(a, dim=1)
    #rospy.loginfo(a)
    return AIResponse(predigt[0])

def add_AI_server():
    rospy.init_node('ai_server')
    s = rospy.Service('ai_server', AI, handel_AI_request)
    rospy.spin()


if __name__ == '__main__':
   try:
       add_AI_server()
   except rospy.ROSInterruptException:
       pass        
