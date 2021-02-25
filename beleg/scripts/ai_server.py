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
# dynamically find path to given package in ROS
PATH = rospack.get_path('beleg')
# loading pretrained weights and biases into state_dict
state_dict =torch.load(PATH + '/scripts/mnist_simple_fc.pt')

# feed trained parameters into the model
model.load_state_dict(state_dict)

def  handle_AI_request(req):
    """[summary]
        This method takes an img_msg and returns a predicting what number (int32) this image represents to the Network.
    Args:
        req ([sensor_msgs/Image]): ["request" -> image as sensor_msg]

    Returns:
        [int32]: [predicted result]
    """
    # converting the sensor_msg to numpy array
    img = cv_bridge.imgmsg_to_cv2(req.image)
    # converting numpy array to PyTorch Tensor
    img = torch.Tensor(img)
    # feeding Tensor to the model
    a = model(img)
    # choosing the index of the highes value from the models returned Tensor
    predict = torch.argmax(a, dim=1)
    # Tensor to int ♥
    return AIResponse(predict[0])

def add_AI_server():
    """[summary]
        initialized the service node, waiting for incomming requests
    """
    # creating node
    rospy.init_node('ai_server')
    # setting the node te be a service, passing the relevant object (AI) and method(handle_AI_request) necessary to work
    s = rospy.Service('ai_server', AI, handle_AI_request)
    # keep running!
    rospy.spin()


if __name__ == '__main__':
   try:
       add_AI_server()
   except rospy.ROSInterruptException:
       pass        
