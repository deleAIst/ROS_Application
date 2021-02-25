from torch import nn
import  torch.nn.functional as F

class FCNet(nn.Module):
    """[summary]
        Creating a fully connected Vanilla Neural Network.
        Structure contains 3 Layer:
            - 1st Layer consists of 28x28*1 (784) neurons, one for each pixel of the input data (*1 due to the image is monochrome)*
            - 2nd Layer is a hidden Layer, meant to learn about the features important for image recognition (ideally e.g. edge detection, shapes ect.)
            - 3rd Layer represents the output-layer mapping the 200 neurons of the hiddenlayer to 10 output neurons, representing the numbers to be classified

        *   - activation function is set to ReLu (rectified linear unit, designed to introduce non-linearity to the network)

        Due to the fact, that the activation value of the output neurons do not represent probabilities, 
        a logistic softmax function is applied to these values in the very end of the process to interprete them as likelihood.
    """
    def __init__(self, input_shape):
        """[summary]
            Initialization of the Network components
        Args:
            input_shape ([array]): [Dimensions of the input]
        """
        super(FCNet, self).__init__()

        #selecting the channel (since the image is black&white there is only one channel representing brightness on a gray-scale)
        self.channels = input_shape[0]
        #determining the input shape 
        self.height = input_shape[1]
        self.width = input_shape[2]

        # input layer
        self.layer_1  = nn.Sequential(
            nn.Linear(in_features=self.height*self.width *self.channels, out_features= 200), 
            nn.ReLU()
        )
        
        #hidden layer
        self.layer_2 = nn.Linear(in_features=200, out_features=200)
        
        #output layer
        self.layer_3 = nn.Linear(in_features=200, out_features=10)

    def forward(self, x):
        """[summary]
            passing values through the network
        Args:
            x ([Tensor]): [image represented as PyTensor]

        Returns:
            [Tensor]: [Tensor contaning 10 values, representing the likelihood for each category]
        """
        x = x.view(-1, self.height * self.width)
        x = self.layer_1(x)
        # introducing a 2nd activation function (ReLU again)
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)

        return F.log_softmax(x, dim=1)
