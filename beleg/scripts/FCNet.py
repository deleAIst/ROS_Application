from torch import nn
import  torch.nn.functional as F

class FCNet(nn.Module):
    def __init__(self, input_shape):
        super(FCNet, self).__init__()


        self.channels = input_shape[0]
        self.height = input_shape[1]
        self.width = input_shape[2]


        self.layer_1  = nn.Sequential(
            nn.Linear(in_features=self.height*self.width *self.channels, out_features= 200), 
            nn.ReLU()
        )

        self.layer_2 = nn.Linear(in_features=200, out_features=200)
        self.layer_3 = nn.Linear(in_features=200, out_features=10)


    def forward(self, x):
        x = x.view(-1, self.height * self.width)
        x = self.layer_1(x)
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)

        return F.log_softmax(x, dim=1)