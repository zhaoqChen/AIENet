import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MRF(nn.Module):
    def __init__(self, channel):
        super(MRF, self).__init__()

        self.conv1_left = nn.Conv2d(channel, 32, 3, 1, 1, bias=False) 
        self.conv1_right = nn.Conv2d(channel, 128, 3, 1, 1, bias=False)
        self.norm1 = nn.InstanceNorm2d(64, affine=True)

        self.conv2_1 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.conv2_2 = nn.Conv2d(32, 32, 5, 1, 2, bias = False)
        self.conv2_3 = nn.Conv2d(32, 32, 9, 1, 4, bias=False)
        self.conv2_4 = nn.Conv2d(32, 32, 13, 1, 6, bias=False)

        self.conv3 = nn.Conv2d(128,128,3,1,1, bias = False)

        self.norm64 = nn.InstanceNorm2d(32, affine=True)
        self.norm256 = nn.InstanceNorm2d(128, affine=True)
        self.norm512 = nn.InstanceNorm2d(256, affine=True)


    def forward(self, x):
        # x: [1,64,320,320]
        x_left = F.relu(self.norm64(self.conv1_left(x))) 
        x_right = F.relu(self.norm256(self.conv1_right(x))) 
        
        x2_1 = F.relu(self.norm64(self.conv2_1(x_left))) 
        x2_2 = F.relu(self.norm64(self.conv2_2(x_left))) 
        x2_3 = F.relu(self.norm64(self.conv2_3(x_left))) 
        x2_4 = F.relu(self.norm64(self.conv2_4(x_left))) 

        x2 = torch.cat([torch.cat([torch.cat([x2_1, x2_2], 1),x2_3], 1),x2_4],1) 

        x3 = F.relu(self.norm256(self.conv3(x2))) 

        x4 = torch.cat([x_right, x3],1) 

        return x4



class AIENet(nn.Module):
    def __init__(self):
        super(AIENet, self).__init__()
        self.conv0 = nn.Conv2d(3, 32, 3, 1, 1, bias=False) # kernel_size=3 
        self.conv1 = nn.Conv2d(32, 64, 3, 2, 1, bias=False) # kernel_size=3
        self.norm1 = nn.InstanceNorm2d(64, affine=True)
        self.spp_64 = SPPCSPC(64)
        self.spp_256 = SPPCSPC(256)
        self.deconv1_1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        # self.deconv1_1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)

        self.conv2 = nn.Conv2d(128, 64, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64, 32, 3, 1, 1, bias=False)
        self.conv4 = nn.Conv2d(32, 3, 3, 1, 1, bias=False)


    def forward(self, x):
    # x:[1,3,640,640]
        x_0 = F.relu(self.conv0(x))      
        x_1 = F.relu(self.conv1(x_0))     
        x_2 = F.relu(self.spp_64(x_1)) 
        x_3 = F.relu(self.spp_256(x_2))
        x_4 = F.relu(self.spp_256(x_3))
        x_5 = F.relu(self.deconv1_1(x_4))
        x_6 = F.relu(self.conv2(x_5))    
        x_7 = F.relu(self.conv3(x_6))     
        x_7 = x_7 + x_0
        x_8 = F.relu(self.conv4(x_7))    

        return x_8
