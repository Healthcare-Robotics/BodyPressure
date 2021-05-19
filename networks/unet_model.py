""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, fc_out_size, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.fc1 = nn.Sequential(
            nn.Linear(8*3*1024 // factor, fc_out_size),  # 89600, out_size),
        )

    def forward(self, x):
        x1 = self.inc(x)
        print ("x1: ", x1.shape)
        x2 = self.down1(x1)
        print ("x2: ", x2.shape)
        x3 = self.down2(x2)
        print ("x3: ", x3.shape)
        x4 = self.down3(x3)
        print ("x4: ", x4.shape)
        x5 = self.down4(x4)
        print ("x5: ", x5.shape)
        x = self.up1(x5, x4)
        print ("x: ", x.shape)
        x = self.up2(x, x3)
        print ("x: ", x.shape)
        x = self.up3(x, x2)
        print ("x: ", x.shape)
        x = self.up4(x, x1)
        print ("x: ", x.shape)
        logits = self.outc(x)
        print ("logits: ", logits.shape, logits.type(), logits)
        return logits

    def forward_half(self, x):
        x1 = self.inc(x)
        #print "x1: ", x1.shape
        x2 = self.down1(x1)
        #print "x2: ", x2.shape
        x3 = self.down2(x2)
        #print "x3: ", x3.shape
        x4 = self.down3(x3)
        #print "x4: ", x4.shape
        x5 = self.down4(x4)
        #print "x5: ", x5.shape

        x5_size = x5.size()
        x5_reshaped = x5.view(x5_size[0],x5_size[1]*x5_size[2]*x5_size[3])
        #print x5_reshaped.shape
        output = self.fc1(x5_reshaped)
        #print output.shape


        return output
