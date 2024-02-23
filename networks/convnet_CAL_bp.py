import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import scipy.stats as ss
import torchvision
import time


from lib_py.visualization_lib_bp import VisualizationLib
from lib_py.kinematics_lib_bp import KinematicsLib


class CNN(nn.Module):
    def __init__(self, CTRL_PNL = None):
        '''
        Create components of a CNN classifier and initialize their weights.

        Arguments:
            mat_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            out_size (int): Number of classes to score
        '''

        super(CNN, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        #print mat_size

        self.count = 0


        #select_architecture = 0
        #if select_architecture == 0:
        self.CNN_pack_1x1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )

        self.CNN_pack_3x3 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )

        if torch.cuda.is_available():
            self.GPU = True
            # Use for self.GPU
            dtype = torch.cuda.FloatTensor
            # print('######################### CUDA is available! #############################')
            if CTRL_PNL['CNN'] == 'resnetunet':
                self.resnet_zeros = torch.Tensor(np.ones((128, in_channels, 128, 5))).type(torch.cuda.FloatTensor)
            if CTRL_PNL['CNN'] == 'resnet':
                self.resnet_zeros = torch.Tensor(np.zeros((128, 1, 1, 27))).type(torch.cuda.FloatTensor)

                self.resnet_ccx = torch.Tensor(np.ones((128, 1, 64, 27))).type(torch.cuda.FloatTensor)
                for i in range(64):
                    self.resnet_ccx[:, :, i, :] *=  i/6.3
                self.resnet_ccy = torch.Tensor(np.zeros((128, 1, 64, 27))).type(torch.cuda.FloatTensor)
                for j in range(27):
                    self.resnet_ccy[:, :, :, j] *=  i/2.6


        else:
            self.GPU = False
            # Use for CPU
            dtype = torch.FloatTensor
            # print('############################## USING CPU #################################')
            if CTRL_PNL['CNN'] == 'resnetunet':
                self.resnet_zeros = torch.Tensor(np.zeros((128, in_channels, 128, 5))).type(torch.FloatTensor)
            if CTRL_PNL['CNN'] == 'resnet':
                self.resnet_zeros = torch.Tensor(np.zeros((128, 1, 1, 27))).type(torch.FloatTensor)


            #if CTRL_PNL['CNN'] == 'resnet':
            #    self.resnet_ccx = torch.Tensor(np.zeros((128, 1, 64, 27))).type(torch.FloatTensor)
            #    self.resnet_ccy = torch.Tensor(np.zeros((128, 1, 64, 27))).type(torch.FloatTensor)
        self.dtype = dtype

        self.zeros_z = torch.zeros(128, 24, 1).type(self.dtype)
        self.sqrt_filler = torch.zeros(128).type(self.dtype)
        self.sqrt_filler+= 0.0000001





    def forward(self, CTRL_PNL, INPUT_DICT, OUTPUT_DICT):

        if CTRL_PNL['GPU'] == True:
            self.GPU = True
            self.dtype = torch.cuda.FloatTensor
        else:
            self.GPU = False
            self.dtype = torch.FloatTensor

        current_batch_size = INPUT_DICT['batch_pimg'].size()[0]

        if CTRL_PNL['train_only_CAL'] == True:
            OUTPUT_DICT['batch_pimg_est'] = INPUT_DICT['batch_mdm_gt'].clone().unsqueeze(1) * -1
        else:
            OUTPUT_DICT['batch_pimg_est'] = OUTPUT_DICT['batch_mdm_est'].clone().unsqueeze(1) * -1

        OUTPUT_DICT['batch_pimg_est'][OUTPUT_DICT['batch_pimg_est'] < 0] = 0
        OUTPUT_DICT['batch_pimg_est'] = torch.cat((OUTPUT_DICT['batch_pimg_est'], self.resnet_ccx[0:current_batch_size, :, :, :], self.resnet_ccy[0:current_batch_size, :, :, :]), dim =1 )
        OUTPUT_DICT['batch_pimg_est'] = self.CNN_pack_3x3(OUTPUT_DICT['batch_pimg_est']).squeeze()

        if len(OUTPUT_DICT['batch_pimg_est'].size()) == 2:
            OUTPUT_DICT['batch_pimg_est'] = OUTPUT_DICT['batch_pimg_est'].unsqueeze(0)

        #print("*************")
        #print(OUTPUT_DICT['batch_pimg_est'].size(), 'out pimg est')

        #OUTPUT_DICT['batch_pimg_est'] *= 100.
        #OUTPUT_DICT['batch_pimg_est'] -= 10.
        OUTPUT_DICT['batch_pimg_est'][OUTPUT_DICT['batch_pimg_est'] <= 0.0] = 0.
        OUTPUT_DICT['batch_pimg_cntct_est'] = OUTPUT_DICT['batch_pimg_est'].clone()
        OUTPUT_DICT['batch_pimg_cntct_est'][OUTPUT_DICT['batch_pimg_cntct_est'] > 0] = 1.

        return OUTPUT_DICT

