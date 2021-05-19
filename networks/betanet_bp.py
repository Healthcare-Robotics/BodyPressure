import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import scipy.stats as ss
import torchvision
import time


from visualization_lib_bp import VisualizationLib
from kinematics_lib_bp import KinematicsLib


class FC(nn.Module):
    def __init__(self, CTRL_PNL = None):
        '''
        Create components of a CNN classifier and initialize their weights.

        Arguments:
            mat_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            out_size (int): Number of classes to score
        '''

        super(FC, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        #print mat_size

        self.count = 0


        #select_architecture = 0
        #if select_architecture == 0:
        self.FC_pack_64 = nn.Sequential(
            nn.Linear(12, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

        if torch.cuda.is_available():
            self.GPU = True
            # Use for self.GPU
            dtype = torch.cuda.FloatTensor


        else:
            self.GPU = False
            # Use for CPU
            dtype = torch.FloatTensor

        self.dtype = dtype





    def forward(self, CTRL_PNL, INPUT_DICT, OUTPUT_DICT):

        if CTRL_PNL['GPU'] == True:
            self.GPU = True
            self.dtype = torch.cuda.FloatTensor
        else:
            self.GPU = False
            self.dtype = torch.FloatTensor

        #current_batch_size = INPUT_DICT['batch_pimg'].size()[0]

        if CTRL_PNL['train_only_betanet'] == True:
            batch_wtht = self.FC_pack_64(torch.cat((INPUT_DICT['batch_gender'], INPUT_DICT['batch_betas']), axis = 1))
        else:
            batch_wtht = self.FC_pack_64(torch.cat((INPUT_DICT['batch_gender'], OUTPUT_DICT['batch_betas_est_post_clip']), axis = 1))

        OUTPUT_DICT['batch_weight_kg_est'] = batch_wtht[:, 0:1]
        OUTPUT_DICT['batch_height_est'] = batch_wtht[:, 1:2]

        return OUTPUT_DICT

