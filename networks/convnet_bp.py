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
from lib_py.mesh_depth_lib_bp import MeshDepthLib

from resnet import resnet34 as ResNet34
#from resnet import resnet50 as ResNet50
from resnetunet import ResNetUNet as ResNetUNet34 #use from resnetunet_dp for a 64x27 instead of a 128x54
from unet_model import UNet
#from learned_reconstruction import fcn_resnet50


class CNN(nn.Module):
    def __init__(self, out_size, loss_vector_type, in_channels = 3, CTRL_PNL = None):
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
        self.loss_vector_type = loss_vector_type
        # print ('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

        self.count = 0



        if CTRL_PNL['CNN'] == 'resnet':
            self.resnet = ResNet34(pretrained=False, num_classes=out_size)
            self.resnet._modules['conv1'] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias = False)
            self.resnet_fc_last = torch.nn.Sequential(*(list(self.resnet.children())[9:10]))


        if CTRL_PNL['CNN'] == 'resnetunet':
            self.resnetunet = ResNetUNet34(n_input_class = in_channels, n_scores = out_size, n_out_class = 4)
            self.resnetunet._modules['layer0'][0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias = False)
            #print self.resnetunet._modules

        #print self.resnet._modules['layer3']
        #print self.resnet._modules['layer3']
        #print self.resnet._modules['layer3'][0]

        #for item in self.resnet._modules['layer3'][0].modules():
        #    print "modules", item

        #print self.resnet._modules['layer3'][0].modules()

        #for (name, layer) in self.resnet._modules.items():
            # iteration over outer layers
        #    print((name, layer))
        #for (name, layer) in self.resnet._modules.items():
            # iteration over outer layers
        #    print((name))


        #self.resnet_fcn = fcn_resnet50(num_classes=4)#torch.hub.load('pytorch/vision:v0.6.0', 'fcn_resnet101', pretrained=False)

        #modify the first layer so it only takes as input 1 channel
        #self.resnet_fcn._modules['backbone']['conv1'] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias = False)

        #self.resnet_orig._modules = self.resnet_orig._modules['backbone']

        #print 'backbone', self.resnet_orig._modules['backbone']
        #print self.resnet_fcn._modules




        #self.FCN_recon = nn.Sequential(
        #    nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1, bias=False),
        #    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #    nn.ReLU(inplace=True),
        #    nn.Dropout(p=0.1, inplace=False),
        #    nn.Conv2d(512, 21, kernel_size=1, stride=1),
        #)

        #print self.FCN_recon


        #self.unet = UNet(n_classes=1,n_channels=1, fc_out_size=out_size, bilinear=False)


        print ('Out size:', out_size)

        if torch.cuda.is_available():
            self.GPU = True
            # Use for self.GPU
            dtype = torch.cuda.FloatTensor
            # logger.info('######################### CUDA is available! #############################')
            if CTRL_PNL['CNN'] == 'resnetunet':
                self.resnet_zeros = torch.Tensor(np.ones((128, in_channels, 128, 5))).type(torch.cuda.FloatTensor)

        else:
            self.GPU = False
            # Use for CPU
            dtype = torch.FloatTensor
            # print('############################## USING CPU #################################')
            if CTRL_PNL['CNN'] == 'resnetunet':
                self.resnet_zeros = torch.Tensor(np.zeros((128, in_channels, 128, 5))).type(torch.FloatTensor)


        self.dtype = dtype

        self.zeros_z = torch.zeros(128, 24, 1).type(self.dtype)



    def forward_slp_direct(self, images, CTRL_PNL, targets=None, is_training = True):



        #images = torch.cat((images[:, 1:CTRL_PNL['num_input_channels_batch0']-2, :, :], images[:, CTRL_PNL['num_input_channels_batch0']:, :, :]), dim = 1)

        #print ("ConvNet input size: ", images.size())


        OUTPUT_DICT = {}

        if CTRL_PNL['CNN'] == "original":
            if CTRL_PNL['all_tanh_activ'] == True:
                scores_cnn = self.CNN_packtanh(images)
            else:
                scores_cnn = self.CNN_pack1(images)
            scores_size = scores_cnn.size()
            scores_cnn = scores_cnn.view(images.size(0),scores_size[1] *scores_size[2]*scores_size[3])
            scores = self.CNN_fc1(scores_cnn)


        elif CTRL_PNL['CNN'] == "resnet":
            images_to_manip = images.clone()
            for i in range(9):
                newmodel = torch.nn.Sequential(*(list(self.resnet.children())[i:i+1]))
                images_to_manip = newmodel(images_to_manip)#['out']
                #print(i, images_to_manip.shape)
            scores = self.resnet_fc_last(images_to_manip.squeeze())
            #print (scores.shape, "scores shape")


        elif CTRL_PNL['CNN'] == "resnetunet":
            images_padded = torch.cat((self.resnet_zeros[0:images.size()[0]], images, self.resnet_zeros[0:images.size()[0]]), dim = 3)
            scores, recon = self.resnetunet.forward(images_padded, is_training = is_training, verbose=False)
            OUTPUT_DICT['fcn_recon'] = recon



        if len(scores.size()) == 1:
            scores = scores.unsqueeze(0)



        #OUTPUT_DICT['batch_targets_est'] = torch.zeros((90, 72)).type(torch.cuda.FloatTensor).data  # don't set this equal to the scores UNTIL we have the scores in a SMPL format.


        OUTPUT_DICT['batch_betas_est'] = None
        OUTPUT_DICT['batch_root_atan2_est'] = None
        OUTPUT_DICT['batch_angles_est']  = None
        OUTPUT_DICT['batch_root_xyz_est'] = None

        OUTPUT_DICT['batch_betas_est_post_clip'] = None
        OUTPUT_DICT['batch_root_xyz_est_post_clip'] = None

        OUTPUT_DICT['batch_mdm_est'] = None
        OUTPUT_DICT['batch_cm_est'] = None
        OUTPUT_DICT['batch_mdm_est'] = None
        OUTPUT_DICT['batch_cm_est'] = None
        #

        #compare the output joints to the target values
        targets = torch.reshape(targets, (-1, 24, 3))
        targets = targets[:, :, 0:2]

        #if is_training ==False:
        scores = scores.unsqueeze(0)
        scores = scores.unsqueeze(0)
        scores = F.pad(scores, (0, 48, 0, 0))
        scores = scores.squeeze(0)
        scores = scores.squeeze(0)

        smpl_joint_back_num = [1,2,4,5,7,8,12,15,16,17,18,19,20,21]
        leeds_joint_back_num = [2,3,1,4,0,5,12,13,8,9,7,10,6,11]
        offset = 28
        for i in range(14):
            leeds_idx = leeds_joint_back_num[i]
            smpl_idx = smpl_joint_back_num[i]
            scores[:, offset+smpl_idx*2:offset+smpl_idx*2+2] = scores[:, leeds_idx*2:leeds_idx*2+2]

        scores = scores.unsqueeze(0)
        scores = scores.unsqueeze(0)
        scores = F.pad(scores, (-28, 0, 0, 0))
        scores = scores.squeeze(0)
        scores = scores.squeeze(0)
        scores = torch.reshape(scores, (-1, 24, 2))

        #print('targets size', targets.size(), targets[0])
        #print('scores size', scores.size(), scores[0])

        #targets are in SMPL order already. keep that way.

        out_batch_tar_est = torch.cat((scores, self.zeros_z[0:scores.size()[0], :]), dim=2)


        OUTPUT_DICT['batch_targets_est'] = torch.reshape(out_batch_tar_est, (-1, 72)).data*1000 #don't set this equal to the scores UNTIL we have the scores in a SMPL format.


        scores = torch.reshape(scores, (-1, 48))
        targets = torch.reshape(targets, (-1, 48))

        scores = scores.unsqueeze(0)
        scores = scores.unsqueeze(0)
        scores = F.pad(scores, (14, 48, 0, 0))
        scores = scores.squeeze(0)
        scores = scores.squeeze(0)
        scores[:, 14:62] = targets[:, 0:48]/1000 - scores[:, 14:62]
        scores[:, 62:110] = ((scores[:, 14:62].clone())+0.0000001).pow(2)

        ct = 0
        for joint_num in [1,2,4,5,7,8,12,15,16,17,18,19,20,21]:

            scores[:, ct] = (scores[:, 62+joint_num*2] +
                             scores[:, 63+joint_num*2]).sqrt()
            ct += 1

        return scores, OUTPUT_DICT








    def forward_kinematic_angles_ptA(self, images, CTRL_PNL, OUTPUT_EST_DICT, is_training = True):

        OUTPUT_DICT = {}

        self.GPU = CTRL_PNL['GPU']
        self.dtype = CTRL_PNL['dtype']


        if CTRL_PNL['first_pass'] == False:
            pass
        else:
            if CTRL_PNL['GPU'] == True:
                self.GPU = True
                self.dtype = torch.cuda.FloatTensor
            else:
                self.GPU = False
                self.dtype = torch.FloatTensor


        #print ("ConvNet input size: ", images.size())

        if CTRL_PNL['CNN'] == "resnet":
            images_to_manip = images.clone()

            #pass through each layer of the resnet
            for i in range(9):
                newmodel = torch.nn.Sequential(*(list(self.resnet.children())[i:i+1]))
                images_to_manip = newmodel(images_to_manip)#['out']
            scores = self.resnet_fc_last(images_to_manip.squeeze())

            if len(scores.size()) == 1:
                scores = scores.unsqueeze(0)




        elif CTRL_PNL['CNN'] == "resnetunet":
            images_padded = torch.cat((self.resnet_zeros[0:images.size()[0]], images, self.resnet_zeros[0:images.size()[0]]), dim = 3)
            scores, recon = self.resnetunet.forward(images_padded, is_training = is_training, verbose=False)

            #recon = recon * 100.
            #scores = self.resnet_fc_last(images_to_manip.squeeze())

            if CTRL_PNL['recon_map_input_est'] == False:
                OUTPUT_DICT['batch_dimg_est'] = recon[:, 0, :, :].squeeze()*100.
                OUTPUT_DICT['batch_dimg_cntct_est'] = recon[:, 1, :, :].squeeze()

                OUTPUT_DICT['batch_pimg_est'] = nn.AvgPool2d(2, stride=2)(recon[:, 2:3, :, :].clone()).type(CTRL_PNL['dtype']).squeeze()*10.
                OUTPUT_DICT['batch_pimg_cntct_est'] = nn.AvgPool2d(2, stride=2)(recon[:, 3:4, :, :].clone()).type(CTRL_PNL['dtype']).squeeze()
            else:
                OUTPUT_DICT['batch_dimg_est'] = recon[:, 0, :, :].squeeze()*100. + OUTPUT_EST_DICT['dimg'].squeeze()
                OUTPUT_DICT['batch_dimg_cntct_est'] = recon[:, 1, :, :].squeeze() + OUTPUT_EST_DICT['dimg_cntct'].squeeze()

                OUTPUT_DICT['batch_pimg_est'] = nn.AvgPool2d(2, stride=2)(recon[:, 2:3, :, :].clone()).type(CTRL_PNL['dtype']).squeeze()*10. + OUTPUT_EST_DICT['pimg'].squeeze()
                OUTPUT_DICT['batch_pimg_cntct_est'] = nn.AvgPool2d(2, stride=2)(recon[:, 3:4, :, :].clone()).type(CTRL_PNL['dtype']).squeeze() + OUTPUT_EST_DICT['pimg_cntct'].squeeze()



            if len(OUTPUT_DICT['batch_pimg_est'].size()) == 2:
                OUTPUT_DICT['batch_dimg_est'] = OUTPUT_DICT['batch_dimg_est'].unsqueeze(0)
                OUTPUT_DICT['batch_dimg_cntct_est'] = OUTPUT_DICT['batch_dimg_cntct_est'].unsqueeze(0)
                OUTPUT_DICT['batch_pimg_est'] = OUTPUT_DICT['batch_pimg_est'].unsqueeze(0)
                OUTPUT_DICT['batch_pimg_cntct_est'] = OUTPUT_DICT['batch_pimg_cntct_est'].unsqueeze(0)

            OUTPUT_DICT['batch_dimg_cntct_est_mult'] = OUTPUT_DICT['batch_dimg_cntct_est'].clone()
            OUTPUT_DICT['batch_dimg_cntct_est_mult'][OUTPUT_DICT['batch_dimg_cntct_est_mult'] < 0.5] = 0.
            OUTPUT_DICT['batch_dimg_cntct_est_mult'][OUTPUT_DICT['batch_dimg_cntct_est_mult'] > 0.5] = 1.


            OUTPUT_DICT['batch_pimg_cntct_est_mult'] = OUTPUT_DICT['batch_pimg_cntct_est'].clone()
            OUTPUT_DICT['batch_pimg_cntct_est_mult'][OUTPUT_DICT['batch_pimg_cntct_est_mult'] < 0.5] = 0.
            OUTPUT_DICT['batch_pimg_cntct_est_mult'][OUTPUT_DICT['batch_pimg_cntct_est_mult'] > 0.5] = 1.


            OUTPUT_DICT['batch_pimg_est'] = OUTPUT_DICT['batch_pimg_est'].clone()*OUTPUT_DICT['batch_pimg_cntct_est_mult']
            OUTPUT_DICT['batch_dimg_est'] = OUTPUT_DICT['batch_dimg_est'].clone()*OUTPUT_DICT['batch_dimg_cntct_est_mult']




        elif CTRL_PNL['CNN'] == "resnetfcn":
            scores1 = self.resnet_fcn(images)['out']
            print (scores1)
            print (scores1.shape)


        if len(scores.size()) == 1:
            scores = scores.unsqueeze(0)


        return scores, OUTPUT_DICT

