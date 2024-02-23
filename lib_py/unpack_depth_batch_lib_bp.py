#!/usr/bin/env python
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import *

import random
from scipy import ndimage
import scipy.stats as ss
from scipy.ndimage.interpolation import zoom


from lib_py.kinematics_lib_bp import KinematicsLib
from lib_py.preprocessing_lib_bp import PreprocessingLib
from scipy.ndimage.filters import gaussian_filter

# PyTorch libraries
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import transforms
from torch.autograd import Variable
from scipy import stats

MAT_WIDTH = 0.762 #metres
MAT_HEIGHT = 1.854 #metres
MAT_HALF_WIDTH = MAT_WIDTH/2
NUMOFTAXELS_X = 84#73 #taxels
NUMOFTAXELS_Y = 47#30
INTER_SENSOR_DISTANCE = 0.0286#metres
LOW_TAXEL_THRESH_X = 0
LOW_TAXEL_THRESH_Y = 0
HIGH_TAXEL_THRESH_X = (NUMOFTAXELS_X - 1)
HIGH_TAXEL_THRESH_Y = (NUMOFTAXELS_Y - 1)



# import hrl_lib.util as ut
try:
    import cPickle as pickle
except:
    import pickle as pickle
# from hrl_lib.util import load_pickle
def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

class UnpackDepthBatchLib():


    def unpack_batch(self, batch, is_training, model, model_smpl_pmr, model_CAL, model_betanet, CTRL_PNL):

        INPUT_DICT = {}
        OUTPUT_EST_DICT = {}

        # print batch[0].type(), "batch 0 tensor type"
        if CTRL_PNL['depth_out_unet']:
            output_depth_idx = batch[0].size(1) - 4

            if CTRL_PNL['mod'] == 2:
                output_depth_idx -= 4
                dimg_est = batch[0][:, output_depth_idx+4:, :, :]

                if CTRL_PNL['onlyhuman_labels'] == True:
                    depth_noblanket = batch[0][:, output_depth_idx:output_depth_idx+4, :, :]
                    batch[0] = batch[0][:, 0:output_depth_idx, :, :]
                else:
                    batch[0] = batch[0][:, 0:output_depth_idx+4, :, :]

                OUTPUT_EST_DICT['dimg'] = torch.cat(
                    (torch.cat((dimg_est[:, 0:1, :, :], dimg_est[:, 1:2, :, :]), dim=3),
                     torch.cat((dimg_est[:, 2:3, :, :], dimg_est[:, 3:4, :, :]), dim=3)), dim=2).type(CTRL_PNL['dtype'])
                OUTPUT_EST_DICT['dimg_cntct'] = OUTPUT_EST_DICT['dimg'].clone()
                OUTPUT_EST_DICT['dimg_cntct'][OUTPUT_EST_DICT['dimg_cntct'] > 0] = 1
                OUTPUT_EST_DICT['dimg_cntct'][OUTPUT_EST_DICT['dimg_cntct'] <= 0] = 0

            else:
                if CTRL_PNL['onlyhuman_labels'] == True:
                    depth_noblanket = batch[0][:, output_depth_idx:, :, :]
                    batch[0] = batch[0][:, 0:output_depth_idx, :, :]
                #else:
                #    batch[0] = batch[0][:, 0:output_depth_idx+4, :, :]


            if CTRL_PNL['onlyhuman_labels'] == True:
                INPUT_DICT['batch_dimg_noblanket_gt'] = torch.cat(
                    (torch.cat((depth_noblanket[:, 0:1, :, :], depth_noblanket[:, 1:2, :, :]), dim=3),
                     torch.cat((depth_noblanket[:, 2:3, :, :], depth_noblanket[:, 3:4, :, :]), dim=3)), dim=2).type(CTRL_PNL['dtype']).squeeze()

                if len(INPUT_DICT['batch_dimg_noblanket_gt'].size()) == 2:
                    INPUT_DICT['batch_dimg_noblanket_gt'] = INPUT_DICT['batch_dimg_noblanket_gt'].unsqueeze(0)

                INPUT_DICT['batch_dimg_noblanket_gt'][INPUT_DICT['batch_dimg_noblanket_gt'] <= 0.5] = 0.
                INPUT_DICT['batch_dimg_noblanket_cntct_gt'] = INPUT_DICT['batch_dimg_noblanket_gt'].clone()
                INPUT_DICT['batch_dimg_noblanket_cntct_gt'][INPUT_DICT['batch_dimg_noblanket_cntct_gt'] > 0] = 1.



        adj_ext_idx = 0
        # 0:72: positions.
        batch.append(batch[1][:, 72:82])  # betas
        batch.append(batch[1][:, 82:154])  # angles
        batch.append(batch[1][:, 154:157])  # root pos
        batch.append(batch[1][:, 157:159])  # gender switch
        batch.append(batch[1][:, 159])  # synth vs real switch
        batch.append(batch[1][:, 160:161])  # mass, kg
        batch.append(batch[1][:, 161:162])  # height, ??

        if CTRL_PNL['adjust_ang_from_est'] == True:
            adj_ext_idx += 4
            batch.append(batch[1][:, 162:172]) #betas est
            batch.append(batch[1][:, 172:244]) #angles est
            batch.append(batch[1][:, 244:247]) #root pos est
            batch.append(batch[1][:, 247:253]) #root atan2 est
            batch.append(batch[1][:, 253:254]) #bed vertical shift est
            #print "appended root", batch[-1], batch[12]

            extra_smpl_angles = batch[10]
            extra_targets = batch[11]
        else:
            extra_smpl_angles = None
            extra_targets = None



        start_depth = batch[0].size(1) - 4

        depth_batch_idx = len(batch)
        batch.append(batch[0][:, start_depth:, :, :])

        batch[-1] = torch.cat( (torch.cat((batch[-1][:, 0:1, :, :] , batch[-1][:, 1:2, :, :] ), dim = 3),
                          torch.cat((batch[-1][:, 2:3, :, :] , batch[-1][:, 3:4, :, :] ), dim = 3)), dim = 2)

        if CTRL_PNL['slp'] is not 'real':
            batch[-1][:, :, 127:, :] *= 0

        if False:#CTRL_PNL['clean_slp_depth'] == True:
            batch = PreprocessingLib().clean_depth_images(batch)



        batch[0] = batch[0][:, 0:start_depth, :, :]

        # cut it off so batch[2] is only the xyz marker targets
        batch[1] = batch[1][:, 0:72]


        #print(batch[-1].size(), torch.std(batch[-1]), torch.max(batch[-1]))
        if CTRL_PNL['depth_noise'] == True and is_training == True and CTRL_PNL['train_only_CAL'] == False and CTRL_PNL['train_only_betanet'] == False:
            batch[-1], batch[1], batch[4], INPUT_DICT['bed_vertical_shift'] = PreprocessingLib().preprocessing_add_depth_calnoise(batch[-1], batch[1], batch[4])
        else:
            INPUT_DICT['bed_vertical_shift'] = torch.Tensor(np.zeros((batch[0].size()[0], 1)))
        INPUT_DICT['bed_vertical_shift'] = INPUT_DICT['bed_vertical_shift'].type(CTRL_PNL['dtype'])



        if CTRL_PNL['slp_noise'] == True and is_training == True:
            _, batch[-1], batch[1] = PreprocessingLib().preprocessing_add_slp_noise(batch[0].data.cpu().numpy(), batch[-1],  batch[1],
                                                                                  pmat_chan_idx = (CTRL_PNL['num_input_channels_batch0']-2),
                                                                                  norm_std_coeffs = CTRL_PNL['norm_std_coeffs'],
                                                                                  is_training = is_training,
                                                                                  normalize_per_image = CTRL_PNL['normalize_per_image'])



        targets, betas = Variable(batch[1].type(CTRL_PNL['dtype']), requires_grad=False), \
                         Variable(batch[2].type(CTRL_PNL['dtype']), requires_grad=False)

        angles_gt = Variable(batch[3].type(CTRL_PNL['dtype']), requires_grad=is_training)
        root_shift = Variable(batch[4].type(CTRL_PNL['dtype']), requires_grad=is_training)
        gender_switch = Variable(batch[5].type(CTRL_PNL['dtype']), requires_grad=is_training)
        synth_real_switch = Variable(batch[6].type(CTRL_PNL['dtype']), requires_grad=is_training)

        if CTRL_PNL['adjust_ang_from_est'] == True:
            OUTPUT_EST_DICT['betas'] = Variable(batch[9].type(CTRL_PNL['dtype']), requires_grad=is_training)
            OUTPUT_EST_DICT['angles'] = Variable(extra_smpl_angles.type(CTRL_PNL['dtype']), requires_grad=is_training)
            OUTPUT_EST_DICT['root_shift'] = Variable(extra_targets.type(CTRL_PNL['dtype']), requires_grad=is_training)
            OUTPUT_EST_DICT['root_atan2'] = Variable(batch[12].type(CTRL_PNL['dtype']), requires_grad=is_training)
            OUTPUT_EST_DICT['bed_vertical_shift'] = Variable(batch[13].type(CTRL_PNL['dtype']), requires_grad=is_training)



        if CTRL_PNL['recon_map_input_est'] == True and CTRL_PNL['CNN'] == 'resnet':
            input_recon_maps = batch[0][:, 0:2, : ,:]# * 10.
            input_recon_maps = nn.Upsample(scale_factor=2, mode='nearest')(input_recon_maps).type(CTRL_PNL['dtype'])
            batch[0] = batch[0][:, 2:, :, :]
            images_up = torch.cat((batch[depth_batch_idx].type(CTRL_PNL['dtype']),
                                  input_recon_maps), dim = 1)
        elif CTRL_PNL['recon_map_input_est'] == True and CTRL_PNL['CNN'] == 'resnetunet':
            OUTPUT_EST_DICT['pimg'] = batch[0][:, 0, : ,:].type(CTRL_PNL['dtype'])
            OUTPUT_EST_DICT['pimg_cntct'] = OUTPUT_EST_DICT['pimg'].clone()
            OUTPUT_EST_DICT['pimg_cntct'][OUTPUT_EST_DICT['pimg_cntct'] > 0] = 1
            OUTPUT_EST_DICT['pimg_cntct'][OUTPUT_EST_DICT['pimg_cntct'] <= 0] = 0
            batch[0] = batch[0][:, 2:, :, :]
            images_up = torch.cat((batch[depth_batch_idx].type(CTRL_PNL['dtype']),
                                  OUTPUT_EST_DICT['dimg'],
                                  OUTPUT_EST_DICT['dimg_cntct']*100.), dim = 1)
        else:
            images_up = batch[depth_batch_idx].type(CTRL_PNL['dtype'])


        if CTRL_PNL['mesh_recon_map_labels'] == True:
            if CTRL_PNL['mesh_recon_map_labels_test'] == True or is_training == True:
                INPUT_DICT['batch_mdm_gt'] = batch[0][:, 1, : ,:].type(CTRL_PNL['dtype'])
                INPUT_DICT['batch_cm_gt'] = batch[0][:, 2, : ,:].type(CTRL_PNL['dtype'])
        else:
            INPUT_DICT['batch_mdm_gt'] = None
            INPUT_DICT['batch_cm_gt'] = None



        INPUT_DICT['batch_pimg'] = batch[0][:, 0, : ,:].type(CTRL_PNL['dtype'])
        #print(INPUT_DICT['batch_pimg'], 'batch pimg input')
        INPUT_DICT['batch_pimg_cntct'] = batch[0][:, 0, : ,:].type(CTRL_PNL['dtype']).clone()
        INPUT_DICT['batch_pimg_cntct'][INPUT_DICT['batch_pimg_cntct'] > 0] = 1.


        #if CTRL_PNL['train_only_betanet'] == True:
        INPUT_DICT['batch_betas'] = Variable(batch[2].type(CTRL_PNL['dtype']), requires_grad=False)


        INPUT_DICT['batch_gender'] = gender_switch.data
        INPUT_DICT['batch_images'] = images_up.data
        INPUT_DICT['batch_targets'] = targets.data

        # if CTRL_PNL['train_only_betanet'] == False:
        #     print(images_up.size(), 'network input size')

        OUTPUT_DICT = {}
        scores = None


        if CTRL_PNL['loss_vector_type'] == 'direct' and CTRL_PNL['slp'] == 'real':
            scores, OUTPUT_DICT = model.forward_slp_direct(images=images_up,
                                                           CTRL_PNL=CTRL_PNL,
                                                           targets=targets,
                                                           is_training=is_training,
                                                           )  # scores is a variable with 27 for 10 euclidean errors and 17 lengths in meters. targets est is a numpy array in mm.
        elif CTRL_PNL['train_only_betanet'] == False and CTRL_PNL['train_only_CAL'] == False:
            scores, OUTPUT_DICT = model.forward_kinematic_angles_ptA(images=images_up,
                                                                 CTRL_PNL=CTRL_PNL,
                                                                 OUTPUT_EST_DICT=OUTPUT_EST_DICT,
                                                                 is_training=is_training
                                                                 )  # scores is a variable with 27 for 10 euclidean errors and 17 lengths in meters. targets est is a numpy array in mm.

            scores, OUTPUT_DICT = model_smpl_pmr.forward_kinematic_angles_ptB(images=images_up,
                                                                 scores = scores,
                                                                 gender_switch=gender_switch,
                                                                 synth_real_switch=synth_real_switch,
                                                                 CTRL_PNL=CTRL_PNL,
                                                                 OUTPUT_DICT = OUTPUT_DICT,
                                                                 OUTPUT_EST_DICT=OUTPUT_EST_DICT,
                                                                 INPUT_DICT = INPUT_DICT,
                                                                 targets=targets,
                                                                 is_training=is_training,
                                                                 betas=betas,
                                                                 angles_gt=angles_gt,
                                                                 root_shift=root_shift,
                                                                 )  # scores is a variable with 27 for 10 euclidean errors and 17 lengths in meters. targets est is a numpy array in mm.


        if CTRL_PNL['CNN'] == 'resnet' and CTRL_PNL['train_only_betanet'] == False and CTRL_PNL['mesh_recon_map_output'] == True:
            OUTPUT_DICT = model_CAL.forward(CTRL_PNL =CTRL_PNL,
                                               INPUT_DICT = INPUT_DICT,
                                               OUTPUT_DICT = OUTPUT_DICT)

        if CTRL_PNL['train_only_CAL'] == False:
            OUTPUT_DICT = model_betanet.forward(CTRL_PNL =CTRL_PNL,
                                               INPUT_DICT = INPUT_DICT,
                                               OUTPUT_DICT = OUTPUT_DICT)




        INPUT_DICT['batch_weight_kg'] = Variable(batch[7].type(CTRL_PNL['dtype']), requires_grad=False)
        INPUT_DICT['batch_height'] = Variable(batch[8].type(CTRL_PNL['dtype']), requires_grad=False)
        #print(targets.size(), targets.data[0], scores.size())





        return scores, INPUT_DICT, OUTPUT_DICT

