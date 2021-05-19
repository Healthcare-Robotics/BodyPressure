#!/usr/bin/env python
import sys
import os
import time
import numpy as np
# PyTorch libraries
import argparse




def get_depthnet_options(p):

    p.add_option('--hd', action='store_true', dest='hd', default=False,
                 help='Read and write to data on an external harddrive.')

    p.add_option('--losstype', action='store', type = 'string', dest='losstype', default='anglesDC',
                 help='Choose direction cosine or euler angle regression.')

    p.add_option('--X_is', action='store', type = 'string', dest='X_is', default='W',
                 help='Select W or B for white box or black box.')

    p.add_option('--j_d_ratio', action='store', type = 'float', dest='j_d_ratio', default=0.5, #PMR parameter to adjust loss function 2
                 help='Set the loss mix: joints to depth planes. Only used for PMR regression.')

    p.add_option('--prev_device', action='store', type = 'int', dest='prev_device', default=0,
                 help='Choose a GPU core that it was previously on.')

    p.add_option('--device', action='store', type = 'int', dest='device', default=0,
                 help='Choose a GPU core.')

    p.add_option('--qt', action='store_true', dest='quick_test', default=False,
                 help='Do a quick test.')

    p.add_option('--pmr', action='store_true', dest='pmr', default=False,
                 help='Run PMR on input plus precomputed spatial maps.')

    p.add_option('--go200', action='store_true', dest='go200', default=False,
                 help='Run network 1 for 100 to 200 epochs.')

    p.add_option('--small', action='store_true', dest='small', default=False,
                 help='Make the dataset 1/4th of the original size.')

    p.add_option('--hns', action='store_true', dest='hns', default=False,
                 help='Half network size.')

    p.add_option('--htwt', action='store_true', dest='htwt', default=False,
                 help='Include height and weight info on the input.')

    p.add_option('--pcsum', action='store_true', dest='pimg_cntct_sum', default=False,
                 help='Cut contact and sobel from input.')

    p.add_option('--omit_pimg_cntct_sobel', action='store_true', dest='omit_pimg_cntct_sobel', default=False,
                 help='Cut pressuremap and contact and sobel from input.')

    p.add_option('--calnoise', action='store_true', dest='calnoise', default=False,
                 help='Apply calibration noise to the input to facilitate sim to real transfer.')

    p.add_option('--slpnoise', action='store_true', dest='slpnoise', default=False,
                 help='Apply slp noise to the input to reduce overfitting.')

    p.add_option('--half_shape_wt', action='store_true', dest='half_shape_wt', default=False,
                 help='Half betas.')

    p.add_option('--slp', action='store', type = 'string', dest='slp', default='none',
                 help='Train on SLP 2D targets.')

    p.add_option('--nosmpl', action='store_true', dest='nosmpl', default=False,
                 help='Remove SMPL from loss.')

    p.add_option('--v2v', action='store_true', dest='v2v', default=False,
                 help='Use a per vertex loss.')

    p.add_option('--train_only_betanet', action='store_true', dest='train_only_betanet', default=False,
                 help='Train only the betanet.')

    p.add_option('--train_only_CAL', action='store_true', dest='train_only_CAL', default=False,
                 help='Train only the adjust Q to P net.')

    p.add_option('--no_depthnoise', action='store_true', dest='no_depthnoise', default=False,
                 help='Apply depth calibration noise to the input to facilitate sim to real transfer.')

    p.add_option('--no_loss_htwt', action='store_true', dest='noloss_htwt', default=False,
                 help='Use root in loss function.')

    p.add_option('--no_loss_angles', action='store_true', dest='no_reg_angles', default=False,
                 help='Remove loss from joint angles.')

    p.add_option('--no_loss_root', action='store_true', dest='no_loss_root', default=False,
                 help='Use root in loss function.')

    p.add_option('--no_loss_betas', action='store_true', dest='no_loss_betas', default=False,
                 help='Compute a loss on the betas.')

    p.add_option('--no_blanket', action='store_true', dest='no_blanket', default=False,
                 help='Use blankets.')

    p.add_option('--slp_depth', action='store_true', dest='slp_depth', default=True,
                 help='Train for SLP dataset using camera positioned at their height above bed.')

    p.add_option('--verbose', '--v',  action='store_true', dest='verbose',
                 default=True, help='Printout everything (under construction).')

    p.add_option('--log_interval', type=int, default=200, metavar='N',
                 help='number of batches between logging train status') #if you visualize too often it will slow down training.
    return p