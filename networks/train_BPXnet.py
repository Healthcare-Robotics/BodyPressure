#!/usr/bin/env python
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import *

# PyTorch libraries
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable


try:
    import cPickle as pkl
    def load_pickle(filename):
        with open(filename, 'rb') as f:
            return pkl.load(f)
except:
    import pickle as pkl
    def load_pickle(filename):
        with open(filename, 'rb') as f:
            return pkl.load(f, encoding = 'latin1')


txtfile = open("../FILEPATH.txt")
FILEPATH = txtfile.read().replace("\n", "")
txtfile.close()


import sys
sys.path.insert(0, '../lib_py')
sys.path.insert(-1,FILEPATH+'smpl/smpl_webuser3')
sys.path.insert(-1,FILEPATH)
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
#sys.path.remove('/home/henry/git/pytorch_HMR/src')

print(sys.path)
#import chumpy as ch

import convnet_bp as convnet
import fixedwt_smpl_pmr_net as fixedwt_smpl_pmr
import fixedwt_smpl_pmr_net as fixedwt_smpl_pmr
import convnet_CAL_bp as convnet_CAL
import betanet_bp as betanet
# import tf.transformations as tft

# Pose Estimation Libraries
from visualization_lib_bp import VisualizationLib
from tensorprep_lib_bp import TensorPrepLib
from unpack_depth_batch_lib_bp import UnpackDepthBatchLib
from filename_input_lib_bp import FileNameInputLib
from slp_prep_lib_bp import SLPPrepLib


import random
from scipy import ndimage
import scipy.stats as ss

np.set_printoptions(threshold=sys.maxsize)

MAT_WIDTH = 0.762  # metres
MAT_HEIGHT = 1.854  # metres
MAT_HALF_WIDTH = MAT_WIDTH / 2
NUMOFTAXELS_X = 64  # 73 #taxels
NUMOFTAXELS_Y = 27  # 30
NUMOFOUTPUTDIMS = 3
NUMOFOUTPUTNODES_TRAIN = 24
NUMOFOUTPUTNODES_TEST = 10
INTER_SENSOR_DISTANCE = 0.0286  # metres

torch.set_num_threads(1)
if torch.cuda.is_available():
    # Use for GPU
    GPU = True
    dtype = torch.cuda.FloatTensor
    dtypeLong = torch.cuda.LongTensor
    print ('######################### CUDA is available! #############################')
else:
    # Use for CPU
    GPU = False
    dtype = torch.FloatTensor
    dtypeLong = torch.LongTensor
    print ('############################## USING CPU #################################')


class PhysicalTrainer():

    def __init__(self, train_files_f, train_files_m, test_files_f, test_files_m, opt):

        self.CTRL_PNL = {}
        if opt.train_only_CAL == True or opt.train_only_betanet == True:
            self.CTRL_PNL['num_epochs'] = 500
            opt.mod = 1
            opt.pmr = True
        elif opt.mod == 1:
            self.CTRL_PNL['num_epochs'] = 100
        elif opt.mod == 2:
            self.CTRL_PNL['num_epochs'] = 40

        self.opt = opt



        if opt.X_is == 'W':
            self.CTRL_PNL['CNN'] = 'resnet'
            self.CTRL_PNL['depth_out_unet'] = False
            self.CTRL_PNL['onlyhuman_labels'] = False
        elif opt.X_is == 'B':
            self.CTRL_PNL['CNN'] = 'resnetunet'
            self.CTRL_PNL['depth_out_unet'] = True
            self.CTRL_PNL['onlyhuman_labels'] = True
        else:
            print('you need to select a valid X_is. choose "W" for white box net or "B" for black box net.')
            sys.exit()


        self.CTRL_PNL['mod'] = opt.mod
        self.CTRL_PNL['nosmpl'] = opt.nosmpl
        self.CTRL_PNL['loss_vector_type'] = opt.losstype
        self.CTRL_PNL['verbose'] = opt.verbose
        self.CTRL_PNL['batch_size'] = 128
        self.CTRL_PNL['incl_inter'] = True
        self.CTRL_PNL['shuffle'] = True
        self.CTRL_PNL['double_network_size'] = False
        self.CTRL_PNL['incl_ht_wt_channels'] = opt.htwt
        self.CTRL_PNL['pimg_cntct_sum'] = opt.pimg_cntct_sum
        self.CTRL_PNL['omit_pimg_cntct_sobel'] = opt.omit_pimg_cntct_sobel
        self.CTRL_PNL['incl_pmat_cntct_input'] = False
        self.CTRL_PNL['lock_root'] = False
        self.CTRL_PNL['num_input_channels'] = 1
        self.CTRL_PNL['GPU'] = GPU
        self.CTRL_PNL['dtype'] = dtype
        self.CTRL_PNL['dtypeLong'] = dtypeLong
        if opt.no_loss_root == True: self.CTRL_PNL['loss_root'] = False
        else: self.CTRL_PNL['loss_root'] = True
        if opt.no_reg_angles == True: self.CTRL_PNL['regr_angles'] = False
        else: self.CTRL_PNL['regr_angles'] = True
        if opt.no_loss_betas == True: self.CTRL_PNL['loss_betas'] = False
        else: self.CTRL_PNL['loss_betas'] = True
        if opt.no_depthnoise == True: self.CTRL_PNL['depth_noise'] = False
        else: self.CTRL_PNL['depth_noise'] = True
        self.CTRL_PNL['noloss_htwt'] = opt.noloss_htwt
        self.CTRL_PNL['mesh_recon_map_labels'] = opt.pmr #can only be true if we have 100% synthetic data for training
        self.CTRL_PNL['mesh_recon_map_labels_test'] = opt.pmr #False #can only be true is we have 100% synth for testing
        self.CTRL_PNL['mesh_recon_map_output'] = self.CTRL_PNL['mesh_recon_map_labels']
        self.CTRL_PNL['mesh_recon_output'] = self.CTRL_PNL['mesh_recon_map_output']
        if opt.v2v == True: self.CTRL_PNL['mesh_recon_output'] = True
        if opt.mod == 1:
            self.CTRL_PNL['adjust_ang_from_est'] = False #starts angles from scratch
            self.CTRL_PNL['recon_map_input_est'] = False #do this if we're working in a two-part regression
        elif opt.mod == 2:
            self.CTRL_PNL['adjust_ang_from_est'] = True #gets betas and angles from prior estimate
            self.CTRL_PNL['recon_map_input_est'] = True #do this if we're working in a two-part regression
        self.CTRL_PNL['clip_sobel'] = True
        self.CTRL_PNL['clip_betas'] = True
        self.CTRL_PNL['full_body_rot'] = True
        self.CTRL_PNL['normalize_per_image'] = False
        self.CTRL_PNL['normalize_std'] = False
        self.CTRL_PNL['all_tanh_activ'] = True
        self.CTRL_PNL['slp_noise'] = opt.slpnoise
        self.CTRL_PNL['first_pass'] = True
        self.CTRL_PNL['align_procr'] = False
        self.CTRL_PNL['depth_in'] = True
        self.CTRL_PNL['slp'] = opt.slp
        self.CTRL_PNL['clean_slp_depth'] = False
        self.CTRL_PNL['train_only_betanet'] = opt.train_only_betanet
        self.CTRL_PNL['train_only_CAL'] = opt.train_only_CAL
        self.CTRL_PNL['compute_forward_maps'] = False
        self.CTRL_PNL['v2v'] = opt.v2v
        self.CTRL_PNL['x_y_offset_synth'] = [12, -35]


        if GPU == True:
            torch.cuda.set_device(self.opt.device)

        self.weight_joints = 1.0#self.opt.j_d_ratio*2
        self.weight_depth_planes = (1-self.opt.j_d_ratio)#*2



        if self.CTRL_PNL['recon_map_input_est'] == True: #for a two part regression
            self.CTRL_PNL['num_input_channels'] += 2
        self.CTRL_PNL['num_input_channels_batch0'] = np.copy(self.CTRL_PNL['num_input_channels'])

        pmat_std_from_mult = ['N/A', 14.64204661, 19.90905848383454, 23.07018866032369, 0.0, 25.50538629767412]

        self.CTRL_PNL['norm_std_coeffs'] =  [1./41.80684362163343,  #contact
                                             1./45.08513083167194,  #neg est depth
                                             1./43.55800622930469,  #cm est
                                             1./pmat_std_from_mult[int(1)], #pmat x5
                                             1./1.0, #pmat sobel
                                             1./1.0,                #OUTPUT DO NOTHING
                                             1./1.0,                #OUTPUT DO NOTHING
                                             1. / 30.216647403350,  #weight
                                             1. / 14.629298141231]  #height

        if self.CTRL_PNL['normalize_std'] == False:
            for i in range(9):
                self.CTRL_PNL['norm_std_coeffs'][i] *= 0.
                self.CTRL_PNL['norm_std_coeffs'][i] += 1.


        self.CTRL_PNL['convnet_fp_prefix'] = FILEPATH + 'data_BP/convnets/'

        if self.CTRL_PNL['mesh_recon_output'] == True: #we need all the vertices if we're going to regress the depth maps
            self.verts_list = "all"
        else:
            self.verts_list = [1325, 336, 1032, 4515, 1374, 4848, 1739, 5209, 1960, 5423]



        self.v2v_zeros = Variable(torch.Tensor(np.zeros((self.CTRL_PNL['batch_size'], 6890))).type(dtype), requires_grad=True)
        self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)
        self.output_size_train = (NUMOFOUTPUTNODES_TRAIN, NUMOFOUTPUTDIMS)
        self.output_size_val = (NUMOFOUTPUTNODES_TEST, NUMOFOUTPUTDIMS)
        self.parents = np.array([4294967295, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]).astype(np.int32)

        #python3.6 train_depthnet.py --depthnoise --slp 'synth' --rgangs --loss_root --loss_betas --viz --log_interval 200 --blanket --slp_depth --depth_out_unet --cnn 'resnetunet'
        #################################### PREP TRAINING DATA ##########################################

        x_map_ct = 5
        pmat_gt_idx = 0
        depth_in_idx = 1
        recon_gt_idx = 1
        if self.CTRL_PNL['recon_map_input_est'] == True:
            x_map_ct += 2
            pmat_gt_idx += 2
            depth_in_idx += 2
            recon_gt_idx += 2
        if self.CTRL_PNL['mesh_recon_map_labels'] == True:
            x_map_ct += 2
            depth_in_idx += 2
        if self.CTRL_PNL['depth_out_unet'] == True:
            x_map_ct += 4
        if self.CTRL_PNL['depth_out_unet'] == True and self.CTRL_PNL['mod'] == 2:
            x_map_ct += 4


        if self.opt.quick_test == True:
            if self.opt.slp == 'synth':
                train_x = np.zeros((17042, x_map_ct, 64, 27)).astype(np.float32)
            elif self.opt.slp == 'real':
                train_x = np.zeros((135*1, x_map_ct, 64, 27)).astype(np.float32)
        else:
            if self.opt.slp == 'synth':
                #train_x = np.zeros((85114, x_map_ct, 64, 27)).astype(np.float32)
                train_x = np.zeros((97495, x_map_ct, 64, 27)).astype(np.float32)
            elif self.opt.slp == 'real':
                #train_x = np.zeros((135*69, x_map_ct, 64, 27)).astype(np.float32)
                train_x = np.zeros((135*79, x_map_ct, 64, 27)).astype(np.float32)
            elif self.opt.slp == 'mixedreal':
                #train_x = np.zeros((85114+135*69, x_map_ct, 64, 27)).astype(np.float32)
                train_x = np.zeros((97495+135*79, x_map_ct, 64, 27)).astype(np.float32)


        #load training ysnth data
        if opt.small == True and opt.mod == 1:
            reduce_data = True
        else:
            reduce_data = False


        if self.opt.slp == 'real' or self.opt.slp == 'mixedreal':
            dat_f_real = {}
            dat_f_real_u = SLPPrepLib().load_slp_files_to_database(train_files_f[0], dana_lab_path, PM='uncover', depth='uncover', mass_ht_list = train_subj_mass_list_f, markers_gt_type = '3D', use_pc = False, depth_out_unet = self.CTRL_PNL['depth_out_unet'], pm_adjust_mm = self.CTRL_PNL['x_y_offset_synth'])
            dat_f_real_c1 = SLPPrepLib().load_slp_files_to_database(train_files_f[0], dana_lab_path, PM='cover1', depth='cover1', mass_ht_list = train_subj_mass_list_f, markers_gt_type = '3D', use_pc = False, depth_out_unet = self.CTRL_PNL['depth_out_unet'], pm_adjust_mm = self.CTRL_PNL['x_y_offset_synth'])
            dat_f_real_c2 = SLPPrepLib().load_slp_files_to_database(train_files_f[0], dana_lab_path, PM='cover2', depth='cover2', mass_ht_list = train_subj_mass_list_f, markers_gt_type = '3D', use_pc = False, depth_out_unet = self.CTRL_PNL['depth_out_unet'], pm_adjust_mm = self.CTRL_PNL['x_y_offset_synth'])
            for item in dat_f_real_c1:
                dat_f_real[item] = []
                for i in range(len(dat_f_real_c1[item])): #assign 45 per subject
                    try:
                        dat_f_real[item].append(dat_f_real_u[item][i])
                    except:
                        dat_f_real[item].append(dat_f_real_u['overhead_depthcam_noblanket'][i])
                for i in range(len(dat_f_real_c1[item])):  #assign 45 per subject
                    dat_f_real[item].append(dat_f_real_c1[item][i])
                for i in range(len(dat_f_real_c2[item])):  #assign 45 per subject
                    dat_f_real[item].append(dat_f_real_c2[item][i])

            if self.opt.pmr == True:
                dat_f_real = SLPPrepLib().load_slp_gt_maps_est_maps(train_files_f[0], dat_f_real, FileNameInputLib1.data_fp_suffix, depth_out_unet = self.CTRL_PNL['depth_out_unet'])

            dat_m_real = {}
            dat_m_real_u = SLPPrepLib().load_slp_files_to_database(train_files_m[0], dana_lab_path, PM='uncover', depth='uncover', mass_ht_list = train_subj_mass_list_m, markers_gt_type = '3D', use_pc = False, depth_out_unet = self.CTRL_PNL['depth_out_unet'], pm_adjust_mm = self.CTRL_PNL['x_y_offset_synth'])
            dat_m_real_c1 = SLPPrepLib().load_slp_files_to_database(train_files_m[0], dana_lab_path, PM='cover1', depth='cover1', mass_ht_list = train_subj_mass_list_m, markers_gt_type = '3D', use_pc = False, depth_out_unet = self.CTRL_PNL['depth_out_unet'], pm_adjust_mm = self.CTRL_PNL['x_y_offset_synth'])
            dat_m_real_c2 = SLPPrepLib().load_slp_files_to_database(train_files_m[0], dana_lab_path, PM='cover2', depth='cover2', mass_ht_list = train_subj_mass_list_m, markers_gt_type = '3D', use_pc = False, depth_out_unet = self.CTRL_PNL['depth_out_unet'], pm_adjust_mm = self.CTRL_PNL['x_y_offset_synth'])
            for item in dat_m_real_c1:
                dat_m_real[item] = []
                for i in range(len(dat_m_real_c1[item])): #assign 45 per subject
                    try:
                        dat_m_real[item].append(dat_m_real_u[item][i])
                    except:
                        dat_m_real[item].append(dat_m_real_u['overhead_depthcam_noblanket'][i])
                for i in range(len(dat_m_real_c1[item])): #assign 45 per subject
                    dat_m_real[item].append(dat_m_real_c1[item][i])
                for i in range(len(dat_m_real_c1[item])): #assign 45 per subject
                    dat_m_real[item].append(dat_m_real_c2[item][i])

            if self.opt.pmr == True:
                dat_m_real = SLPPrepLib().load_slp_gt_maps_est_maps(train_files_m[0], dat_m_real, FileNameInputLib1.data_fp_suffix, depth_out_unet = self.CTRL_PNL['depth_out_unet'])

        else:
            dat_f_real = None
            dat_m_real = None

        if self.opt.slp == 'real':
            dat_f_synth = None
            dat_m_synth = None
        else:
            dat_f_synth = TensorPrepLib().load_files_to_database(train_files_f[1:2]+train_files_f[3:4], creation_type = 'synth', reduce_data = reduce_data, depth_in = True)
            dat_m_synth = TensorPrepLib().load_files_to_database(train_files_m[1:2]+train_files_m[3:4], creation_type = 'synth', reduce_data = reduce_data, depth_in = True)

        #allocate pressure images
        train_x = TensorPrepLib().prep_images(train_x, dat_f_real, dat_m_real, dat_f_synth, dat_m_synth, filter_sigma = 0.5, start_map_idx = pmat_gt_idx)



        if self.CTRL_PNL['mesh_recon_map_labels'] == True:
            #Initialize the precomputed depth and contact maps.
            train_x = TensorPrepLib().prep_reconstruction_gt(train_x, dat_f_real, dat_m_real, dat_f_synth, dat_m_synth, start_map_idx = recon_gt_idx)

        if self.CTRL_PNL['recon_map_input_est'] == True:
            #Initialize the precomputed depth and contact map input estimates
            train_x = TensorPrepLib().prep_reconstruction_input_est(train_x, dat_f_real, dat_m_real, dat_f_synth, dat_m_synth, start_map_idx = 0, cnn_type = self.CTRL_PNL['CNN'])



        im_ct = 0
        if self.opt.slp == 'real' or self.opt.slp == 'mixedreal':
            train_x = TensorPrepLib().prep_depth_input_images(train_x, dat_f_real, dat_m_real, None, None, start_map_idx = depth_in_idx, depth_type = 'all_meshes')
            im_ct = len(dat_f_real['images']) + len(dat_m_real['images'])
        if self.opt.slp != 'real':
            if self.opt.no_blanket == False:
               train_x = TensorPrepLib().prep_depth_input_images(train_x, None, None, dat_f_synth, dat_m_synth, start_map_idx = depth_in_idx, depth_type = 'all_meshes', mix_bl_nobl = True, im_ct = im_ct) #'all_meshes')#'
            else:
               train_x = TensorPrepLib().prep_depth_input_images(train_x, None, None, dat_f_synth, dat_m_synth, start_map_idx = depth_in_idx, depth_type = 'no_blanket', mix_bl_nobl = False, im_ct = im_ct) #'all_meshes')#'


        if self.CTRL_PNL['depth_out_unet'] == True:
            train_x = TensorPrepLib().prep_depth_input_images(train_x, dat_f_real, dat_m_real, dat_f_synth, dat_m_synth, start_map_idx = depth_in_idx+4, depth_type = 'human_only')

        if self.CTRL_PNL['depth_out_unet'] == True and self.CTRL_PNL['mod'] == 2:
            train_x = TensorPrepLib().prep_depth_input_images(train_x, dat_f_real, dat_m_real, dat_f_synth, dat_m_synth, start_map_idx = depth_in_idx+8, depth_type = 'unet_est')

        for i in range(x_map_ct):
            print(i, np.max(train_x[:, i, :, :]), np.std(train_x[:, i, :, :]), train_x.dtype)

        self.train_x_tensor = torch.Tensor(train_x) #this converts the int16/short array to a float32 tensor. idk how to fix this yet.




        train_y_flat = []  # Initialize the training ground truth list
        if self.opt.slp == 'real' or self.opt.slp == 'mixedreal':
            for gender_synth in [["f", dat_f_real], ["m", dat_m_real]]:
                train_y_flat = SLPPrepLib().prep_labels_slp(train_y_flat, gender_synth[1], num_repeats = 1,
                                                                z_adj = -0.075, gender = gender_synth[0], is_synth = True,
                                                                markers_gt_type = '3D',
                                                                initial_angle_est = self.CTRL_PNL['adjust_ang_from_est'],
                                                                cnn_type = self.CTRL_PNL['CNN'])
        if self.opt.slp != 'real':
            for gender_synth in [["f", dat_f_synth], ["m", dat_m_synth]]:
                train_y_flat = TensorPrepLib().prep_labels(train_y_flat, gender_synth[1],
                                                                z_adj = -0.075, gender = gender_synth[0], is_synth = True,
                                                                loss_vector_type = self.CTRL_PNL['loss_vector_type'],
                                                                initial_angle_est = self.CTRL_PNL['adjust_ang_from_est'],
                                                                cnn_type = self.CTRL_PNL['CNN'], x_y_adjust_mm = self.CTRL_PNL['x_y_offset_synth'])
            del dat_f_synth
            del dat_m_synth



        train_y_flat = np.array(train_y_flat)

        self.train_y_tensor = torch.Tensor(train_y_flat)






        #################################### PREP TESTING DATA ##########################################
        # load in the test file
        test_dat_f_real = {}
        test_dat_f_real_u = SLPPrepLib().load_slp_files_to_database(test_files_f[0], dana_lab_path, PM='uncover', depth='uncover', mass_ht_list = test_subj_mass_list_f, markers_gt_type = '3D', use_pc = False, depth_out_unet = self.CTRL_PNL['depth_out_unet'], pm_adjust_mm = self.CTRL_PNL['x_y_offset_synth'])
        test_dat_f_real_c1 = SLPPrepLib().load_slp_files_to_database(test_files_f[0], dana_lab_path, PM='cover1', depth='cover1', mass_ht_list = test_subj_mass_list_f, markers_gt_type = '3D', use_pc = False, depth_out_unet = self.CTRL_PNL['depth_out_unet'], pm_adjust_mm = self.CTRL_PNL['x_y_offset_synth'])
        test_dat_f_real_c2 = SLPPrepLib().load_slp_files_to_database(test_files_f[0], dana_lab_path, PM='cover2', depth='cover2', mass_ht_list = test_subj_mass_list_f, markers_gt_type = '3D', use_pc = False, depth_out_unet = self.CTRL_PNL['depth_out_unet'], pm_adjust_mm = self.CTRL_PNL['x_y_offset_synth'])
        for item in test_dat_f_real_c1:
            test_dat_f_real[item] = []
            for i in range(len(test_dat_f_real_c1[item])): #assign 45 per subject
                try:
                    test_dat_f_real[item].append(test_dat_f_real_u[item][i])
                except:
                    test_dat_f_real[item].append(test_dat_f_real_u['overhead_depthcam_noblanket'][i])
            for i in range(len(test_dat_f_real_c1[item])): #assign 45 per subject
                test_dat_f_real[item].append(test_dat_f_real_c1[item][i])
            for i in range(len(test_dat_f_real_c2[item])): #assign 45 per subject
                test_dat_f_real[item].append(test_dat_f_real_c2[item][i])

        if self.opt.pmr == True:
            test_dat_f_real = SLPPrepLib().load_slp_gt_maps_est_maps(test_files_f[0], test_dat_f_real, FileNameInputLib1.data_fp_suffix, depth_out_unet = self.CTRL_PNL['depth_out_unet'])

        test_dat_m_real = {}
        test_dat_m_real_u = SLPPrepLib().load_slp_files_to_database(test_files_m[0], dana_lab_path, PM='uncover', depth='uncover', mass_ht_list = test_subj_mass_list_m, markers_gt_type = '3D', use_pc = False, depth_out_unet = self.CTRL_PNL['depth_out_unet'], pm_adjust_mm = self.CTRL_PNL['x_y_offset_synth'])
        test_dat_m_real_c1 = SLPPrepLib().load_slp_files_to_database(test_files_m[0], dana_lab_path, PM='cover1', depth='cover1', mass_ht_list = test_subj_mass_list_m, markers_gt_type = '3D', use_pc = False, depth_out_unet = self.CTRL_PNL['depth_out_unet'], pm_adjust_mm = self.CTRL_PNL['x_y_offset_synth'])
        test_dat_m_real_c2 = SLPPrepLib().load_slp_files_to_database(test_files_m[0], dana_lab_path, PM='cover2', depth='cover2', mass_ht_list = test_subj_mass_list_m, markers_gt_type = '3D', use_pc = False, depth_out_unet = self.CTRL_PNL['depth_out_unet'], pm_adjust_mm = self.CTRL_PNL['x_y_offset_synth'])
        for item in test_dat_m_real_c1:
            test_dat_m_real[item] = []
            for i in range(len(test_dat_m_real_c1[item])): #assign 45 per subject
                try:
                    test_dat_m_real[item].append(test_dat_m_real_u[item][i])
                except:
                    test_dat_m_real[item].append(test_dat_m_real_u['overhead_depthcam_noblanket'][i])
            for i in range(len(test_dat_m_real_c1[item])): #assign 45 per subject
                test_dat_m_real[item].append(test_dat_m_real_c1[item][i])
            for i in range(len(test_dat_m_real_c1[item])): #assign 45 per subject
                test_dat_m_real[item].append(test_dat_m_real_c2[item][i])

        if self.opt.pmr == True:
            test_dat_m_real = SLPPrepLib().load_slp_gt_maps_est_maps(test_files_m[0], test_dat_m_real, FileNameInputLib1.data_fp_suffix, depth_out_unet = self.CTRL_PNL['depth_out_unet'])



        test_dat_f_synth = None
        test_dat_m_synth = None




        if self.opt.quick_test == True:
            test_x = np.zeros((135*1, x_map_ct, 64, 27)).astype(np.float32)
        else:
            #test_x = np.zeros((135*10, x_map_ct, 64, 27)).astype(np.float32)
            test_x = np.zeros((135*22, x_map_ct, 64, 27)).astype(np.float32)

        #allocate pressure images
        test_x = TensorPrepLib().prep_images(test_x, test_dat_f_real, test_dat_m_real, test_dat_f_synth, test_dat_m_synth, filter_sigma = 0.5, start_map_idx = pmat_gt_idx)


        if self.CTRL_PNL['mesh_recon_map_labels'] == True:
            #Initialize the precomputed depth and contact maps.
            test_x = TensorPrepLib().prep_reconstruction_gt(test_x, test_dat_f_real, test_dat_m_real, test_dat_f_synth, test_dat_m_synth, start_map_idx = recon_gt_idx)

        if self.CTRL_PNL['recon_map_input_est'] == True:
            #Initialize the precomputed depth and contact map input estimates
            test_x = TensorPrepLib().prep_reconstruction_input_est(test_x, test_dat_f_real, test_dat_m_real, test_dat_f_synth, test_dat_m_synth, start_map_idx = 0, cnn_type = self.CTRL_PNL['CNN'])


        im_ct = 0
        if self.opt.slp == 'real' or self.opt.slp == 'mixedreal':
            test_x = TensorPrepLib().prep_depth_input_images(test_x, test_dat_f_real, test_dat_m_real, None, None, start_map_idx = depth_in_idx, depth_type = 'all_meshes')
            im_ct = len(test_dat_f_real['images']) + len(test_dat_m_real['images'])
        if self.opt.slp != 'real':
            if self.opt.no_blanket == False:
               test_x = TensorPrepLib().prep_depth_input_images(test_x, None, None, test_dat_f_synth, test_dat_m_synth, start_map_idx = depth_in_idx, depth_type = 'all_meshes', mix_bl_nobl = True, im_ct = im_ct) #'all_meshes')#'
            else:
               test_x = TensorPrepLib().prep_depth_input_images(test_x, None, None, test_dat_f_synth, test_dat_m_synth, start_map_idx = depth_in_idx, depth_type = 'no_blanket', mix_bl_nobl = False, im_ct = im_ct) #'all_meshes')#'


        if self.CTRL_PNL['depth_out_unet'] == True:
            test_x = TensorPrepLib().prep_depth_input_images(test_x, test_dat_f_real, test_dat_m_real, test_dat_f_synth, test_dat_m_synth, start_map_idx = depth_in_idx+4, depth_type = 'human_only')

        if self.CTRL_PNL['depth_out_unet'] == True and self.CTRL_PNL['mod'] == 2:
            test_x = TensorPrepLib().prep_depth_input_images(test_x, test_dat_f_real, test_dat_m_real, test_dat_f_synth, test_dat_m_synth, start_map_idx = depth_in_idx+8, depth_type = 'unet_est')


        self.test_x_tensor = torch.Tensor(test_x)


        test_y_flat = []  # Initialize the ground truth listhave
        #if self.opt.slp == 'real' or self.opt.slp == 'mixedreal':
        for gender_synth in [["f", test_dat_f_real], ["m", test_dat_m_real]]:
            test_y_flat = SLPPrepLib().prep_labels_slp(test_y_flat, gender_synth[1], num_repeats = 1,
                                                            z_adj = -0.075, gender = gender_synth[0], is_synth = True,
                                                            markers_gt_type = '3D',
                                                            initial_angle_est = self.CTRL_PNL['adjust_ang_from_est'],
                                                            cnn_type = self.CTRL_PNL['CNN'])


        if self.opt.slp != 'real':
            for gender_synth in [["f", test_dat_f_synth], ["m", test_dat_m_synth]]:
                test_y_flat = TensorPrepLib().prep_labels(test_y_flat, gender_synth[1],
                                                            z_adj = -0.075, gender = gender_synth[0], is_synth = True,
                                                            loss_vector_type = self.CTRL_PNL['loss_vector_type'],
                                                            initial_angle_est = self.CTRL_PNL['adjust_ang_from_est'],
                                                            cnn_type = self.CTRL_PNL['CNN'], x_y_adjust_mm = self.CTRL_PNL['x_y_offset_synth'])#[14, 10])
            del test_dat_f_synth
            del test_dat_m_synth

        test_y_flat = np.array(test_y_flat)

        self.test_y_tensor = torch.Tensor(test_y_flat)

        self.save_name = '_' + str(opt.mod) + '_' + opt.losstype + \
                         '_' + str(self.train_x_tensor.size()[0]) + 'ct' + \
                         '_' + str(self.CTRL_PNL['batch_size']) + 'b' + \
                         '_x' + str(1) + 'pm'


        if self.CTRL_PNL['mesh_recon_map_labels'] == True:
            self.save_name += '_' + str(self.opt.j_d_ratio) + 'rtojtdpth'
        if self.CTRL_PNL['recon_map_input_est'] == True:
            self.save_name += '_depthestin'
        if self.CTRL_PNL['adjust_ang_from_est'] == True:
            self.save_name += '_angleadj'
        if self.CTRL_PNL['regr_angles'] == True:
            self.save_name += '_rgangs'
        if self.CTRL_PNL['loss_betas'] == True:
            self.save_name += '_lb'
        if self.CTRL_PNL['v2v'] == True:
            self.save_name += '_lv2v'
        if self.CTRL_PNL['noloss_htwt'] == True:
            self.save_name += '_nlhw'

        if self.opt.slp_depth == True or self.opt.slp == "real":
            if self.opt.no_blanket == True:
                self.save_name += '_slpnb'
            else:
                self.save_name += '_slpb'
        else:
            self.save_name += '_ppnb'


        if self.CTRL_PNL['incl_ht_wt_channels'] == True:
            self.save_name += '_htwt'
        if self.CTRL_PNL['depth_noise'] == True:
            self.save_name += '_dpns'
        if self.CTRL_PNL['slp_noise'] == True:
            self.save_name += '_slpns'

        if  self.CTRL_PNL['loss_root'] == True:
            self.save_name += '_rt'
        if  self.CTRL_PNL['pimg_cntct_sum'] == True:
            self.save_name += '_pcsum'
        if  self.CTRL_PNL['omit_pimg_cntct_sobel'] == True:
            self.save_name += '_opcs'
        if  self.opt.half_shape_wt == True:
            self.save_name += '_hsw'
        if  self.CTRL_PNL['depth_out_unet'] == True:
            self.save_name += '_dou'

        print ('appending to', 'train' + self.save_name)
        self.train_val_losses = {}
        self.train_val_losses['loss_eucl'] = []
        self.train_val_losses['loss_betas'] = []
        self.train_val_losses['loss_angs'] = []
        self.train_val_losses['loss_v2v'] = []
        self.train_val_losses['loss_bedht'] = []
        self.train_val_losses['loss_bodymass'] = []
        self.train_val_losses['loss_bodyheight'] = []
        self.train_val_losses['loss_bodyrot'] = []
        self.train_val_losses['loss_meshdepth'] = []
        self.train_val_losses['loss_meshcntct'] = []
        self.train_val_losses['train_loss'] = []
        self.train_val_losses['val_loss'] = []
        self.train_val_losses['epoch_ct'] = []
        if self.CTRL_PNL['depth_out_unet'] == True:
            self.train_val_losses['unet_loss'] = []





    def init_convnet_train(self):

        print (self.train_x_tensor.shape, 'Input training tensor shape')
        print (self.train_y_tensor.shape, 'Output training tensor shape')

        print (self.test_x_tensor.shape, 'Input testing tensor shape')
        print (self.test_y_tensor.shape, 'Output testing tensor shape')

        self.train_dataset = torch.utils.data.TensorDataset(self.train_x_tensor, self.train_y_tensor)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, self.CTRL_PNL['batch_size'], shuffle=self.CTRL_PNL['shuffle'])

        self.test_dataset = torch.utils.data.TensorDataset(self.test_x_tensor, self.test_y_tensor)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, self.CTRL_PNL['batch_size'], shuffle=self.CTRL_PNL['shuffle'])



        print ("Loading convnet model................................")

        if self.opt.slp == 'real' and self.opt.losstype == 'direct':
            fc_output_size = 28
        else:
            fc_output_size = 89## 10 + 3 + 24*3 --- betas, root shift, rotations


        if self.CTRL_PNL['train_only_betanet'] == False and self.CTRL_PNL['train_only_CAL'] == False:
            self.model = convnet.CNN(fc_output_size, self.CTRL_PNL['loss_vector_type'], in_channels=self.CTRL_PNL['num_input_channels'], CTRL_PNL = self.CTRL_PNL)
            #self.model = torch.load(self.CTRL_PNL['convnet_fp_prefix'] + 'resnet34_2_anglesDC_108160ct_128b_x1pm_0.5rtojtdpth_depthestin_angleadj_rgangs_lb_lv2v_slpb_dpns_rt_40e_0.0001lr.pt',map_location={'cuda:' + str(self.opt.prev_device): 'cuda:' + str(self.opt.device)})
            self.model_smpl_pmr = fixedwt_smpl_pmr.SMPL_PMR(self.CTRL_PNL['loss_vector_type'], self.CTRL_PNL['batch_size'], verts_list = self.verts_list, CTRL_PNL = self.CTRL_PNL)
        else:
            self.model = None
            self.model_smpl_pmr = None


        if self.CTRL_PNL['CNN'] == 'resnet' and self.CTRL_PNL['train_only_betanet'] == False:
            if self.CTRL_PNL['train_only_CAL'] == True:
                self.model_CAL = convnet_CAL.CNN(CTRL_PNL = self.CTRL_PNL)
            else:
                self.model_CAL = torch.load(self.CTRL_PNL['convnet_fp_prefix'] + 'CAL_10665ct_128b_500e_0.0001lr.pt',map_location={'cuda:' + str(self.opt.prev_device): 'cuda:' + str(self.opt.device)})
        else:
            self.model_CAL = None

        if self.CTRL_PNL['train_only_CAL'] == False:
            if self.CTRL_PNL['train_only_betanet'] == True:
                self.model_betanet = betanet.FC(CTRL_PNL = self.CTRL_PNL)
            else:
                self.model_betanet = torch.load(self.CTRL_PNL['convnet_fp_prefix'] + 'betanet_108160ct_128b_volfrac_500e_0.0001lr.pt', map_location={'cuda:' + str(self.opt.prev_device): 'cuda:' + str(self.opt.device)})
            print('loaded betanet')
        else:
            self.model_betanet = None


        learning_rate = 0.0001#0.0001

        # Run model on GPU if available
        if self.CTRL_PNL['train_only_betanet'] == False and self.CTRL_PNL['train_only_CAL'] == False:
            if GPU == True: self.model = self.model.cuda()
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=0.0005) #start with .00005
        if self.CTRL_PNL['CNN'] == 'resnet' and self.CTRL_PNL['train_only_betanet'] == False:
            if GPU == True: self.model_CAL = self.model_CAL.cuda()
            self.optimizer_CAL = optim.Adam(self.model_CAL.parameters(), lr=learning_rate, weight_decay=0.0005) #start with .00005
        if self.CTRL_PNL['CNN'] == 'resnet' and self.CTRL_PNL['train_only_CAL'] == False:
            if GPU == True: self.model_betanet = self.model_betanet.cuda()
            self.optimizer_betanet = optim.Adam(self.model_betanet.parameters(), lr=learning_rate, weight_decay=0.0005) #start with .00005


        # train the model one epoch at a time
        for epoch in range(1, self.CTRL_PNL['num_epochs'] + 1):

            self.t1 = time.time()
            self.train_convnet(epoch)

            try:
                self.t2 = time.time() - self.t1
            except:
                self.t2 = 0
            print ('Time taken by epoch',epoch,':',self.t2,' seconds')

            if epoch == self.CTRL_PNL['num_epochs'] or epoch == 40 or epoch == 100 or epoch == 140 or epoch == 200 or epoch == 500 or epoch ==300:# or epoch == 50 or epoch == 60 or epoch == 70 or epoch == 80 or epoch == 90:
            #if epoch == self.CTRL_PNL['num_epochs'] or epoch == 20 or epoch == 25 or epoch == 30 or epoch == 35 or epoch == 40 or epoch == 45 or epoch == 50 or epoch == 55 or epoch == 60 or epoch == 65 or epoch == 75 or epoch == 80 or epoch == 90 or  epoch == 100:# or epoch == 50 or epoch == 60 or epoch == 70 or epoch == 80 or epoch == 90:


                if self.opt.go200 == True:
                    epoch_log = epoch + 100
                else:
                    epoch_log = epoch + 0

                if self.CTRL_PNL['CNN'] == 'resnet':
                    cnn_name = 'resnet34'
                elif self.CTRL_PNL['CNN'] == 'resnetunet':
                    cnn_name = 'resnetunet34'

                if self.CTRL_PNL['train_only_betanet'] == False and self.CTRL_PNL['train_only_CAL'] == False:
                    torch.save(self.model, self.CTRL_PNL['convnet_fp_prefix']+cnn_name+self.save_name+'_'+str(epoch_log)+'e'+'_'+str(learning_rate)+'lr.pt')
                    pkl.dump(self.train_val_losses,open(self.CTRL_PNL['convnet_fp_prefix']+cnn_name+'_losses'+self.save_name+'_'+str(epoch_log)+'e_'+str(learning_rate)+'lr.p', 'wb'))
                if self.CTRL_PNL['train_only_CAL'] == True:
                    torch.save(self.model_CAL, self.CTRL_PNL['convnet_fp_prefix']+'CAL_'+str(self.train_x_tensor.size()[0]) + 'ct_'+ str(self.CTRL_PNL['batch_size']) + 'b_' +str(epoch_log)+'e_'+str(learning_rate)+'lr.pt')
                    pkl.dump(self.train_val_losses,open(self.CTRL_PNL['convnet_fp_prefix']+'CAL_losses_'+str(self.train_x_tensor.size()[0]) + 'ct_'+ str(self.CTRL_PNL['batch_size']) + 'b_' +str(epoch_log)+'e_'+str(learning_rate)+'lr.p', 'wb'))
                if self.CTRL_PNL['train_only_betanet'] == True:
                    torch.save(self.model_betanet, self.CTRL_PNL['convnet_fp_prefix']+'betanet_'+str(self.train_x_tensor.size()[0]) + 'ct_'+str(self.CTRL_PNL['batch_size']) + 'b_' +str(epoch_log)+'e_'+str(learning_rate)+'lr.pt')
                    pkl.dump(self.train_val_losses,open(self.CTRL_PNL['convnet_fp_prefix']+'betanet_losses_'+str(self.train_x_tensor.size()[0]) + 'ct_'+str(self.CTRL_PNL['batch_size']) + 'b_' +str(epoch_log)+'e_'+str(learning_rate)+'lr.p', 'wb'))

        print (self.train_val_losses, 'trainval')
        # Save the model (architecture and weights)




    def train_convnet(self, epoch):

        self.criterion = nn.L1Loss()
        self.criterion2 = nn.MSELoss()


        with torch.autograd.set_detect_anomaly(True):

            # This will loop a total = training_images/batch_size times
            for batch_idx, batch in enumerate(self.train_loader):
                if self.CTRL_PNL['train_only_betanet'] == False and self.CTRL_PNL['train_only_CAL'] == False:
                    self.model.train()
                    self.optimizer.zero_grad()
                if self.CTRL_PNL['CNN'] == 'resnet' and self.CTRL_PNL['train_only_betanet'] == False:
                    self.model_CAL.train()
                    self.optimizer_CAL.zero_grad()
                if self.CTRL_PNL['CNN'] == 'resnet' and self.CTRL_PNL['train_only_CAL'] == False:
                    self.model_betanet.train()
                    self.optimizer_betanet.zero_grad()


                scores, INPUT_DICT, OUTPUT_DICT = \
                    UnpackDepthBatchLib().unpack_batch(batch, is_training=True, model = self.model, model_smpl_pmr = self.model_smpl_pmr, \
                                                       model_CAL = self.model_CAL, model_betanet = self.model_betanet, CTRL_PNL=self.CTRL_PNL)
                self.CTRL_PNL['first_pass'] = False


                loss = 0

                if self.CTRL_PNL['slp'] == 'real' and self.CTRL_PNL['loss_vector_type'] == 'direct':
                    scores_zeros = Variable(torch.Tensor(np.zeros((batch[0].shape[0], scores.size()[1]))).type(dtype), requires_grad=True)
                    loss += self.criterion(scores[:, 0:14], scores_zeros[:, 0:14])

                elif self.CTRL_PNL['train_only_CAL'] == False and self.CTRL_PNL['train_only_betanet'] == False:
                    scores_zeros = Variable(torch.Tensor(np.zeros((batch[0].shape[0], scores.size()[1]))).type(dtype), requires_grad=True)
                    OSA = 6
                    if self.CTRL_PNL['loss_root'] == True:
                        loss_bodyrot = self.criterion(scores[:, 10:16], scores_zeros[:, 10:16])
                    else:
                        loss_bodyrot = self.criterion(scores[:, 10:16], scores_zeros[:, 10:16]) * 0.0

                    loss_eucl = self.criterion(scores[:, 10+OSA:34+OSA], scores_zeros[:, 10+OSA:34+OSA])

                    if self.CTRL_PNL['nosmpl'] == True:
                        loss_eucl *= 0.0

                    loss += (loss_eucl + loss_bodyrot)

                    if self.CTRL_PNL['CNN'] == 'resnet':
                        loss_bedht = self.criterion(INPUT_DICT['bed_vertical_shift'], OUTPUT_DICT['bed_vertical_shift_est']) * (1./40.0)
                        loss += loss_bedht
                    if self.CTRL_PNL['loss_betas'] == True:
                        loss_betas = self.criterion(INPUT_DICT['batch_betas'], OUTPUT_DICT['batch_betas_est_post_clip']) * (1/1.728158146914805)
                        loss += loss_betas
                    if self.CTRL_PNL['regr_angles'] == True:
                        loss_angs = self.criterion2(scores[:, 37+OSA:106+OSA], scores_zeros[:, 37+OSA:106+OSA])
                        loss += loss_angs
                    if self.CTRL_PNL['v2v'] == True:
                        verts_xyz_err = INPUT_DICT['batch_verts_gt'] - OUTPUT_DICT['batch_verts_est']
                        verts_eucl_err = torch.linalg.norm(verts_xyz_err, dim = 2)
                        loss_v2v = self.criterion(verts_eucl_err, self.v2v_zeros[0:batch[0].shape[0] ,:]) * (1/0.1752780723422608)
                        loss += loss_v2v





                if self.CTRL_PNL['train_only_CAL'] == False and self.CTRL_PNL['noloss_htwt'] == False:
                    loss_bodymass = self.criterion(INPUT_DICT['batch_weight_kg'], OUTPUT_DICT['batch_weight_kg_est']) * (1./22.964567075304505)
                    loss_bodyheight = self.criterion(INPUT_DICT['batch_height'], OUTPUT_DICT['batch_height_est']) * (1./0.14554191884382228)
                    loss += (loss_bodymass + loss_bodyheight)





                self.start_pmr_epoch = 0

                if self.CTRL_PNL['train_only_CAL'] == False and self.CTRL_PNL['train_only_betanet'] == False and self.CTRL_PNL['mesh_recon_map_labels'] == True:
                    INPUT_DICT['batch_mdm_gt'][INPUT_DICT['batch_mdm_gt'] > 0] = 0
                    OUTPUT_DICT['batch_mdm_est'][OUTPUT_DICT['batch_mdm_est'] > 0] = 0

                    loss_mesh_depth = self.criterion2(INPUT_DICT['batch_mdm_gt'], OUTPUT_DICT['batch_mdm_est']) * (1. / 44.46155340000357) * (1. / 44.46155340000357)
                    loss_mesh_contact = self.criterion(INPUT_DICT['batch_cm_gt']/100., OUTPUT_DICT['batch_cm_est']) * (1. / 0.4428100696329912)
                    loss += loss_mesh_depth
                    loss += loss_mesh_contact


                if self.CTRL_PNL['train_only_CAL'] == True or self.CTRL_PNL['CNN'] == 'resnetunet':
                    loss_unet = 0
                    loss_pimg = self.criterion2(INPUT_DICT['batch_pimg'], OUTPUT_DICT['batch_pimg_est']) * (1. / 14.64204661) * (1. / 14.64204661)
                    loss_pimg_contact = self.criterion(INPUT_DICT['batch_pimg_cntct'], OUTPUT_DICT['batch_pimg_cntct_est']) * (1. / 0.4428100696329912)
                    loss_unet += loss_pimg
                    loss_unet += loss_pimg_contact
                    loss += loss_pimg
                    loss += loss_pimg_contact


                if self.CTRL_PNL['CNN'] == 'resnetunet' and self.CTRL_PNL['depth_out_unet'] == True:
                    loss_dimg = self.criterion2(INPUT_DICT['batch_dimg_noblanket_gt'], OUTPUT_DICT['batch_dimg_est'])* (1. / 864.8205975)* (1. / 864.8205975)
                    loss_dimg_contact = self.criterion(INPUT_DICT['batch_dimg_noblanket_cntct_gt'], OUTPUT_DICT['batch_dimg_cntct_est'])* (1. / 0.461730105)
                    loss_unet += loss_dimg
                    loss_unet += loss_dimg_contact
                    loss += loss_dimg
                    loss += loss_dimg_contact




                loss.backward()
                if self.CTRL_PNL['train_only_betanet'] == False and self.CTRL_PNL['train_only_CAL'] == False:
                    self.optimizer.step()
                if self.CTRL_PNL['CNN'] == 'resnet' and self.CTRL_PNL['train_only_CAL'] == True:
                    self.optimizer_CAL.step()
                if self.CTRL_PNL['CNN'] == 'resnet' and self.CTRL_PNL['train_only_betanet'] == True:
                    self.optimizer_betanet.step()
                loss *= 1000

                if batch_idx% opt.log_interval == 0:# and batch_idx > 220:

                    val_n_batches = 4
                    print ("evaluating on ", val_n_batches, batch_idx, opt.log_interval)

                    im_display_idx = 0 #random.randint(0,INPUT_DICT['batch_images'].size()[0])


                    self.VIZ_DICT = {}
                    self.VIZ_DICT = VisualizationLib().get_depthnet_viz_maps(im_display_idx, INPUT_DICT, OUTPUT_DICT, self.VIZ_DICT, self.CTRL_PNL)
                    self.VIZ_DICT = VisualizationLib().get_fcn_recon_viz_maps(im_display_idx, INPUT_DICT, OUTPUT_DICT, self.VIZ_DICT, self.CTRL_PNL)


                    self.tar_sample = INPUT_DICT['batch_targets']
                    self.tar_sample = self.tar_sample[im_display_idx, :].squeeze().cpu() / 1000

                    if self.CTRL_PNL['train_only_betanet'] == False and self.CTRL_PNL['train_only_CAL'] == False:
                        VisualizationLib().print_error_train(INPUT_DICT['batch_targets'].cpu(), OUTPUT_DICT['batch_targets_est'].cpu(),
                                                             self.output_size_train, self.CTRL_PNL['loss_vector_type'],
                                                             data='train')

                        self.sc_sample = OUTPUT_DICT['batch_targets_est'].clone()
                        self.sc_sample = self.sc_sample[im_display_idx, :].squeeze() / 1000
                        self.sc_sample = self.sc_sample.view(self.output_size_train).cpu()
                    else:
                        self.sc_sample = None

                    train_loss = loss.data.item()
                    examples_this_epoch = batch_idx * len(INPUT_DICT['batch_images'])
                    epoch_progress = 100. * batch_idx / len(self.train_loader)

                    val_loss = self.validate_convnet(n_batches=val_n_batches)


                    print_text_list = [ 'Train Epoch: {} ',
                                        '[{}',
                                        '/{} ',
                                        '({:.0f}%)]\t']
                    print_vals_list = [epoch,
                                      examples_this_epoch,
                                      len(self.train_loader.dataset),
                                      epoch_progress]
                    if self.CTRL_PNL['loss_vector_type'] != 'direct' and self.CTRL_PNL['train_only_CAL'] == False and self.CTRL_PNL['train_only_betanet'] == False:
                        print_text_list.append('Train Loss Joints: {:.2f}')
                        print_vals_list.append(1000*loss_eucl.data)
                        self.train_val_losses['loss_eucl'].append(1000*loss_eucl.data.item())
                        if self.CTRL_PNL['loss_betas'] == True:
                            print_text_list.append('\n\t\t\t\t\t\t   Betas Loss: {:.2f}')
                            print_vals_list.append(1000*loss_betas.data)
                            self.train_val_losses['loss_betas'].append(1000*loss_betas.data.item())
                        print_text_list.append('\n\t\t\t\t\t\tBody Rot Loss: {:.2f}')
                        print_vals_list.append(1000*loss_bodyrot.data)
                        self.train_val_losses['loss_bodyrot'].append(1000*loss_bodyrot.data.item())
                        if self.CTRL_PNL['regr_angles'] == True:
                            print_text_list.append('\n\t\t\t\t\t\t  Angles Loss: {:.2f}')
                            print_vals_list.append(1000*loss_angs.data)
                            self.train_val_losses['loss_angs'].append(1000*loss_angs.data.item())
                        if self.CTRL_PNL['v2v'] == True:
                            print_text_list.append('\n\t\t\t\t\t\t     v2v Loss: {:.2f}')
                            print_vals_list.append(1000*loss_v2v.data)
                            self.train_val_losses['loss_v2v'].append(1000*loss_v2v.data.item())
                        if self.CTRL_PNL['CNN'] == 'resnet':
                            print_text_list.append('\n\t\t\t\t\t\t Bed Ht. Loss: {:.2f}')
                            print_vals_list.append(1000*loss_bedht.data)
                            self.train_val_losses['loss_bedht'].append(1000*loss_bedht.data.item())
                        if self.CTRL_PNL['mesh_recon_map_labels'] == True:
                            if epoch >= self.start_pmr_epoch:
                                print_text_list.append('\n\t\t\t\t\t\t   Mesh Depth: {:.2f}')
                                print_vals_list.append(1000*loss_mesh_depth.data)
                                self.train_val_losses['loss_meshdepth'].append(1000*loss_mesh_depth.data.item())
                                print_text_list.append('\n\t\t\t\t\t\t Mesh Contact: {:.2f}')
                                print_vals_list.append(1000*loss_mesh_contact.data)
                                self.train_val_losses['loss_meshcntct'].append(1000*loss_mesh_contact.data.item())


                    if self.CTRL_PNL['loss_vector_type'] != 'direct' and self.CTRL_PNL['train_only_CAL'] == False and self.CTRL_PNL['noloss_htwt'] == False:
                        print_text_list.append('\n\t\t\t\t\t       Body Mass Loss: {:.2f}')
                        print_vals_list.append(1000*loss_bodymass.data)
                        self.train_val_losses['loss_bodymass'].append(1000*loss_bodymass.data.item())
                        print_text_list.append('\n\t\t\t\t\t     Body Height Loss: {:.2f}')
                        print_vals_list.append(1000*loss_bodyheight.data)
                        self.train_val_losses['loss_bodyheight'].append(1000*loss_bodyheight.data.item())


                    if self.CTRL_PNL['train_only_CAL'] == True or self.CTRL_PNL['CNN'] == 'resnetunet':
                        print_text_list.append('\n\t\t\t\t\t\t Pressure Img: {:.2f}')
                        print_vals_list.append(1000 * loss_pimg.data)
                        print_text_list.append('\n\t\t\t\t\t       Pressure Cntct: {:.2f}')
                        print_vals_list.append(1000 * loss_pimg_contact.data)

                    if self.CTRL_PNL['depth_out_unet'] == True:
                        print_text_list.append('\n\t\t\t\t\t\t   Depth Img: {:.2f}')
                        print_vals_list.append(1000 * loss_dimg.data)
                        print_text_list.append('\n\t\t\t\t\t          Depth Cntct: {:.2f}')
                        print_vals_list.append(1000 * loss_dimg_contact.data)


                    print_text_list.append('\n\t\t\t\t\t\t   Total Loss: {:.2f}')
                    print_vals_list.append(train_loss)

                    print_text_list.append('\n\t\t\t\t\t  Val Total Loss: {:.2f}')
                    print_vals_list.append(val_loss)

                    print_text = ''
                    for item in print_text_list:
                        print_text += item
                    print(print_text.format(*print_vals_list))


                    print ('appending to alldata losses')
                    self.train_val_losses['train_loss'].append(train_loss)
                    if self.CTRL_PNL['depth_out_unet'] == True:
                        self.train_val_losses['unet_loss'].append(1000*loss_unet.data.item())
                    self.train_val_losses['epoch_ct'].append(epoch)
                    self.train_val_losses['val_loss'].append(val_loss)


    def validate_convnet(self, verbose=False, n_batches=None):
        if self.CTRL_PNL['train_only_betanet'] == False and self.CTRL_PNL['train_only_CAL'] == False:
            self.model.eval()
        if self.CTRL_PNL['CNN'] == 'resnet' and self.CTRL_PNL['train_only_betanet'] == False:
            self.model_CAL.eval()
        if self.CTRL_PNL['CNN'] == 'resnet' and self.CTRL_PNL['train_only_CAL'] == False:
            self.model_betanet.eval()

        loss = 0.
        n_examples = 0
        batch_ct = 1

        for batch_i, batch in enumerate(self.test_loader):

            scores, INPUT_DICT_VAL, OUTPUT_DICT_VAL = \
                UnpackDepthBatchLib().unpack_batch(batch, is_training=False, model=self.model, model_smpl_pmr = self.model_smpl_pmr,
                                                   model_CAL = self.model_CAL, model_betanet = self.model_betanet, CTRL_PNL=self.CTRL_PNL)

            loss_to_add = 0


            if self.CTRL_PNL['slp'] == 'real' and self.CTRL_PNL['loss_vector_type'] == 'direct':
                scores_zeros = Variable(torch.Tensor(np.zeros((batch[0].shape[0], scores.size()[1]))).type(dtype), requires_grad=False)
                loss_to_add += self.criterion(scores[:, 0:14], scores_zeros[:, 0:14])
            elif self.CTRL_PNL['train_only_CAL'] == False and self.CTRL_PNL['train_only_betanet'] == False:
                scores_zeros = Variable(torch.Tensor(np.zeros((batch[0].shape[0], scores.size()[1]))).type(dtype), requires_grad=False)
                OSA = 6
                if self.CTRL_PNL['loss_root'] == True:
                    loss_bodyrot = float(self.criterion(scores[:, 10:16], scores_zeros[:, 10:16]))
                else:
                    loss_bodyrot = float(self.criterion(scores[:, 10:16], scores_zeros[:, 10:16]) * 0.0)

                loss_eucl = float(self.criterion(scores[:, 10+OSA:34+OSA], scores_zeros[:,  10+OSA:34+OSA]))

                if self.CTRL_PNL['CNN'] == 'resnet' and self.CTRL_PNL['nosmpl'] == False:
                    loss_to_add += self.criterion(INPUT_DICT_VAL['bed_vertical_shift'], OUTPUT_DICT_VAL['bed_vertical_shift_est']) * (1. / 40.0)

                if self.CTRL_PNL['nosmpl'] == False:
                    loss_to_add += (loss_bodyrot + loss_eucl)# + loss_bodymass)

                if self.CTRL_PNL['loss_betas'] == True and self.CTRL_PNL['nosmpl'] == False:
                    loss_betas = self.criterion(INPUT_DICT_VAL['batch_betas'], OUTPUT_DICT_VAL['batch_betas_est_post_clip']) * (1/1.728158146914805)
                    loss_to_add += (loss_betas)

                if self.CTRL_PNL['regr_angles'] == True and self.CTRL_PNL['nosmpl'] == False:
                    loss_angs = float(self.criterion(scores[:, 37+OSA:106+OSA], scores_zeros[:, 37+OSA:106+OSA]))
                    loss_to_add += (loss_angs)

                if self.CTRL_PNL['v2v'] == True and self.CTRL_PNL['nosmpl'] == False:
                    verts_xyz_err = INPUT_DICT_VAL['batch_verts_gt'] - OUTPUT_DICT_VAL['batch_verts_est']
                    verts_eucl_err = torch.linalg.norm(verts_xyz_err, dim = 2)
                    loss_v2v = self.criterion(verts_eucl_err, self.v2v_zeros[0:batch[0].shape[0] ,:]) * (1/0.1752780723422608)
                    loss_to_add += loss_v2v


            if self.CTRL_PNL['train_only_CAL'] == False and self.CTRL_PNL['noloss_htwt'] == False:
                loss_bodymass = self.criterion(INPUT_DICT_VAL['batch_weight_kg'], OUTPUT_DICT_VAL['batch_weight_kg_est']) * (1./22.964567075304505)
                loss_bodyheight = self.criterion(INPUT_DICT_VAL['batch_height'], OUTPUT_DICT_VAL['batch_height_est']) * (1./0.14554191884382228)
                loss_to_add += (loss_bodymass + loss_bodyheight)


            if self.CTRL_PNL['train_only_CAL'] == False and self.CTRL_PNL['train_only_betanet'] == False and self.CTRL_PNL['mesh_recon_map_labels'] == True:
                INPUT_DICT_VAL['batch_mdm_gt'][INPUT_DICT_VAL['batch_mdm_gt'] > 0] = 0
                #if self.CTRL_PNL['mesh_bottom_dist'] == True:
                OUTPUT_DICT_VAL['batch_mdm_est'][OUTPUT_DICT_VAL['batch_mdm_est'] > 0] = 0
                loss_mesh_depth = float(self.criterion2(INPUT_DICT_VAL['batch_mdm_gt'],OUTPUT_DICT_VAL['batch_mdm_est'])  * (1. / 44.46155340000357) * (1. / 44.46155340000357))
                loss_mesh_contact = float(self.criterion(INPUT_DICT_VAL['batch_cm_gt']/100.,OUTPUT_DICT_VAL['batch_cm_est'])  * (1. / 0.4428100696329912))
                loss_to_add += loss_mesh_depth
                loss_to_add += loss_mesh_contact

            if self.CTRL_PNL['train_only_CAL'] == True or self.CTRL_PNL['CNN'] == 'resnetunet':
                loss_pimg = float(self.criterion2(INPUT_DICT_VAL['batch_pimg'], OUTPUT_DICT_VAL['batch_pimg_est']) * (1. / 14.64204661) * (1. / 14.64204661))
                loss_pimg_contact = float(self.criterion(INPUT_DICT_VAL['batch_pimg_cntct'], OUTPUT_DICT_VAL['batch_pimg_cntct_est']) * (1. / 0.4428100696329912))
                loss_to_add += loss_pimg
                loss_to_add += loss_pimg_contact

            if self.CTRL_PNL['depth_out_unet'] == True:
                loss_dimg = self.criterion2(INPUT_DICT_VAL['batch_dimg_noblanket_gt'], OUTPUT_DICT_VAL['batch_dimg_est']) * (1. / 864.8205975) * (1. / 864.8205975)
                loss_dimg_contact = self.criterion(INPUT_DICT_VAL['batch_dimg_noblanket_cntct_gt'], OUTPUT_DICT_VAL['batch_dimg_cntct_est']) * (1. / 0.461730105)
                loss_to_add += loss_dimg
                loss_to_add += loss_dimg_contact



            loss += float(loss_to_add) #if this isn't turned into a float it sucks up a ton of memory

            n_examples += self.CTRL_PNL['batch_size']

            if n_batches and (batch_i >= n_batches):
                break

            batch_ct += 1


        loss /= batch_ct
        loss *= 1000


        if self.opt.slp_depth == True or self.opt.slp == 'real':
            max_depth = 2200
        else:
            max_depth = 1750

        if self.opt.visualize == True:
            VisualizationLib().visualize_depth_net(VIZ_DICT = self.VIZ_DICT,
                                                      targets_raw = self.tar_sample, scores_net1 = self.sc_sample,
                                                      block=False, max_depth = max_depth)

        return loss






if __name__ == "__main__":

    import optparse

    from optparse_lib import get_depthnet_options

    p = optparse.OptionParser()

    p = get_depthnet_options(p)

    p.add_option('--mod', action='store', type = 'int', dest='mod', default=1,
                 help='Choose a network.')

    p.add_option('--viz', action='store_true', dest='visualize', default=False,  help='Visualize training.')
    opt, args = p.parse_args()


    if opt.hd == True:
        dana_lab_path = '/media/henry/multimodal_data_2/data/SLP/danaLab/'
    else:
        dana_lab_path = FILEPATH +'data_BP/SLP/danaLab/'




    FileNameInputLib1 = FileNameInputLib(opt, depth = False)

    if opt.quick_test == True:
        if opt.slp == 'real':
            train_database_file_real_f, train_database_file_real_m, train_subj_mass_list_f, train_subj_mass_list_m = FileNameInputLib1.get_qt_dana_slp(True)
            test_database_file_real_f, test_database_file_real_m, test_subj_mass_list_f, test_subj_mass_list_m = FileNameInputLib1.get_qt_dana_slp(False)
            train_database_file_synth_f, train_database_file_synth_m = None, None
            test_database_file_synth_f, test_database_file_synth_m = None, None
        elif opt.slp == 'synth':
            train_database_file_real_f, train_database_file_real_m = None, None
            test_database_file_real_f, test_database_file_real_m, test_subj_mass_list_f, test_subj_mass_list_m = FileNameInputLib1.get_qt_dana_slp(False)
            train_database_file_synth_f, train_database_file_synth_m = FileNameInputLib1.get_qt_slpsynth_pressurepose(True, '')#_nonoise')
            test_database_file_synth_f, test_database_file_synth_m = FileNameInputLib1.get_qt_slpsynth_pressurepose(False, '')#_nonoise')
        else:
            sys.exit()

    else:
        if opt.slp == 'real':
            train_database_file_real_f, train_database_file_real_m, train_subj_mass_list_f, train_subj_mass_list_m = FileNameInputLib1.get_dana_slp(True)
            test_database_file_real_f, test_database_file_real_m, test_subj_mass_list_f, test_subj_mass_list_m = FileNameInputLib1.get_dana_slp(False)
            train_database_file_synth_f, train_database_file_synth_m = None, None
            test_database_file_synth_f, test_database_file_synth_m = None, None
        elif opt.slp == 'synth':
            train_database_file_real_f, train_database_file_real_m = None, None
            test_database_file_real_f, test_database_file_real_m, test_subj_mass_list_f, test_subj_mass_list_m = FileNameInputLib1.get_dana_slp(False)
            train_database_file_synth_f, train_database_file_synth_m = FileNameInputLib1.get_slpsynth_pressurepose(True, '')#_nonoise')
            test_database_file_synth_f, test_database_file_synth_m = FileNameInputLib1.get_slpsynth_pressurepose(False, '')#_nonoise')
        elif opt.slp == 'mixedreal':
            train_database_file_real_f, train_database_file_real_m, train_subj_mass_list_f, train_subj_mass_list_m = FileNameInputLib1.get_dana_slp(True)
            test_database_file_real_f, test_database_file_real_m, test_subj_mass_list_f, test_subj_mass_list_m = FileNameInputLib1.get_dana_slp(False)
            train_database_file_synth_f, train_database_file_synth_m = FileNameInputLib1.get_slpsynth_pressurepose(True, '')#_nonoise')
            test_database_file_synth_f, test_database_file_synth_m = FileNameInputLib1.get_slpsynth_pressurepose(False, '')#_nonoise')
        else:
            sys.exit()



    train_files_f = [train_database_file_real_f, train_database_file_synth_f]
    test_files_f = [test_database_file_real_f, test_database_file_synth_f]
    train_files_m = [train_database_file_real_m, train_database_file_synth_m]
    test_files_m = [test_database_file_real_m, test_database_file_synth_m]


    FileNameInputLib2 = FileNameInputLib(opt, depth = True)

    if opt.quick_test == True:
        if opt.slp == 'real':
            train_database_file_real_depth_f, train_database_file_real_depth_m, train_subj_mass_list_f, train_subj_mass_list_m = FileNameInputLib2.get_qt_dana_slp(True)
            test_database_file_real_depth_f, test_database_file_real_depth_m, test_subj_mass_list_f, test_subj_mass_list_m = FileNameInputLib2.get_qt_dana_slp(False)
            train_database_file_synth_depth_f, train_database_file_synth_depth_m = None, None
            test_database_file_synth_depth_f, test_database_file_synth_depth_m = None, None
        elif opt.slp == 'synth':
            train_database_file_real_depth_f, train_database_file_real_depth_m = None, None
            test_database_file_real_depth_f, test_database_file_real_depth_m = None, None
            train_database_file_synth_depth_f, train_database_file_synth_depth_m = FileNameInputLib2.get_qt_slpsynth_pressurepose(True, '')
            test_database_file_synth_depth_f, test_database_file_synth_depth_m = FileNameInputLib2.get_qt_slpsynth_pressurepose(False, '')
        else:
            sys.exit()
    else:
        if opt.slp == 'real':
            train_database_file_real_depth_f, train_database_file_real_depth_m, train_subj_mass_list_f, train_subj_mass_list_m = FileNameInputLib2.get_dana_slp(True)
            test_database_file_real_depth_f, test_database_file_real_depth_m, test_subj_mass_list_f, test_subj_mass_list_m = FileNameInputLib2.get_dana_slp(False)
            train_database_file_synth_depth_f, train_database_file_synth_depth_m = None, None
            test_database_file_synth_depth_f, test_database_file_synth_depth_m = None, None
        elif opt.slp == 'synth':
            train_database_file_real_depth_f, train_database_file_real_depth_m = None, None
            test_database_file_real_depth_f, test_database_file_real_depth_m, test_subj_mass_list_f, test_subj_mass_list_m = FileNameInputLib2.get_dana_slp(False)
            train_database_file_synth_depth_f, train_database_file_synth_depth_m = FileNameInputLib2.get_slpsynth_pressurepose(True, '')
            test_database_file_synth_depth_f, test_database_file_synth_depth_m = FileNameInputLib2.get_slpsynth_pressurepose(False, '')
        elif opt.slp == 'mixedreal':
            train_database_file_real_depth_f, train_database_file_real_depth_m, train_subj_mass_list_f, train_subj_mass_list_m = FileNameInputLib2.get_dana_slp(True)
            test_database_file_real_depth_f, test_database_file_real_depth_m, test_subj_mass_list_f, test_subj_mass_list_m = FileNameInputLib2.get_dana_slp(False)
            train_database_file_synth_depth_f, train_database_file_synth_depth_m = FileNameInputLib2.get_slpsynth_pressurepose(True, '')#_nonoise')
            test_database_file_synth_depth_f, test_database_file_synth_depth_m = FileNameInputLib2.get_slpsynth_pressurepose(False, '')#_nonoise')
        else:
            sys.exit()



    train_files_f += [train_database_file_real_depth_f, train_database_file_synth_depth_f]
    test_files_f += [test_database_file_real_depth_f, test_database_file_synth_depth_f]
    train_files_m += [train_database_file_real_depth_m, train_database_file_synth_depth_m]
    test_files_m += [test_database_file_real_depth_m, test_database_file_synth_depth_m]
    #print "  "
    #for item in train_files_f[1]: print item

    p = PhysicalTrainer(train_files_f, train_files_m, test_files_f, test_files_m, opt)


    p.init_convnet_train()

