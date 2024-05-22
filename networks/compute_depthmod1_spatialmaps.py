#!/usr/bin/env python
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import *


#PyTorch libraries
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import chumpy as ch


txtfile = open("/home/ganyong/Githubwork/Examples/BodyPressure/FILEPATH.txt")
FILEPATH = txtfile.read().replace("\n", "")
txtfile.close()


import sys
sys.path.insert(0, '../lib_py')
sys.path.insert(-1,FILEPATH+'smpl/smpl_webuser3')
sys.path.insert(0, FILEPATH)
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

try:
    import cPickle as pkl
    from smpl.smpl_webuser.serialization import load_model as LOAD_MODEL
    def load_pickle(filename):
        with open(filename, 'rb') as f:
            return pkl.load(f)
except:
    import pickle as pkl
    from smpl.smpl_webuser3.serialization import load_model as LOAD_MODEL
    def load_pickle(filename):
        with open(filename, 'rb') as f:
            return pkl.load(f, encoding = 'latin1')

import fixedwt_smpl_pmr_net as fixedwt_smpl_pmr

# Pose Estimation Libraries
from visualization_lib_bp import VisualizationLib
from preprocessing_lib_bp import PreprocessingLib
from tensorprep_lib_bp import TensorPrepLib
from unpack_depth_batch_lib_bp import UnpackDepthBatchLib
from filename_input_lib_bp import FileNameInputLib
from slp_prep_lib_bp import SLPPrepLib


import random
from scipy import ndimage
import scipy.stats as ss
from scipy.ndimage.interpolation import zoom

np.set_printoptions(threshold=sys.maxsize)

MAT_WIDTH = 0.762 #metres
MAT_HEIGHT = 1.854 #metres
MAT_HALF_WIDTH = MAT_WIDTH/2
NUMOFTAXELS_X = 64#73 #taxels
NUMOFTAXELS_Y = 27#30
NUMOFOUTPUTDIMS = 3
NUMOFOUTPUTNODES_TRAIN = 24
NUMOFOUTPUTNODES_TEST = 10
INTER_SENSOR_DISTANCE = 0.0286#metres
LOW_TAXEL_THRESH_X = 0
LOW_TAXEL_THRESH_Y = 0
HIGH_TAXEL_THRESH_X = (NUMOFTAXELS_X - 1)
HIGH_TAXEL_THRESH_Y = (NUMOFTAXELS_Y - 1)

DROPOUT = False


torch.set_num_threads(1)
if torch.cuda.is_available():
    # Use for GPU
    GPU = True
    dtype = torch.cuda.FloatTensor
    print('######################### CUDA is available! #############################')
else:
    # Use for CPU
    GPU = False
    dtype = torch.FloatTensor
    print('############################## USING CPU #################################')


class PhysicalTrainer():

    def __init__(self, test_files_f, test_files_m, opt, file_prefix):
        self.CTRL_PNL = {}

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
        self.CTRL_PNL['batch_size'] = 128
        self.CTRL_PNL['loss_vector_type'] = opt.losstype
        self.CTRL_PNL['verbose'] = opt.verbose
        self.opt = opt
        self.CTRL_PNL['num_epochs'] = 500
        self.CTRL_PNL['incl_inter'] = True
        self.CTRL_PNL['shuffle'] = False
        self.CTRL_PNL['half_network_size'] = opt.hns
        self.CTRL_PNL['double_network_size'] = False
        self.CTRL_PNL['incl_ht_wt_channels'] = opt.htwt
        self.CTRL_PNL['omit_pimg_cntct_sobel'] = opt.omit_pimg_cntct_sobel
        self.CTRL_PNL['incl_pmat_cntct_input'] = True
        self.CTRL_PNL['lock_root'] = False
        self.CTRL_PNL['num_input_channels'] = 2
        self.CTRL_PNL['GPU'] = GPU
        self.CTRL_PNL['dtype'] = dtype
        self.CTRL_PNL['repeat_real_data_ct'] = 1
        self.CTRL_PNL['regr_angles'] = 1

        self.CTRL_PNL['mesh_recon_map_labels'] = False
        self.CTRL_PNL['dropout'] = False
        self.CTRL_PNL['mesh_recon_map_labels_test'] = True #can only be true is we have 100% synth for testing
        self.CTRL_PNL['mesh_recon_map_output'] = True
        self.CTRL_PNL['mesh_recon_output'] = True
        self.CTRL_PNL['recon_map_input_est'] = False #do this if we're working in a two-part regression
        self.CTRL_PNL['adjust_ang_from_est'] = self.CTRL_PNL['recon_map_input_est'] #holds betas and root same as prior estimate
        self.CTRL_PNL['clip_sobel'] = True
        self.CTRL_PNL['clip_betas'] = True
        self.CTRL_PNL['full_body_rot'] = True
        self.CTRL_PNL['normalize_per_image'] = True
        if self.CTRL_PNL['normalize_per_image'] == False:
            self.CTRL_PNL['normalize_std'] = True
        else:
            self.CTRL_PNL['normalize_std'] = False
        self.CTRL_PNL['all_tanh_activ'] = True
        self.CTRL_PNL['pmat_mult'] = int(1)
        self.CTRL_PNL['slp_noise'] = opt.slpnoise
        if opt.no_depthnoise == True: self.CTRL_PNL['depth_noise'] = False
        else: self.CTRL_PNL['depth_noise'] = True
        self.CTRL_PNL['cal_noise_amt'] = 0.2
        self.CTRL_PNL['first_pass'] = True
        self.CTRL_PNL['align_procr'] = False
        self.CTRL_PNL['depth_in'] = True
        self.CTRL_PNL['clean_slp_depth'] = False
        self.CTRL_PNL['train_only_betanet'] = opt.train_only_betanet
        self.CTRL_PNL['train_only_CAL'] = opt.train_only_CAL
        self.CTRL_PNL['compute_forward_maps'] = True
        self.CTRL_PNL['v2v'] = opt.v2v
        self.CTRL_PNL['x_y_offset_synth'] = [12, -35]


        if np.shape(test_files_f)[1] == 0:
            self.gender = "m"
        elif np.shape(test_files_m)[1] == 0:
            self.gender = "f"

        self.file_prefix = file_prefix

        if GPU == True:
            torch.cuda.set_device(self.opt.device)

        if opt.losstype == 'direct':
            self.CTRL_PNL['mesh_recon_map_labels'] = False
            self.CTRL_PNL['mesh_recon_map_output'] = False



        if self.CTRL_PNL['incl_pmat_cntct_input'] == True:
            self.CTRL_PNL['num_input_channels'] += 1
        if self.CTRL_PNL['recon_map_input_est'] == True: #for a two part regression
            self.CTRL_PNL['num_input_channels'] += 2
        self.CTRL_PNL['num_input_channels_batch0'] = np.copy(self.CTRL_PNL['num_input_channels'])
        self.CTRL_PNL['num_input_channels'] += 1



        pmat_std_from_mult = ['N/A', 14.64204661, 19.90905848383454, 23.07018866032369, 0.0, 25.50538629767412]


        self.CTRL_PNL['norm_std_coeffs'] =  [1./41.80684362163343,  #contact
                                             1./16.69545796387731,  #pos est depth
                                             1./45.08513083167194,  #neg est depth
                                             1./43.55800622930469,  #cm est
                                             1./pmat_std_from_mult[int(self.CTRL_PNL['pmat_mult'])], #pmat x5
                                             1./1.0,                #bed height mat
                                             1./1.0,                #OUTPUT DO NOTHING
                                             1./1.0,                #OUTPUT DO NOTHING
                                             1. / 30.216647403350,  #weight
                                             1. / 14.629298141231]  #height



        if self.CTRL_PNL['normalize_std'] == False:
            for i in range(9):
                self.CTRL_PNL['norm_std_coeffs'][i] *= 0.
                self.CTRL_PNL['norm_std_coeffs'][i] += 1.

        if self.CTRL_PNL['mesh_recon_output'] == True: #we need all the vertices if we're going to regress the depth maps
            self.verts_list = "all"
        else:
            self.verts_list = [1325, 336, 1032, 4515, 1374, 4848, 1739, 5209, 1960, 5423]


        self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)
        self.output_size_train = (NUMOFOUTPUTNODES_TRAIN, NUMOFOUTPUTDIMS)
        self.output_size_val = (NUMOFOUTPUTNODES_TEST, NUMOFOUTPUTDIMS)
        self.parents = np.array([4294967295, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]).astype(np.int32)



        #################################### PREP TESTING DATA ##########################################
        #load training ysnth data
        if opt.small == True:
            reduce_data = True
        else:
            reduce_data = False


        # load in the test file
        #if self.opt.slp == 'real' or self.opt.slp == 'mixedreal':
        try:
            if self.gender == 'f':
                test_dat_f_real = {}
                test_dat_f_real_u = SLPPrepLib().load_slp_files_to_database(test_files_f[0], dana_lab_path, PM='uncover', depth='uncover', mass_ht_list = test_subj_mass_list_f, markers_gt_type = '3D', use_pc = False, depth_out_unet = self.CTRL_PNL['depth_out_unet'], pm_adjust_mm = self.CTRL_PNL['x_y_offset_synth'])
                test_dat_f_real_c1 = SLPPrepLib().load_slp_files_to_database(test_files_f[0], dana_lab_path, PM='cover1', depth='cover1', mass_ht_list = test_subj_mass_list_f, markers_gt_type = '3D', use_pc = False, depth_out_unet = self.CTRL_PNL['depth_out_unet'], pm_adjust_mm = self.CTRL_PNL['x_y_offset_synth'])
                test_dat_f_real_c2 = SLPPrepLib().load_slp_files_to_database(test_files_f[0], dana_lab_path, PM='cover2', depth='cover2', mass_ht_list = test_subj_mass_list_f, markers_gt_type = '3D', use_pc = False, depth_out_unet = self.CTRL_PNL['depth_out_unet'], pm_adjust_mm = self.CTRL_PNL['x_y_offset_synth'])
                for item in test_dat_f_real_c1:
                    test_dat_f_real[item] = []
                    for i in range(len(test_dat_f_real_c1[item])):
                        try:
                            test_dat_f_real[item].append(test_dat_f_real_u[item][i])
                        except:
                            test_dat_f_real[item].append(test_dat_f_real_u['overhead_depthcam_noblanket'][i])
                    for i in range(len(test_dat_f_real_c1[item])):
                        test_dat_f_real[item].append(test_dat_f_real_c1[item][i])
                    for i in range(len(test_dat_f_real_c2[item])):
                        test_dat_f_real[item].append(test_dat_f_real_c2[item][i])
                test_dat_m_real = None

            elif self.gender == 'm':
                test_dat_m_real = {}
                test_dat_m_real_u = SLPPrepLib().load_slp_files_to_database(test_files_m[0], dana_lab_path, PM='uncover', depth='uncover', mass_ht_list = test_subj_mass_list_m, markers_gt_type = '3D', use_pc = False, depth_out_unet = self.CTRL_PNL['depth_out_unet'], pm_adjust_mm = self.CTRL_PNL['x_y_offset_synth'])
                test_dat_m_real_c1 = SLPPrepLib().load_slp_files_to_database(test_files_m[0], dana_lab_path, PM='cover1', depth='cover1', mass_ht_list = test_subj_mass_list_m, markers_gt_type = '3D', use_pc = False, depth_out_unet = self.CTRL_PNL['depth_out_unet'], pm_adjust_mm = self.CTRL_PNL['x_y_offset_synth'])
                test_dat_m_real_c2 = SLPPrepLib().load_slp_files_to_database(test_files_m[0], dana_lab_path, PM='cover2', depth='cover2', mass_ht_list = test_subj_mass_list_m, markers_gt_type = '3D', use_pc = False, depth_out_unet = self.CTRL_PNL['depth_out_unet'], pm_adjust_mm = self.CTRL_PNL['x_y_offset_synth'])
                for item in test_dat_m_real_c1:
                    test_dat_m_real[item] = []
                    for i in range(len(test_dat_m_real_c1[item])):
                        try:
                            test_dat_m_real[item].append(test_dat_m_real_u[item][i])
                        except:
                            test_dat_m_real[item].append(test_dat_m_real_u['overhead_depthcam_noblanket'][i])
                    for i in range(len(test_dat_m_real_c1[item])):
                        test_dat_m_real[item].append(test_dat_m_real_c1[item][i])
                    for i in range(len(test_dat_m_real_c1[item])):
                        test_dat_m_real[item].append(test_dat_m_real_c2[item][i])
                test_dat_f_real = None
        except:
            test_dat_m_real = None
            test_dat_f_real = None


        if self.opt.slp == 'real':
            test_dat_f_synth = None
            test_dat_m_synth = None
        else:
            test_dat_f_synth = TensorPrepLib(opt=self.opt).load_files_to_database(test_files_f, creation_type = 'synth', reduce_data = reduce_data, depth_in = True)
            test_dat_m_synth = TensorPrepLib(opt=self.opt).load_files_to_database(test_files_m, creation_type = 'synth', reduce_data = reduce_data, depth_in = True)



        print(test_files_f, test_files_m, 'f m synth')

        for possible_dat in [test_dat_f_synth, test_dat_m_synth, test_dat_f_real, test_dat_m_real]:
            print('try ...')
            if possible_dat is not None and possible_dat:
                self.dat = possible_dat

                #for item in self.dat:
                #    print('possible dat', item, len(self.dat[item]))

                if self.CTRL_PNL['compute_forward_maps'] == False:
                    self.dat['verts'] = []
                else:
                    self.dat['angles_est'] = []
                    self.dat['root_xyz_est'] = []
                    self.dat['betas_est'] = []
                    self.dat['root_atan2_est'] = []
                    if self.CTRL_PNL['CNN'] == 'resnet':
                        self.dat['mdm_est'] = []
                        self.dat['cm_est'] = []
                        self.dat['bed_vertical_shift_est'] = []
                    elif self.CTRL_PNL['CNN'] == 'resnetunet':
                        self.dat['pimg_est'] = []
                        #self.dat['pimg_cntct_est'] = []
                        self.dat['dimg_est'] = []
                        #self.dat['dimg_cntct_est'] = []

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


        len_test = 0
        try:
            len_test += np.shape(test_dat_f_synth['images'])[0]
        except:
            pass
        try:
            len_test +=  np.shape(test_dat_m_synth['images'])[0]
        except:
            pass
        try:
            len_test += np.shape(test_dat_f_real['images'])[0]
        except:
            pass
        try:
            len_test += np.shape(test_dat_m_real['images'])[0]
        except:
            pass


        test_x = np.zeros((len_test, x_map_ct, 64, 27)).astype(np.float32)

        print(test_x.shape)

        #allocate pressure images
        test_x = TensorPrepLib().prep_images(test_x, test_dat_f_real, test_dat_m_real, test_dat_f_synth, test_dat_m_synth, filter_sigma = 0.5, start_map_idx = pmat_gt_idx)


        if self.CTRL_PNL['mesh_recon_map_labels'] == True:
            #Initialize the precomputed depth and contact maps. only synth has this label.
            test_x = TensorPrepLib().prep_reconstruction_gt(test_x, test_dat_f_real, test_dat_m_real, test_dat_f_synth, test_dat_m_synth, start_map_idx = recon_gt_idx)

        if self.CTRL_PNL['recon_map_input_est'] == True:
            #Initialize the precomputed depth and contact map input estimates
            test_x = TensorPrepLib().prep_reconstruction_input_est(test_x, test_dat_f_real, test_dat_m_real, test_dat_f_synth, test_dat_m_synth, start_map_idx = 0)



        im_ct = 0
        if test_x.shape[0] == 135:
            test_x = TensorPrepLib().prep_depth_input_images(test_x, test_dat_f_real, test_dat_m_real, test_dat_f_synth, test_dat_m_synth, start_map_idx = depth_in_idx, depth_type = 'all_meshes')
            print('doing a real subj')
        else:
            if self.opt.no_blanket == False:
               test_x = TensorPrepLib().prep_depth_input_images(test_x, test_dat_f_real, test_dat_m_real, test_dat_f_synth, test_dat_m_synth, start_map_idx = depth_in_idx, depth_type = 'all_meshes', mix_bl_nobl = True) #'all_meshes')#'
            else:
               test_x = TensorPrepLib().prep_depth_input_images(test_x, test_dat_f_real, test_dat_m_real, test_dat_f_synth, test_dat_m_synth, start_map_idx = depth_in_idx, depth_type = 'no_blanket', mix_bl_nobl = False) #'all_meshes')#'


        if self.CTRL_PNL['depth_out_unet'] == True:
            test_x = TensorPrepLib().prep_depth_input_images(test_x, test_dat_f_real, test_dat_m_real, test_dat_f_synth, test_dat_m_synth, start_map_idx = depth_in_idx+4, depth_type = 'human_only')


        self.test_x_tensor = torch.Tensor(test_x)



        test_y_flat = []  # Initialize the ground truth listhave
        for gender_synth in [["f", test_dat_f_real], ["m", test_dat_m_real]]:
            test_y_flat = SLPPrepLib().prep_labels_slp(test_y_flat, gender_synth[1], num_repeats = 1,
                                                        z_adj = -0.075, gender = gender_synth[0], is_synth = True, markers_gt_type = '3D') #not sure if we should us is_synth true or false???

        for gender_synth in [["f", test_dat_f_synth], ["m", test_dat_m_synth]]:
            test_y_flat = TensorPrepLib().prep_labels(test_y_flat, gender_synth[1],
                                                        z_adj = -0.075, gender = gender_synth[0], is_synth = True,
                                                        loss_vector_type = self.CTRL_PNL['loss_vector_type'],
                                                        initial_angle_est = self.CTRL_PNL['adjust_ang_from_est'], x_y_adjust_mm = self.CTRL_PNL['x_y_offset_synth'])

        self.test_y_tensor = torch.Tensor(test_y_flat)


        print (self.test_x_tensor.shape, 'Input testing tensor shape')
        print (self.test_y_tensor.shape, 'Output testing tensor shape')

        if self.test_x_tensor.size()[0] == 135:
            self.CTRL_PNL['slp'] = 'real'
        else:
            self.CTRL_PNL['slp'] = 'synth'


    def init_convnet_test(self):

        if self.CTRL_PNL['verbose']: print (self.test_x_tensor.size(), 'length of the testing dataset')
        if self.CTRL_PNL['verbose']: print (self.test_y_tensor.size(), 'size of the testing database output')


        #self.test_x_tensor = self.test_x_tensor.unsqueeze(1)
        self.test_dataset = torch.utils.data.TensorDataset(self.test_x_tensor, self.test_y_tensor)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, self.CTRL_PNL['batch_size'], shuffle=self.CTRL_PNL['shuffle'])

        if self.CTRL_PNL['CNN'] == "resnet":
            self.model_name = 'resnet34_1_'+str(self.opt.losstype)
        elif self.CTRL_PNL['CNN'] == "resnetunet":
            self.model_name = 'resnetunet34_1_'+str(self.opt.losstype)


        if self.opt.slp == 'synth': self.model_name += '_97495ct'
        elif self.opt.slp == 'mixedreal': self.model_name += '_108160ct'
        elif self.opt.slp == 'real': self.model_name += '_10665ct'

        self.model_name += '_128b_x'+str(self.CTRL_PNL['pmat_mult'])+'pm'


        if self.opt.no_reg_angles == False: self.model_name += '_rgangs'
        if self.opt.no_loss_betas == False: self.model_name += '_lb'
        if self.opt.noloss_htwt == True: self.model_name += '_nlhw'
        if self.opt.slp_depth == True or self.opt.slp == "real":
            if self.opt.no_blanket == False or self.opt.slp == "real":
                self.model_name += '_slpb'
            else:
                self.model_name += '_slpnb'
        else:
            self.model_name += '_ppnb'
        if self.opt.htwt == True: self.model_name += '_htwt'
        if self.opt.slpnoise == True: self.model_name += '_slpns'
        if self.opt.no_depthnoise == False: self.model_name += '_dpns'
        if self.opt.no_loss_root == False: self.model_name += '_rt'
        if self.opt.omit_pimg_cntct_sobel == True: self.model_name += '_opcs'
        if self.CTRL_PNL['depth_out_unet'] == True: self.model_name += '_dou'


        self.model_name += '_100e_'+str(0.0001)+'lr'

        if GPU == True:
            self.model = torch.load(FILEPATH + 'data_BP/convnets/'+self.model_name + '.pt', map_location={'cuda:' + str(self.opt.prev_device):'cuda:' + str(self.opt.device)}).cuda()
            self.model_CAL = torch.load(FILEPATH + 'data_BP/convnets/' + 'CAL_10665ct_128b_500e_0.0001lr.pt',map_location={'cuda:' + str(self.opt.prev_device): 'cuda:' + str(self.opt.device)}).cuda()
            self.model_betanet = torch.load(FILEPATH + 'data_BP/convnets/' + 'betanet_108160ct_128b_volfrac_500e_0.0001lr.pt', map_location={'cuda:' + str(self.opt.prev_device): 'cuda:' + str(self.opt.device)}).cuda()
        else:
            self.model = torch.load(FILEPATH + 'data_BP/convnets/'+self.model_name + '.pt', map_location='cpu')
            self.model_CAL = torch.load(FILEPATH + 'data_BP/convnets/' + 'CAL_10665ct_128b_500e_0.0001lr.pt',map_location='cpu')
            self.model_betanet = torch.load(FILEPATH + 'data_BP/convnets/' + 'betanet_108160ct_128b_volfrac_500e_0.0001lr.pt', map_location='cpu')
        self.model_smpl_pmr = fixedwt_smpl_pmr.SMPL_PMR(self.CTRL_PNL['loss_vector_type'], self.CTRL_PNL['batch_size'], verts_list = "all", CTRL_PNL = self.CTRL_PNL)




        print ('Loaded ConvNet.')

        self.validate_convnet('test')


    def validate_convnet(self, verbose=False, n_batches=None):

        if DROPOUT == True:
            self.model.train()
        else:
            self.model.eval()
        loss = 0.
        n_examples = 0

        for batch_i, batch in enumerate(self.test_loader):

            if DROPOUT == True:
                batch[0] = batch[0].repeat(25, 1, 1, 1)
                batch[1] = batch[1].repeat(25, 1)
            #self.model.train()


            scores, INPUT_DICT, OUTPUT_DICT = \
                UnpackDepthBatchLib().unpack_batch(batch, is_training=True, model=self.model, model_smpl_pmr=self.model_smpl_pmr, \
                                                   model_CAL=self.model_CAL, model_betanet=self.model_betanet, CTRL_PNL=self.CTRL_PNL)


            self.CTRL_PNL['first_pass'] = False

            self.criterion = nn.L1Loss()
            scores_zeros = Variable(torch.Tensor(np.zeros((batch[0].shape[0], scores.size()[1]))).type(dtype),
                                    requires_grad=False)

            loss_curr = self.criterion(scores[:, 10:34], scores_zeros[:, 10:34]).data.item() / 10.

            loss += loss_curr



            print (OUTPUT_DICT['batch_angles_est'].shape, n_examples)
            for item in range(OUTPUT_DICT['batch_angles_est'].shape[0]):
                if self.CTRL_PNL['compute_forward_maps'] == False:
                    self.dat['verts'].append(OUTPUT_DICT['verts'][item].astype(float32))
                    print(np.shape(self.dat['verts']))
                else:
                    self.dat['angles_est'].append(OUTPUT_DICT['batch_angles_est'][item].cpu().numpy().astype(float32))
                    self.dat['root_xyz_est'].append(OUTPUT_DICT['batch_root_xyz_est'][item].cpu().numpy().astype(float32))
                    self.dat['betas_est'].append(OUTPUT_DICT['batch_betas_est'][item].cpu().numpy().astype(float32))
                    self.dat['root_atan2_est'].append(OUTPUT_DICT['batch_root_atan2_est'][item].cpu().numpy().astype(float32))
                    if self.CTRL_PNL['CNN'] == 'resnet':
                        self.dat['mdm_est'].append(OUTPUT_DICT['batch_mdm_est'][item].cpu().numpy().astype(float32))
                        self.dat['cm_est'].append(OUTPUT_DICT['batch_cm_est'][item].cpu().numpy().astype(int16))
                        self.dat['bed_vertical_shift_est'].append(OUTPUT_DICT['bed_vertical_shift_est'][item].data.cpu().numpy().astype(float32))
                    elif self.CTRL_PNL['CNN'] == 'resnetunet':
                        self.dat['pimg_est'].append(OUTPUT_DICT['batch_pimg_est'][item].cpu().detach().numpy().astype(float32))
                        #self.dat['pimg_cntct_est'].append(OUTPUT_DICT['batch_pimg_cntct_est'][item].cpu().detach().numpy().astype(int16))
                        self.dat['dimg_est'].append(OUTPUT_DICT['batch_dimg_est'][item].cpu().detach().numpy().astype(float32))
                        #self.dat['dimg_cntct_est'].append(OUTPUT_DICT['batch_dimg_cntct_est'][item].cpu().detach().numpy().astype(int16))


            n_examples += self.CTRL_PNL['batch_size']
            #print n_examples

            if n_batches and (batch_i >= n_batches):
                break


            try:
                targets_print = torch.cat([targets_print, torch.mean(INPUT_DICT['batch_targets'], dim = 0).unsqueeze(0)], dim=0)
                targets_est_print = torch.cat([targets_est_print, torch.mean(OUTPUT_DICT['batch_targets_est'], dim = 0).unsqueeze(0)], dim=0)
            except:

                targets_print = torch.mean(INPUT_DICT['batch_targets'], dim = 0).unsqueeze(0)
                targets_est_print = torch.mean(OUTPUT_DICT['batch_targets_est'], dim = 0).unsqueeze(0)


            print (targets_print.shape, INPUT_DICT['batch_targets'].shape)
            print (targets_est_print.shape, OUTPUT_DICT['batch_targets_est'].shape)


            if GPU == True:
                error_norm, error_avg, _ = VisualizationLib().print_error_val(targets_print[-2:-1,:].cpu(),
                                                                                   targets_est_print[-2:-1,:].cpu(),
                                                                                   self.output_size_val,
                                                                                   self.CTRL_PNL['loss_vector_type'],
                                                                                   data='validate')
            else:
                error_norm, error_avg, _ = VisualizationLib().print_error_val(targets_print[-2:-1,:],
                                                                              targets_est_print[-2:-1,:],
                                                                                   self.output_size_val,
                                                                                   self.CTRL_PNL['loss_vector_type'],
                                                                                   data='validate')



        if self.test_x_tensor.size()[0] == 135:
            #if self.opt.cnn == 'resnet':
            #    self.dat = self.get_depth_cont_maps_from_synth(self.dat, self.gender)
            for item in self.dat:
                print(item, np.shape(self.dat[item]))
            pkl.dump(self.dat,open(FILEPATH + 'data_BP/mod1est_real/'+self.file_prefix + '_'+self.model_name+'.p', 'wb'))
        else:
            if self.CTRL_PNL['compute_forward_maps'] == False:
                pkl.dump(self.dat,open(self.file_prefix[:-2]+'_v2v.p', 'wb'))
            else:
                pkl.dump(self.dat,open(FILEPATH + 'data_BP/mod1est_synth/'+self.file_prefix[17:-2] + '_'+self.model_name+'.p', 'wb'))




    def get_depth_cont_maps_from_synth(self, realslp_data_dict, gender):


        betas = realslp_data_dict['body_shape']
        pose = realslp_data_dict['joint_angles']
        images = realslp_data_dict['images']
        realslp_data_dict['mesh_depth'] = []
        realslp_data_dict['mesh_contact'] = []
        root_xyz_shift = realslp_data_dict['root_xyz_shift']

        filler_taxels = []
        for i in range(27):
            for j in range(64):
                filler_taxels.append([i, j, 20000])
        filler_taxels = np.array(filler_taxels)

        if gender == 'f':
            model_path = '/home/henry/git/SMPL_python_v.1.0.0/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
            m = LOAD_MODEL(model_path)
        elif gender == 'm':
            model_path = '/home/henry/git/SMPL_python_v.1.0.0/smpl/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl'
            m = LOAD_MODEL(model_path)


        ct = 0
        for index in range(len(betas)):
            print(index)
            #index += 4

            for beta_idx in range(10):
                m.betas[beta_idx] = betas[index][beta_idx]
            for pose_idx in range(72):
                m.pose[pose_idx] = pose[index][pose_idx]

            images[index][images[index] > 0] += 1
            realslp_data_dict['images'][index] = images[index].astype(int8) #convert the original pmat to an int to save space
            curr_root_shift = np.array(root_xyz_shift[index])


            vertices = np.array(m.r) + curr_root_shift + np.array([self.CTRL_PNL['x_y_offset_synth'][0]/1000., -self.CTRL_PNL['x_y_offset_synth'][1]/1000., 0.0]) - np.array(m.J_transformed)[0:1, :]
            vertices_rot = np.copy(vertices)

            bend_loc = 48 * 0.0286

            bed_angle = 0.0

            vertices_rot[:, 1] = vertices[:, 2]*np.sin(bed_angle) - (bend_loc - vertices[:, 1])*np.cos(bed_angle) + bend_loc
            vertices_rot[:, 2] = vertices[:, 2]*np.cos(bed_angle) + (bend_loc - vertices[:, 1])*np.sin(bed_angle)

            vertices_rot = vertices_rot[vertices_rot[:, 1] >= bend_loc]
            vertices = np.concatenate((vertices[vertices[:, 1] < bend_loc], vertices_rot), axis = 0)

            vertices_taxel = vertices/0.0286
            vertices_taxel[:, 2] *= 1000
            vertices_taxel[:, 0] *= 1.04

            time_orig = time.time()

            #joints_taxel_int = (joints_taxel).astype(int)
            vertices_taxel_int = (vertices_taxel).astype(int)


            vertices_taxel_int = np.concatenate((filler_taxels, vertices_taxel_int), axis = 0)

            vertice_sorting_method = vertices_taxel_int[:, 0]*10000000 + vertices_taxel_int[:,1]*100000 + vertices_taxel_int[:,2]
            vertices_taxel_int = vertices_taxel_int[vertice_sorting_method.argsort()]

            vertice_sorting_method_2 = vertices_taxel_int[:, 0]*100 + vertices_taxel_int[:,1]
            unique_keys, indices = np.unique(vertice_sorting_method_2, return_index=True)

            vertices_taxel_int_unique = vertices_taxel_int[indices]


            #print vertices_taxel_int_unique.shape

            vertices_taxel_int_unique = vertices_taxel_int_unique[vertices_taxel_int_unique[:, 0] < 27, :]
            vertices_taxel_int_unique = vertices_taxel_int_unique[vertices_taxel_int_unique[:, 0] >= 0, :]
            vertices_taxel_int_unique = vertices_taxel_int_unique[vertices_taxel_int_unique[:, 1] < 64, :]
            vertices_taxel_int_unique = vertices_taxel_int_unique[vertices_taxel_int_unique[:, 1] >= 0, :]
            #print vertices_taxel_int_unique

            #print vertices_taxel_int_unique

            mesh_matrix = np.flipud(vertices_taxel_int_unique[:, 2].reshape(27, 64).T).astype(float)

            mesh_matrix[mesh_matrix == 20000] = 0
            mesh_matrix *= 0.0286


            #fix holes
            abc = np.zeros((66, 29, 4))
            abc[1:65, 1:28, 0] = np.copy(mesh_matrix)
            abc[1:65, 1:28, 0][abc[1:65, 1:28, 0] > 0] = 0
            abc[1:65, 1:28, 0] = abc[0:64, 0:27, 0] + abc[1:65, 0:27, 0] + abc[2:66, 0:27, 0] + \
                                 abc[0:64, 1:28, 0] + abc[2:66, 1:28, 0] + \
                                 abc[0:64, 2:29, 0] + abc[1:65, 2:29, 0] + abc[2:66, 2:29, 0]
            abc = abc[1:65, 1:28, :]
            abc[:, :, 0] /= 8
            abc[:, :, 1] = np.copy(mesh_matrix)
            abc[:, :, 1][abc[:, :, 1] < 0] = 0
            abc[:, :, 1][abc[:, :, 1] >= 0] = 1
            abc[:, :, 2] = abc[:, :, 0]*abc[:, :, 1]
            abc[:, :, 3] = np.copy(abc[:, :, 2])
            abc[:, :, 3][abc[:, :, 3] != 0] = 1.
            abc[:, :, 3] = 1-abc[:, :, 3]
            mesh_matrix = mesh_matrix*abc[:, :, 3]
            mesh_matrix += abc[:, :, 2]
            #print np.min(mesh_matrix), np.max(mesh_matrix)
            mesh_matrix = mesh_matrix.astype(int32)
            #print np.min(mesh_matrix), np.max(mesh_matrix)

            #make a contact matrix
            contact_matrix = np.copy(mesh_matrix)
            contact_matrix[contact_matrix >= 0] = 0
            contact_matrix[contact_matrix >= 0] = 0
            contact_matrix[contact_matrix < 0] = 1
            contact_matrix = contact_matrix.astype(bool)

            ct += 1

            realslp_data_dict['mesh_depth'].append(mesh_matrix)
            realslp_data_dict['mesh_contact'].append(contact_matrix)

        return realslp_data_dict

if __name__ == "__main__":
    #Initialize trainer with a training database file
    import optparse

    from optparse_lib import get_depthnet_options

    p = optparse.OptionParser()
    p = get_depthnet_options(p)

    opt, args = p.parse_args()
    opt.mod = 1

    if opt.hd == True:
        dana_lab_path = '/media/henry/multimodal_data_2/data/SLP/danaLab/'
    else:
        dana_lab_path = '/mnt/DADES2/SLP/SLP/danaLab/'




    FileNameInputLib1 = FileNameInputLib(opt, depth = False)
    if opt.slp == 'real' or opt.slp == 'mixedreal':
        train_database_file_f, train_database_file_m, train_subj_mass_list_f, train_subj_mass_list_m = FileNameInputLib1.get_dana_slp(True)
        train_files_f = [train_database_file_f]
        train_files_m = [train_database_file_m]

    test_database_file_f, test_database_file_m, test_subj_mass_list_f, test_subj_mass_list_m = FileNameInputLib1.get_dana_slp(False)
    test_files_f = [test_database_file_f]
    test_files_m = [test_database_file_m]


    FileNameInputLib2 = FileNameInputLib(opt, depth = True)
    if opt.slp == 'real' or opt.slp == 'mixedreal':
        train_database_file_depth_f, train_database_file_depth_m, train_subj_mass_list_f, train_subj_mass_list_m = FileNameInputLib2.get_dana_slp(True)
        train_files_f.append(train_database_file_depth_f)
        train_files_m.append(train_database_file_depth_m)

    test_database_file_depth_f, test_database_file_depth_m, test_subj_mass_list_f, test_subj_mass_list_m = FileNameInputLib2.get_dana_slp(False)
    test_files_f.append(test_database_file_depth_f)
    test_files_m.append(test_database_file_depth_m)



    if opt.slp == 'real' or opt.slp == 'mixedreal':
        for i in range(len(train_files_m[0])):
            #print("file prefix: ", train_files_m[0][i])
            p = PhysicalTrainer([[],[]], [[train_files_m[0][i]], [train_files_m[1][i]]], opt, train_files_m[0][i])
            p.init_convnet_test()
            #break

        for i in range(len(train_files_f[0])):
            p = PhysicalTrainer([[train_files_f[0][i]], [train_files_f[1][i]]], [[],[]], opt, train_files_f[0][i])
            p.init_convnet_test()
            #break

    for i in range(len(test_files_m[0])):
        p = PhysicalTrainer([[],[]], [[test_files_m[0][i]], [test_files_m[1][i]]], opt, test_files_m[0][i])
        p.init_convnet_test()
        #break

    for i in range(len(test_files_f[0])):
        p = PhysicalTrainer([[test_files_f[0][i]], [test_files_f[1][i]]], [[],[]], opt, test_files_f[0][i])
        p.init_convnet_test()
        #break



    if opt.slp == 'synth' or opt.slp == 'mixedreal':
        FileNameInputLib1 = FileNameInputLib(opt, depth = False)
        train_database_file_f, train_database_file_m = FileNameInputLib1.get_slpsynth_pressurepose(True, '')#_nonoise')
        test_database_file_f, test_database_file_m = FileNameInputLib1.get_slpsynth_pressurepose(False, '')#_nonoise')

        train_files_f = [train_database_file_f]
        test_files_f = [test_database_file_f]
        train_files_m = [train_database_file_m]
        test_files_m = [test_database_file_m]


        FileNameInputLib2 = FileNameInputLib(opt, depth = True)
        train_database_file_depth_f, train_database_file_depth_m = FileNameInputLib2.get_slpsynth_pressurepose(True, '')
        test_database_file_depth_f, test_database_file_depth_m = FileNameInputLib2.get_slpsynth_pressurepose(False, '')


        train_files_f.append(train_database_file_depth_f)
        test_files_f.append(test_database_file_depth_f)
        train_files_m.append(train_database_file_depth_m)
        test_files_m.append(test_database_file_depth_m)



        for i in range(len(train_files_m[0])):
            #print("file prefix: ", train_files_m[0][i])
            p = PhysicalTrainer([[],[]], [[train_files_m[0][i]], [train_files_m[1][i]]], opt, train_files_m[0][i])
            p.init_convnet_test()
            #break

        for i in range(len(train_files_f[0])):
            p = PhysicalTrainer([[train_files_f[0][i]], [train_files_f[1][i]]], [[],[]], opt, train_files_f[0][i])
            p.init_convnet_test()
            #break

        for i in range(len(test_files_m[0])):
            p = PhysicalTrainer([[],[]], [[test_files_m[0][i]], [test_files_m[1][i]]], opt, test_files_m[0][i])
            p.init_convnet_test()
            #break

        for i in range(len(test_files_f[0])):
            p = PhysicalTrainer([[test_files_f[0][i]], [test_files_f[1][i]]], [[],[]], opt, test_files_f[0][i])
            p.init_convnet_test()
            #break

