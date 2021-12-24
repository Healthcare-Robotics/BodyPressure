
#!/usr/bin/env python

#Bodies at Rest: Code to visualize real dataset.
#(c) Henry M. Clever
#Major updates made for CVPR release: December 10, 2019




txtfile = open("../FILEPATH.txt")
FILEPATH = txtfile.read().replace("\n", "")
txtfile.close()


import sys
sys.path.insert(0, '../lib_py')
sys.path.insert(-1,FILEPATH)
sys.path.insert(-1,FILEPATH+'smpl/smpl_webuser3')
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
print(sys.path, 'sys path for evaluate_depthreal_slp.py')

import numpy as np
import random
import copy
from functools import partial
from scipy import ndimage

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


import fixedwt_smpl_pmr_net as fixedwt_smpl_pmr
import lib_pyrender_depthREALTIME as libPyRender

import optparse

from visualization_lib_bp import VisualizationLib
from preprocessing_lib_bp import PreprocessingLib
from tensorprep_lib_bp import TensorPrepLib
from unpack_depth_batch_lib_bp import UnpackDepthBatchLib
import kinematics_lib_bp as kinematics_lib_br
from slp_prep_lib_bp import SLPPrepLib


try:
    from smpl.smpl_webuser.serialization import load_model
except:
    from smpl.smpl_webuser3.serialization import load_model

import os



#volumetric pose gen libraries
from time import sleep
from scipy.stats import mode
import os.path as osp
import imutils
from scipy.ndimage.filters import gaussian_filter

from scipy.ndimage.interpolation import zoom

import matplotlib.cm as cm #use cm.jet(list)

DATASET_CREATE_TYPE = 1

import cv2

import math
from random import shuffle
import torch
import torch.nn as nn

VERT_CUT, HORIZ_CUT = 0, 50
pre_VERT_CUT = 40
DROPOUT = False

MAT_WIDTH = 0.762  # metres
MAT_HEIGHT = 1.854  # metres
MAT_HALF_WIDTH = MAT_WIDTH / 2
NUMOFTAXELS_X = 64  # 73 #taxels
NUMOFTAXELS_Y = 27  # 30
NUMOFOUTPUTDIMS = 3
NUMOFOUTPUTNODES_TEST = 24
INTER_SENSOR_DISTANCE = 0.0286  # metres
LOW_TAXEL_THRESH_X = 0
LOW_TAXEL_THRESH_Y = 0
HIGH_TAXEL_THRESH_X = (NUMOFTAXELS_X - 1)
HIGH_TAXEL_THRESH_Y = (NUMOFTAXELS_Y - 1)
CAM_BED_DIST = 1.66

#MISC
import time as time
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

MAT_SIZE = (64, 27)


torch.set_num_threads(1)
if False:#torch.cuda.is_available():
    # Use for GPU
    GPU = True
    dtype = torch.cuda.FloatTensor
    torch.cuda.set_device(DEVICE)
    print ('######################### CUDA is available! #############################')
else:
    # Use for CPU
    GPU = False
    dtype = torch.FloatTensor
    print ('############################## USING CPU #################################')


class Viz3DPose():
    def __init__(self, opt):

        self.opt = opt
        if self.opt.no_blanket == True:
            self.opt.ctype = 'uncover'

        if opt.viz == '3D':
            pyrender3D = True
        else:
            pyrender3D = False

        self.pyRender = libPyRender.pyRenderMesh(render = pyrender3D)
            #self.pyRender = libPyRender.pyRenderMesh(render = False)

        self.weight_lbs = 0
        self.height_in  = 0

        self.index_queue = []

        self.reset_pose = False

        self.pressure = None

        self.CTRL_PNL = {}



        if opt.X_is == 'W':
            self.CTRL_PNL['CNN'] = 'resnet'
            self.CTRL_PNL['depth_out_unet'] = False
        elif opt.X_is == 'B':
            self.CTRL_PNL['CNN'] = 'resnetunet'
            self.CTRL_PNL['depth_out_unet'] = True
        else:
            print('you need to select a valid X_is. choose "W" for white box net or "B" for black box net.')
            sys.exit()


        self.CTRL_PNL['slp'] = opt.slp
        self.CTRL_PNL['nosmpl'] = opt.nosmpl
        self.CTRL_PNL['batch_size'] = 1
        self.CTRL_PNL['loss_vector_type'] = opt.losstype
        self.CTRL_PNL['mod'] = opt.mod
        self.CTRL_PNL['verbose'] = False
        self.CTRL_PNL['num_epochs'] = 101
        self.CTRL_PNL['incl_inter'] = True
        self.CTRL_PNL['shuffle'] = False
        self.CTRL_PNL['incl_ht_wt_channels'] = opt.htwt
        if opt.no_loss_root == True: self.CTRL_PNL['loss_root'] = False
        else: self.CTRL_PNL['loss_root'] = True
        if opt.no_reg_angles == True: self.CTRL_PNL['regr_angles'] = False
        else: self.CTRL_PNL['regr_angles'] = True
        if opt.no_loss_betas == True: self.CTRL_PNL['loss_betas'] = False
        else: self.CTRL_PNL['loss_betas'] = True
        if opt.no_depthnoise == True: self.CTRL_PNL['depth_noise'] = False
        else: self.CTRL_PNL['depth_noise'] = True
        self.CTRL_PNL['pimg_cntct_sum'] = opt.pimg_cntct_sum
        self.CTRL_PNL['omit_pimg_cntct_sobel'] = opt.omit_pimg_cntct_sobel
        self.CTRL_PNL['incl_pmat_cntct_input'] = False
        self.CTRL_PNL['num_input_channels'] = 3
        self.CTRL_PNL['GPU'] = GPU
        self.CTRL_PNL['dtype'] = dtype
        self.CTRL_PNL['repeat_real_data_ct'] = 1
        self.CTRL_PNL['mesh_recon_map_labels'] = False #can only be true if we have 100% synthetic data for training
        self.CTRL_PNL['mesh_recon_map_labels_test'] = False #can only be true is we have 100% synth for testing
        if self.opt.pmr == True:
            self.CTRL_PNL['mesh_recon_map_output'] = True #self.CTRL_PNL['mesh_recon_map_labels']
        else:
            self.CTRL_PNL['mesh_recon_map_output'] = False #self.CTRL_PNL['mesh_recon_map_labels']
        self.CTRL_PNL['mesh_recon_output'] = True

        self.CTRL_PNL['recon_map_input_est'] = False  #do this if we're working in a two-part regression
        self.CTRL_PNL['dropout'] = DROPOUT
        self.CTRL_PNL['depth_map_labels'] = False
        self.CTRL_PNL['depth_map_output'] = True
        self.CTRL_PNL['adjust_ang_from_est'] = False#self.CTRL_PNL['depth_map_input_est']  # holds betas and root same as prior estimate
        self.CTRL_PNL['clip_sobel'] = True
        self.CTRL_PNL['clip_betas'] = True
        self.CTRL_PNL['mesh_bottom_dist'] = True
        self.CTRL_PNL['full_body_rot'] = True  # False
        self.CTRL_PNL['normalize_per_image'] = False
        self.CTRL_PNL['normalize_std'] = False
        self.CTRL_PNL['all_tanh_activ'] = True  # False
        self.CTRL_PNL['L2_contact'] = True  # False
        self.CTRL_PNL['pmat_mult'] = int(1)
        self.CTRL_PNL['slp_noise'] = opt.slpnoise
        self.CTRL_PNL['cal_noise_amt'] = 0.2
        self.CTRL_PNL['output_only_prev_est'] = False
        self.CTRL_PNL['double_network_size'] = False
        self.CTRL_PNL['first_pass'] = True
        self.CTRL_PNL['align_procr'] = False
        self.CTRL_PNL['depth_in'] = True
        self.CTRL_PNL['onlyhuman_labels'] = False
        self.CTRL_PNL['slp_real'] = True
        self.CTRL_PNL['train_only_betanet'] = False
        self.CTRL_PNL['train_only_CAL'] = False
        self.CTRL_PNL['compute_forward_maps'] = True
        self.CTRL_PNL['v2v'] = opt.v2v
        self.CTRL_PNL['x_y_offset_synth'] = [12, -35]#[-7, -45]#[200, 200]#


        #if self.opt.slp == 'real':
        self.CTRL_PNL['clean_slp_depth'] = False
        #else:
        #    self.CTRL_PNL['clean_slp_depth'] = True



        if self.CTRL_PNL['recon_map_input_est'] == True: #for a two part regression
            self.CTRL_PNL['num_input_channels'] += 2
        self.CTRL_PNL['num_input_channels_batch0'] = np.copy(self.CTRL_PNL['num_input_channels'])
        self.CTRL_PNL['num_input_channels'] += 1



        pmat_std_from_mult = ['N/A', 14.64204661, 19.90905848383454, 23.07018866032369, 0.0, 25.50538629767412]

        self.CTRL_PNL['norm_std_coeffs'] = [1. / 41.80684362163343,  # contact
                                            1. / 45.08513083167194,  # neg est depth
                                            1. / 43.55800622930469,  # cm est
                                            1. / pmat_std_from_mult[int(self.CTRL_PNL['pmat_mult'])],  # pmat x5
                                            1. / 1.0,  # bed height mat
                                            1. / 1.0,  # OUTPUT DO NOTHING
                                            1. / 1.0,  # OUTPUT DO NOTHING
                                            1. / 30.216647403350,  # weight
                                            1. / 14.629298141231]  # height


        if self.CTRL_PNL['normalize_std'] == False:
            for i in range(9):
                self.CTRL_PNL['norm_std_coeffs'][i] *= 0.
                self.CTRL_PNL['norm_std_coeffs'][i] += 1.


        self.verts_list = "all"


        self.count = 0

        self.CTRL_PNL['filepath_prefix'] = '/home/henry/'
        self.CTRL_PNL['aws'] = False
        self.CTRL_PNL['lock_root'] = False


        self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)
        self.output_size_test = (NUMOFOUTPUTNODES_TEST, NUMOFOUTPUTDIMS)

        self.final_dataset = {}

        self.model_smpl_pmr = fixedwt_smpl_pmr.SMPL_PMR(self.CTRL_PNL['loss_vector_type'], self.CTRL_PNL['batch_size'], verts_list=self.verts_list, CTRL_PNL=self.CTRL_PNL)


    def load_smpl(self):


        #if len(testing_database_file_m) > 0 and len(testing_database_file_f) == 0:
        #    model_path = '../../../git/SMPL_python_v.1.0.0/smpl/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl'
        #elif len(testing_database_file_f) > 0 and len(testing_database_file_m) == 0:
        #model_path = '../../../git/SMPL_python_v.1.0.0/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
        #else:
        #    sys.exit("can only test f or m at one time, not both.")


        model_path_m = FILEPATH+'smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
        self.m_male = load_model(model_path_m)


        model_path_f = FILEPATH+'smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
        self.m_female = load_model(model_path_f)





    def load_slp_data(self):

        if self.opt.p_idx == 0:
            test_crit = 'last12'
        else:
            test_crit = None

        if test_crit == "all_subjects":
            all_subj_str_list = []
            for i in range(1, 103):
                all_subj_str_list.append('%05d' % (i))
        elif test_crit == "last12":
            all_subj_str_list = []
            # for i in range(69, 81):
            # for i in range(71, 81):
            for i in range(81, 103):
                # for i in range(1, 91):
                all_subj_str_list.append('%05d' % (i))
        else:
            all_subj_str_list = ['%05d' % (self.opt.p_idx)]

        phys_arr = np.load(FILEPATH + 'data_BP/SLP/danaLab/physiqueData.npy')
        phys_arr[:, [2, 0]] = phys_arr[:, [0, 2]]
        testing_database_file_f = []
        testing_database_file_m = []
        all_f_subj_mass_ht_list = []
        all_m_subj_mass_ht_list = []

        for i in range(1, 103):
            if phys_arr[int(i) - 1][0] > 100:
                print(i, phys_arr[int(i) - 1][0], phys_arr[int(i) - 1][1], 'mass ht')

        for some_subj in all_subj_str_list:
            gender_bin = phys_arr[int(some_subj) - 1][2]
            print(phys_arr[int(some_subj) - 1])
            if some_subj == '00007': continue
            if int(gender_bin) == 0:
                all_f_subj_mass_ht_list.append([phys_arr[int(some_subj) - 1][0], phys_arr[int(some_subj) - 1][1]])
                testing_database_file_f.append(some_subj)
            else:
                all_m_subj_mass_ht_list.append([phys_arr[int(some_subj) - 1][0], phys_arr[int(some_subj) - 1][1]])
                testing_database_file_m.append(some_subj)







        if self.opt.no_blanket == True:
            test_dat_f_slp = SLPPrepLib().load_slp_files_to_database(testing_database_file_f, dana_lab_path, PM = 'uncover', depth = 'uncover', color = None, mass_ht_list=all_f_subj_mass_ht_list, markers_gt_type = '3D', use_pc = True, pm_adjust_mm = self.CTRL_PNL['x_y_offset_synth'])
            test_dat_m_slp = SLPPrepLib().load_slp_files_to_database(testing_database_file_m, dana_lab_path, PM = 'uncover', depth = 'uncover', color = None, mass_ht_list=all_m_subj_mass_ht_list, markers_gt_type = '3D', use_pc = True, pm_adjust_mm = self.CTRL_PNL['x_y_offset_synth'])

        else:
            test_dat_f_slp = SLPPrepLib().load_slp_files_to_database(testing_database_file_f, dana_lab_path, PM = self.opt.ctype, depth = self.opt.ctype, color = None, mass_ht_list=all_f_subj_mass_ht_list, markers_gt_type = '3D', use_pc = True, pm_adjust_mm = self.CTRL_PNL['x_y_offset_synth'])
            test_dat_m_slp = SLPPrepLib().load_slp_files_to_database(testing_database_file_m, dana_lab_path, PM = self.opt.ctype, depth = self.opt.ctype, color = None, mass_ht_list=all_m_subj_mass_ht_list, markers_gt_type = '3D', use_pc = True, pm_adjust_mm = self.CTRL_PNL['x_y_offset_synth'])

            test_dat_f_slp_nb = SLPPrepLib().load_slp_files_to_database(testing_database_file_f, dana_lab_path, PM = 'uncover', depth = 'uncover', color = None, mass_ht_list=all_f_subj_mass_ht_list, markers_gt_type = '3D', use_pc = True, pm_adjust_mm = self.CTRL_PNL['x_y_offset_synth'])
            test_dat_m_slp_nb = SLPPrepLib().load_slp_files_to_database(testing_database_file_m, dana_lab_path, PM = 'uncover', depth = 'uncover', color = None, mass_ht_list=all_m_subj_mass_ht_list, markers_gt_type = '3D', use_pc = True, pm_adjust_mm = self.CTRL_PNL['x_y_offset_synth'])



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

        try:
            len_f =  np.shape(test_dat_f_slp['images'])[0]
        except:
            len_f = 0
        try:
            len_m =  np.shape(test_dat_m_slp['images'])[0]
        except:
            len_m = 0


        test_x = np.zeros((len_f + len_m, x_map_ct, 64, 27)).astype(np.float32)

        #allocate pressure images
        test_x = TensorPrepLib().prep_images(test_x, test_dat_f_slp, test_dat_m_slp, None, None, filter_sigma = 0.5, start_map_idx = pmat_gt_idx)


        self.mesh_reconstruction_maps = None
        self.reconstruction_maps_input_est = None

        if self.opt.no_blanket == True or self.opt.ctype == 'uncover':
            test_x = TensorPrepLib().prep_depth_input_images(test_x, test_dat_f_slp, test_dat_m_slp, None, None, start_map_idx = depth_in_idx, depth_type = 'no_blanket')
        else:
            test_x = TensorPrepLib().prep_depth_input_images(test_x, test_dat_f_slp, test_dat_m_slp, None, None, start_map_idx = depth_in_idx, depth_type = 'all_meshes')



        self.test_x_tensor = torch.Tensor(test_x)


        test_y_flat = []  # Initialize the ground truth listhave
        test_y_flat = SLPPrepLib().prep_labels_slp(test_y_flat, test_dat_f_slp, num_repeats = 1,
                                                    z_adj = -0.075, gender = "f", is_synth = True, markers_gt_type = '3D')#,  x_y_adjust_mm = [20, 10]) #not sure if we should us is_synth true or false???
        test_y_flat = SLPPrepLib().prep_labels_slp(test_y_flat, test_dat_m_slp, num_repeats = 1,
                                                    z_adj = -0.075, gender = "m", is_synth = True, markers_gt_type = '3D')#,  x_y_adjust_mm = [20, 10])

        test_y_flat = np.array(test_y_flat)
        self.test_y_tensor = torch.Tensor(test_y_flat)

        self.joint_error_list = []
        self.target_list = []
        self.score_list = []
        self.hd_th_dist_list = []

        self.RESULTS_DICT = {}
        if self.opt.pimgerr == True:
            self.RESULTS_DICT['pmat'] = []
            self.RESULTS_DICT['pmat_est'] = []
        else:
            self.RESULTS_DICT['body_roll_rad'] = []
            self.RESULTS_DICT['v_to_gt_err'] = []
            self.RESULTS_DICT['v_limb_to_gt_err'] = []
            self.RESULTS_DICT['gt_to_v_err'] = []
            self.RESULTS_DICT['precision'] = []
            self.RESULTS_DICT['recall'] = []
            self.RESULTS_DICT['overlap_d_err'] = []
            self.RESULTS_DICT['all_d_err'] = []
            self.RESULTS_DICT['betas'] = []
            self.RESULTS_DICT['pimg_error'] = []
            self.RESULTS_DICT['vertex_pressure_list_EST'] = []
            self.RESULTS_DICT['vertex_pressure_list_GT'] = []
            self.RESULTS_DICT['vertex_pressure_list_abs_err'] = []
            self.RESULTS_DICT['vertex_pressure_list_sq_err'] = []

        if self.opt.no_blanket == True:
            dat_pc = test_dat_m_slp['pc'] + test_dat_f_slp['pc']
        else:
            dat_pc = test_dat_m_slp_nb['pc'] + test_dat_f_slp_nb['pc']



        return test_x, dat_pc





    def get_SMPL(self, OUTPUT_DICT, gender):

        print(OUTPUT_DICT['batch_betas_est_post_clip'].size())
        print(OUTPUT_DICT['batch_angles_est_post_clip'].size())
        print(OUTPUT_DICT['batch_root_xyz_est_post_clip'].size())

        betas = OUTPUT_DICT['batch_betas_est_post_clip'][0, :].detach().numpy()
        angles = OUTPUT_DICT['batch_angles_est_post_clip'][0, :].detach().numpy().reshape(72)
        root_shift_est = OUTPUT_DICT['batch_root_xyz_est_post_clip'][0, :].detach().numpy()
        #root_shift_est = torch.mean(batch1[:, 154:157], dim=0).numpy()
        root_shift_est[1] *= -1
        root_shift_est[2] *= -1

        R_root = kinematics_lib_br.matrix_from_dir_cos_angles(angles[0:3])
        flip_root_euler = np.pi
        flip_root_R = kinematics_lib_br.eulerAnglesToRotationMatrix([flip_root_euler, 0.0, 0.0])
        angles[0:3] = kinematics_lib_br.dir_cos_angles_from_matrix(np.matmul(flip_root_R, R_root))


        if gender == "f":
            for beta in range(betas.shape[0]):
                self.m_female.betas[beta] = betas[beta]
            for angle in range(angles.shape[0]):
                self.m_female.pose[angle] = angles[angle]

            smpl_verts = np.array(self.m_female.r)
            for s in range(root_shift_est.shape[0]):
                smpl_verts[:, s] += (root_shift_est[s] - float(self.m_female.J_transformed[0, s]))

            print('FEMALE')

        elif gender == "m":
            for beta in range(betas.shape[0]):
                self.m_male.betas[beta] = betas[beta]
            for angle in range(angles.shape[0]):
                self.m_male.pose[angle] = angles[angle]

            smpl_verts = np.array(self.m_male.r)
            for s in range(root_shift_est.shape[0]):
                smpl_verts[:, s] += (root_shift_est[s] - float(self.m_male.J_transformed[0, s]))

            print('MALE')

        return smpl_verts





    def estimate_pose(self, depth, point_cloud, project_pressure, gender):

        depth = np.expand_dims(np.stack((depth[0:64, 0:27],
                                                depth[0:64, 27:54],
                                                depth[64:128, 0:27],
                                                depth[64:128, 27:54]), axis = 0), 0)

        bedangle = 0

        mat_size = (64, 27)


        pmat_stack = np.expand_dims([np.zeros(mat_size)], 0)
        pmat_stack = torch.Tensor(pmat_stack)

        ground_truth = np.zeros((1, 162))

        if gender == "m":
            ground_truth[0, 157] = 0.0
            ground_truth[0, 158] = 1.0
            m = self.m_male
        elif gender == "f":
            ground_truth[0, 157] = 1.0
            ground_truth[0, 158] = 0.0
            m = self.m_female

        ground_truth[0, 159] = 1.0

        batch1 = torch.Tensor(ground_truth)

        batch = []
        depth_stack = torch.Tensor(depth)

        batch.append(torch.cat((pmat_stack, depth_stack), dim = 1))
        batch.append(batch1)


        batch0 = batch[0].clone()


        NUMOFOUTPUTDIMS = 3
        NUMOFOUTPUTNODES_TRAIN = 24
        self.output_size_train = (NUMOFOUTPUTNODES_TRAIN, NUMOFOUTPUTDIMS)

        self.CTRL_PNL['recon_map_input_est'] = False
        self.CTRL_PNL['adjust_ang_from_est'] = False
        self.CTRL_PNL['mod'] = 1
        if self.opt.pmr == True and self.opt.mod == 2:
            self.CTRL_PNL['compute_forward_maps'] = True
        else:
            self.CTRL_PNL['compute_forward_maps'] = False

        scores, INPUT_DICT, OUTPUT_DICT = UnpackDepthBatchLib().unpack_batch(batch, is_training=False,
                                                                             model=self.model,
                                                                             model_smpl_pmr = self.model_smpl_pmr,
                                                                             model_CAL = self.model_CAL,
                                                                             model_betanet = self.model_betanet,
                                                                             CTRL_PNL = self.CTRL_PNL)

        if self.opt.pmr == True:
            self.CTRL_PNL['compute_forward_maps'] = False
        if self.model2 is not None:
            self.CTRL_PNL['mod'] = 2

            print("Using model 2")

            if self.opt.pmr == True:
                mdm_est_pos = OUTPUT_DICT['batch_mdm_est'].clone().unsqueeze(1)*-1  # / 16.69545796387731
                mdm_est_pos[mdm_est_pos < 0] = 0
                cm_est = OUTPUT_DICT['batch_cm_est'].clone().unsqueeze(1) * 100  # / 43.55800622930469
                batch_cor = []
                batch_cor.append(torch.cat((mdm_est_pos.type(torch.FloatTensor),
                                            cm_est.type(torch.FloatTensor),
                                            batch0[:, 0:, :, :]), dim=1))

            elif self.CTRL_PNL['depth_out_unet']:
                pimg_est = OUTPUT_DICT['batch_pimg_est'].clone().unsqueeze(1).detach()

                dimg_est = OUTPUT_DICT['batch_dimg_est'].clone().unsqueeze(1).detach()
                dimg_est = torch.cat((dimg_est[:, :, 0:64, 0:27],
                                      dimg_est[:, :, 0:64, 27:54],
                                      dimg_est[:, :, 64:128, 0:27],
                                      dimg_est[:, :, 64:128, 27:54]),
                                      dim = 1)
                batch_cor = []
                batch_cor.append(torch.cat((pimg_est.type(torch.FloatTensor),
                                            torch.zeros_like(pimg_est).type(torch.FloatTensor),
                                            batch0[:, 0:, :, :],
                                            dimg_est.type(torch.FloatTensor),
                                            ), dim=1))

                print(batch_cor[0].size())

            batch_cor.append(torch.cat((batch1,
                                        OUTPUT_DICT['batch_betas_est'].cpu(),
                                        OUTPUT_DICT['batch_angles_est'].cpu(),
                                        OUTPUT_DICT['batch_root_xyz_est'].cpu(),
                                        OUTPUT_DICT['batch_root_atan2_est'].cpu()), dim=1))
            if self.CTRL_PNL['depth_out_unet'] == True:
                batch_cor[1] = torch.cat((batch_cor[1],
                                          torch.zeros_like(batch_cor[1][:, 0:1])), dim=1)
            else:
                batch_cor[1] = torch.cat((batch_cor[1],
                                          OUTPUT_DICT['bed_vertical_shift_est'].cpu()), dim=1)

            self.CTRL_PNL['recon_map_input_est'] = True
            self.CTRL_PNL['adjust_ang_from_est'] = True
            scores, INPUT_DICT, OUTPUT_DICT = UnpackDepthBatchLib().unpack_batch(batch_cor, is_training=False,
                                                                                 model=self.model2,
                                                                                 model_smpl_pmr = self.model_smpl_pmr,
                                                                                 model_CAL = self.model_CAL,
                                                                                 model_betanet = self.model_betanet,
                                                                                 CTRL_PNL = self.CTRL_PNL)

        self.CTRL_PNL['first_pass'] = False

        weight_kg_est = float(OUTPUT_DICT['batch_weight_kg_est'].detach().cpu().numpy())
        height_m_est = float(OUTPUT_DICT['batch_height_est'].detach().cpu().numpy())


        print('estimated weight kg:', weight_kg_est)
        print('estimated height m:', height_m_est)


        smpl_verts = OUTPUT_DICT['verts'][0, :, :]



        smpl_faces = np.array(m.f)

        smpl_verts_v2 = self.get_SMPL(OUTPUT_DICT, gender)
        print(smpl_verts, 'v1')
        print(smpl_verts_v2, 'v2')

        camera_point = [1.09898028, 0.46441343, -CAM_BED_DIST]


        if self.opt.viz == '2D':
            viz_type = "2D"
        elif self.opt.viz == '3D':
            viz_type = "3D"
        else:
            viz_type = None


        self.tar_sample = INPUT_DICT['batch_targets']

        self.tar_sample = self.tar_sample.view(-1, 24, 3)
        self.tar_sample = self.tar_sample[0, :].squeeze() / 1000
        sc_sample = OUTPUT_DICT['batch_targets_est'].clone()
        sc_sample = sc_sample[0, :].squeeze() / 1000
        sc_sample = torch.cat((sc_sample, torch.Tensor([0.0, 0.0, 0.0])), dim=0)

        human_joints_3D_est = sc_sample.view([self.output_size_train[0] + 1, self.output_size_train[1]]).cpu().numpy()


        im_display_idx = 0
        self.VIZ_DICT = {}
        self.VIZ_DICT = VisualizationLib().get_depthnet_viz_maps(im_display_idx, INPUT_DICT, OUTPUT_DICT, self.VIZ_DICT, self.CTRL_PNL)
        self.VIZ_DICT = VisualizationLib().get_fcn_recon_viz_maps(im_display_idx, INPUT_DICT, OUTPUT_DICT,  self.VIZ_DICT, self.CTRL_PNL)
        self.CTRL_PNL['recon_map_input_est'] = False


        if self.opt.pmr == True or self.CTRL_PNL['depth_out_unet'] == True:
            pmatV = self.VIZ_DICT['p_img'].data.numpy()
            pmatV_est = self.VIZ_DICT['p_img_est'].data.numpy()

            if self.opt.pimgerr == True:
                self.RESULTS_DICT['pmat'].append(pmatV)
                self.RESULTS_DICT['pmat_est'].append(pmatV_est)

            weight_kg_est = float(OUTPUT_DICT['batch_weight_kg_est'].detach().cpu().numpy())

            print(np.max(pmatV_est), np.sum(pmatV_est), weight_kg_est, 'max pmatVest, sum pmatVest, and est weight')

            self.VIZ_DICT['p_img_est'] =pmatV_est*1.



        if viz_type == "2D":
            VisualizationLib().visualize_depth_net(VIZ_DICT = self.VIZ_DICT,
                                                      targets_raw = self.tar_sample.cpu(), scores_net1 = torch.Tensor(human_joints_3D_est),
                                                      block=True, max_depth = 2200, is_testing = True)


        elif viz_type == "3D":
            #depth = INPUT_DICT_mod1['batch_images'][0, int(INPUT_DICT_mod1['batch_images'].size()[1]) - 2, :].clone().numpy().reshape(128, 54)
            #self.depth_im_render = zoom(depth, 3.435, order = 1)

            self.pyRender.render_mesh_pc_bed_pyrender_everything(smpl_verts, smpl_faces, camera_point,
                                                                 bedangle, pc=point_cloud, pmat_est = pmatV_est,
                                                                 scores=human_joints_3D_est, project_pressure=project_pressure)


        return smpl_verts, human_joints_3D_est, m




    def load_deep_model(self):

        if self.opt.losstype == 'anglesDC':
            NETWORK_1 = "anglesDC_"
            NETWORK_2 = "anglesDC_"
        elif self.opt.losstype == 'direct':
            NETWORK_1 = "direct_"
            NETWORK_2 = "direct_"

        if self.opt.small == True:
            NETWORK_1 += "46000ct_"
            NETWORK_2 += "46000ct_"
        elif self.opt.slp == "real":
            #NETWORK_1 += "9315ct_"
            #NETWORK_2 += "9315ct_"
            NETWORK_1 += "10665ct_"
            NETWORK_2 += "10665ct_"
        elif self.opt.slp == "synth":
            #NETWORK_1 += "85114ct_"
            #NETWORK_2 += "85114ct_"
            NETWORK_1 += "97495ct_"
            NETWORK_2 += "97495ct_"
        elif self.opt.slp == "mixed":
            NETWORK_1 += "183114ct_"
            NETWORK_2 += "183114ct_"
        elif self.opt.slp == "mixedreal":
            #NETWORK_1 += "94429ct_"
            #NETWORK_2 += "94429ct_"
            NETWORK_1 += "108160ct_"
            NETWORK_2 += "108160ct_"
        else:
            NETWORK_1 += "184000ct_"
            NETWORK_2 += "184000ct_"

        if self.opt.go200 == True:
            NETWORK_1 += "128b_x1pm"
            NETWORK_2 += "128b_x1pm"
        elif self.opt.pmr == True:
            if self.opt.mod == 1:
                NETWORK_1 += "128b_x1pm_0.5rtojtdpth"

            elif self.opt.mod == 2:
                NETWORK_1 += "128b_x1pm"#_0.5rtojtdpth"
                NETWORK_2 += "128b_x1pm_0.5rtojtdpth_depthestin_angleadj"
        elif self.opt.X_is == 'B' and self.opt.mod == 2:
            NETWORK_1 += "128b_x1pm"
            NETWORK_2 += "128b_x1pm_depthestin_angleadj"
        else:
            NETWORK_1 += "128b_x1pm"
            NETWORK_2 += "128b_x1pm_angleadj"

        if self.opt.no_reg_angles == False:
            NETWORK_1 += '_rgangs'
            NETWORK_2 += '_rgangs'

        if self.opt.no_loss_betas == False:
            NETWORK_1 += '_lb'
            NETWORK_2 += '_lb'

        if self.opt.v2v == True:
            if self.opt.mod == 1:
                NETWORK_1 += '_lv2v'
            NETWORK_2 += '_lv2v'

        if self.opt.noloss_htwt == True:
            NETWORK_1 += '_nlhw'
            NETWORK_2 += '_nlhw'

        if self.opt.no_blanket == False or self.opt.slp == 'real':
            NETWORK_1 += "_slpb"
            NETWORK_2 += "_slpb"
        else:
            NETWORK_1 += "_slpnb"
            NETWORK_2 += "_slpnb"

        if self.opt.htwt == True:
            NETWORK_1 += "_htwt"
            NETWORK_2 += "_htwt"

        if self.opt.no_depthnoise == False:
            NETWORK_1 += "_dpns"
            NETWORK_2 += "_dpns"

        if self.opt.slpnoise == True:
            NETWORK_1 += "_slpns"
            NETWORK_2 += "_slpns"

        if self.opt.no_loss_root == False:
            NETWORK_1 += "_rt"
            NETWORK_2 += "_rt"

        if self.opt.half_shape_wt == True:
            NETWORK_1 += "_hsw"
            NETWORK_2 += "_hsw"

        if self.opt.X_is == 'B':
            NETWORK_1 += "_dou"
            NETWORK_2 += "_dou"
        elif self.opt.X_is == 'W':
            pass
        else:
            print('you need to select a valid X_is. choose "W" for white box net or "B" for black box net.')
            sys.exit()


        NETWORK_1 += "_100e"
        NETWORK_2 += "_40e"


        if self.opt.X_is == 'B':
            if self.opt.small == True:
                filename1 = FILEPATH + "data_BP/convnets/resnetunet34_1_" + NETWORK_1 + "_0.0001lr.pt"
                filename2 = FILEPATH + "data_BP/convnets/resnetunet34_2_" + NETWORK_2 + "_0.0001lr.pt"
            else:
                filename1 = FILEPATH + "data_BP/convnets/resnetunet34_1_" + NETWORK_1 + "_0.0001lr.pt"
                filename2 = FILEPATH + "data_BP/convnets/resnetunet34_2_" + NETWORK_2 + "_0.0001lr.pt"

            if self.opt.mod == 1:
                filename2 = None
        else:
            if self.opt.small == True:
                filename1 = FILEPATH + "data_BP/convnets/resnet34_1_" + NETWORK_1 + "_0.0001lr.pt"#
                filename2 = FILEPATH + "data_BP/convnets/resnet34_2_" + NETWORK_2 + "_0.0001lr.pt"#_noiseyCorr.pt"
            elif self.opt.slp == "real":
                filename1 = FILEPATH + "data_BP/convnets/resnet34_1_" + NETWORK_1 + "_0.0001lr.pt"
                filename2 = FILEPATH + "data_BP/convnets/resnet34_2_" + NETWORK_2 + "_0.0001lr.pt"#_noiseyCorr.pt"
            else:
                filename1 = FILEPATH + "data_BP/convnets/resnet34_1_" + NETWORK_1 + "_0.0001lr.pt"
                filename2 = FILEPATH + "data_BP/convnets/resnet34_2_" + NETWORK_2 + "_0.0001lr.pt"#_noiseyCorr.pt"

            if self.opt.pmr == False:
                filename2 = None



        if GPU == True:
            for i in range(0, 8):
                try:
                    self.model = torch.load(filename1, map_location={'cuda:' + str(i): 'cuda:0'})
                    self.model = self.model.cuda().eval()
                    print ("Network 1 loaded.")
                    break
                except:
                    pass
            if filename2 is not None and self.opt.mod == 2:
                for i in range(0, 8):
                    try:
                        self.model2 = torch.load(filename2, map_location={'cuda:' + str(i): 'cuda:0'})
                        self.model2 = self.model2.cuda().eval()
                        print ("Network 2 loaded.")
                        break
                    except:
                        pass
            else:
                self.model2 = None
        else:

            if sys.version_info.major < 3:
                print(sys.version_info.major)
                self.model = torch.load(filename1, map_location='cpu')
            else:
                print(sys.version_info.major)
                pkl.load = partial(pkl.load, encoding='latin1')
                pkl.Unpickler = partial(pkl.Unpickler, encoding='latin1')
                self.model = torch.load(filename1, map_location='cpu', pickle_module=pkl)
                print("got here")

            self.model = self.model.eval()
            print ("Network 1 loaded.")
            if filename2 is not None and self.opt.mod == 2:
                if sys.version_info.major < 3:
                    self.model2 = torch.load(filename2, map_location='cpu')
                else:
                    self.model2 = torch.load(filename2, map_location='cpu', pickle_module=pkl)

                self.model2 = self.model2.eval()
                print ("Network 2 loaded.")
            else:
                self.model2 = None


        if self.opt.X_is == 'W':
            if self.opt.slp == 'synth':
                self.model_CAL = torch.load(FILEPATH + 'data_BP/convnets/CAL_97495ct_128b_500e_0.0001lr.pt',map_location='cpu')
            else:
                self.model_CAL = torch.load(FILEPATH + 'data_BP/convnets/CAL_10665ct_128b_500e_0.0001lr.pt',map_location='cpu')
        else:
            self.model_CAL = None
        self.model_betanet = torch.load(FILEPATH + 'data_BP/convnets/betanet_108160ct_128b_volfrac_500e_0.0001lr.pt', map_location='cpu')








if __name__ ==  "__main__":

    import optparse
    from optparse_lib import get_depthnet_options

    p = optparse.OptionParser()

    p = get_depthnet_options(p)

    p.add_option('--mod', action='store', type = 'int', dest='mod', default=1,
                 help='Choose a network.')

    p.add_option('--p_idx', action='store', type='int', dest='p_idx', default=0,
                 help='Choose a participant. Enter a number from 1 to 100.')

    p.add_option('--pose_num', action='store', type='int', dest='pose_num', default=0,
                 help='Choose a pose number between 1 and 45.')

    p.add_option('--viz', action='store', dest='viz', default='None',
                 help='Visualize training. specify `2D` or `3D`.')

    p.add_option('--ctype', action='store', dest='ctype', default='None',
                 help='Visualize training. specify `uncover` or `cover1` or `cover2`.')

    p.add_option('--pimgerr', action='store_true', dest='pimgerr', default=False,
                 help='Compute pressure image error.')


    opt, args = p.parse_args()


    if opt.ctype not in ['uncover', 'cover1', 'cover2']:
        print("need to specify valid ctype of specify `uncover` or `cover1` or `cover2`.")
        sys.exit()


    if opt.hd == True:
        dana_lab_path = '/media/henry/multimodal_data_2/data/SLP/danaLab/'
    else:
        dana_lab_path = FILEPATH + 'data_BP/SLP/danaLab/'





    V3D = Viz3DPose(opt)
    V3D.load_deep_model()
    V3D.load_smpl()




    if opt.pose_num == 0:

        depth = np.zeros((128, 54)) #this is the depth image.
        point_cloud = np.zeros((100, 3)) #this is the point cloud. it is optional. You can just set point_cloud=None if you don't want to use it.

        smpl_verts, human_joints_3D_est, m = V3D.estimate_pose(depth, point_cloud=point_cloud, project_pressure=True, gender = "f")







        print(smpl_verts.shape, human_joints_3D_est.shape)

        print(np.min(smpl_verts[:, 0]), np.max(smpl_verts[:, 0]))
        print(np.min(smpl_verts[:, 1]), np.max(smpl_verts[:, 1]))
        print(np.min(smpl_verts[:, 2]), np.max(smpl_verts[:, 2]))

        print(np.min(human_joints_3D_est[:, 0]), np.max(human_joints_3D_est[:, 0]))
        print(np.min(human_joints_3D_est[:, 1]), np.max(human_joints_3D_est[:, 1]))
        print(np.min(human_joints_3D_est[:, 2]), np.max(human_joints_3D_est[:, 2]))
        input("Press Enter to continue...")


    else:
        test_x, dat_pc = V3D.load_slp_data()

        for im_num in range(opt.pose_num, np.shape(test_x)[0]):
            print("TESTING IM NUM ", im_num)

            depth_orig = test_x[im_num:im_num + 1, 1:, :, :]

            depth = np.concatenate((np.concatenate((depth_orig[0, 0, :, :], depth_orig[0, 1, :, :]), axis=1),
                                    np.concatenate((depth_orig[0, 2, :, :], depth_orig[0, 3, :, :]), axis=1)), axis=0)
            point_cloud = dat_pc[im_num]


            print(depth.shape, point_cloud.shape, 'depth and point cloud shape')





            smpl_verts, human_joints_3D_est, m = V3D.estimate_pose(depth, point_cloud=point_cloud, project_pressure=True, gender = "m")
            print(smpl_verts.shape, human_joints_3D_est.shape)

            print(np.min(smpl_verts[:, 0]), np.max(smpl_verts[:, 0]))
            print(np.min(smpl_verts[:, 1]), np.max(smpl_verts[:, 1]))
            print(np.min(smpl_verts[:, 2]), np.max(smpl_verts[:, 2]))

            print(np.min(human_joints_3D_est[:, 0]), np.max(human_joints_3D_est[:, 0]))
            print(np.min(human_joints_3D_est[:, 1]), np.max(human_joints_3D_est[:, 1]))
            print(np.min(human_joints_3D_est[:, 2]), np.max(human_joints_3D_est[:, 2]))
            input("Press Enter to continue...")









