
#!/usr/bin/env python

#Bodies at Rest: Code to visualize real dataset.
#(c) Henry M. Clever
#Major updates made for CVPR release: December 10, 2019

#python evaluate_depthreal_slp.py --viz '2D' --depthnoise  --p_idx 1 --loss_root --pcsum --small --cnn 'resnetunet' --blanket



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
import lib_pyrender_depth as libPyRender
import lib_pyrender_depth_savefig as libPyRenderSave
import lib_pyrender_depth_plp as libPyRenderPLP


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
    print("importing load model3")
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

        if opt.viz == '3D':
            pyrender3D = True
        else:
            pyrender3D = False

        if opt.savefig == True:
            self.pyRender = libPyRenderSave.pyRenderMesh(render = pyrender3D)
        elif opt.perlimbpeak == True:
            self.pyRender = libPyRenderPLP.pyRenderMesh(render = pyrender3D)
        else:
            self.pyRender = libPyRender.pyRenderMesh(render = pyrender3D)
            #self.pyRender = libPyRender.pyRenderMesh(render = False)

        model_path_m = FILEPATH+'smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
        model_path_f = FILEPATH+'smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'

        self.m_male = load_model(model_path_m)
        self.m_female = load_model(model_path_f)


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

        self.wt_gt_est_ht_gt_est_list = []

        self.pmat_gt_est_list = []
        self.smpl_verts_est_list = []
        self.smpl_verts_gt_list = []


    def evaluate_data(self, testing_database_file_f, testing_database_file_m, model, model2):


        #if len(testing_database_file_m) > 0 and len(testing_database_file_f) == 0:
        #    model_path = '../../../git/SMPL_python_v.1.0.0/smpl/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl'
        #elif len(testing_database_file_f) > 0 and len(testing_database_file_m) == 0:
        model_path = '../../../git/SMPL_python_v.1.0.0/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
        #else:
        #    sys.exit("can only test f or m at one time, not both.")
        self.m = load_model(model_path)



        if self.opt.no_blanket == True:
            if self.opt.savefig == True:
                color_load = 'uncover'
                test_dat_f_slp_lg = SLPPrepLib().load_slp_files_to_database(testing_database_file_f, dana_lab_path, PM = 'uncover', depth = 'uncover', color = color_load, mass_ht_list=all_f_subj_mass_ht_list, image_zoom = 4.585/2, markers_gt_type = '3D', use_pc = True, pm_adjust_mm = self.CTRL_PNL['x_y_offset_synth'])
                test_dat_m_slp_lg = SLPPrepLib().load_slp_files_to_database(testing_database_file_m, dana_lab_path, PM = 'uncover', depth = 'uncover', color = color_load, mass_ht_list=all_m_subj_mass_ht_list, image_zoom = 4.585/2, markers_gt_type = '3D', use_pc = True, pm_adjust_mm = self.CTRL_PNL['x_y_offset_synth'])
            else: pass
            test_dat_f_slp = SLPPrepLib().load_slp_files_to_database(testing_database_file_f, dana_lab_path, PM = 'uncover', depth = 'uncover', color = None, mass_ht_list=all_f_subj_mass_ht_list, markers_gt_type = '3D', use_pc = True, pm_adjust_mm = self.CTRL_PNL['x_y_offset_synth'])
            test_dat_m_slp = SLPPrepLib().load_slp_files_to_database(testing_database_file_m, dana_lab_path, PM = 'uncover', depth = 'uncover', color = None, mass_ht_list=all_m_subj_mass_ht_list, markers_gt_type = '3D', use_pc = True, pm_adjust_mm = self.CTRL_PNL['x_y_offset_synth'])


        else:
            if self.opt.savefig == True:
                test_dat_f_slp_lg = SLPPrepLib().load_slp_files_to_database(testing_database_file_f, dana_lab_path, PM = opt.ctype, depth = opt.ctype, color = opt.ctype, mass_ht_list=all_f_subj_mass_ht_list, image_zoom = 4.585/2, markers_gt_type = '3D', use_pc = True, pm_adjust_mm = self.CTRL_PNL['x_y_offset_synth'])
                test_dat_m_slp_lg = SLPPrepLib().load_slp_files_to_database(testing_database_file_m, dana_lab_path, PM = opt.ctype, depth = opt.ctype, color = opt.ctype, mass_ht_list=all_m_subj_mass_ht_list, image_zoom = 4.585/2, markers_gt_type = '3D', use_pc = True, pm_adjust_mm = self.CTRL_PNL['x_y_offset_synth'])

                test_dat_f_slp_nb_lg = SLPPrepLib().load_slp_files_to_database(testing_database_file_f, dana_lab_path, PM = 'uncover', depth = 'uncover', color = 'uncover', mass_ht_list=all_f_subj_mass_ht_list, image_zoom = 4.585/2, markers_gt_type = '3D', use_pc = True, pm_adjust_mm = self.CTRL_PNL['x_y_offset_synth'])
                test_dat_m_slp_nb_lg = SLPPrepLib().load_slp_files_to_database(testing_database_file_m, dana_lab_path, PM = 'uncover', depth = 'uncover', color = 'uncover', mass_ht_list=all_m_subj_mass_ht_list, image_zoom = 4.585/2, markers_gt_type = '3D', use_pc = True, pm_adjust_mm = self.CTRL_PNL['x_y_offset_synth'])

            else: pass
            test_dat_f_slp = SLPPrepLib().load_slp_files_to_database(testing_database_file_f, dana_lab_path, PM = opt.ctype, depth = opt.ctype, color = None, mass_ht_list=all_f_subj_mass_ht_list, markers_gt_type = '3D', use_pc = True, pm_adjust_mm = self.CTRL_PNL['x_y_offset_synth'])
            test_dat_m_slp = SLPPrepLib().load_slp_files_to_database(testing_database_file_m, dana_lab_path, PM = opt.ctype, depth = opt.ctype, color = None, mass_ht_list=all_m_subj_mass_ht_list, markers_gt_type = '3D', use_pc = True, pm_adjust_mm = self.CTRL_PNL['x_y_offset_synth'])

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


        if self.opt.small == False:
            test_x = np.zeros((len_f + len_m, x_map_ct, 64, 27)).astype(np.float32)
        else:
            test_x = np.zeros((int(len_f + len_m)/4, x_map_ct, 64, 27)).astype(np.float32)

        #allocate pressure images
        test_x = TensorPrepLib().prep_images(test_x, test_dat_f_slp, test_dat_m_slp, None, None, filter_sigma = 0.5, start_map_idx = pmat_gt_idx)


        self.mesh_reconstruction_maps = None
        self.reconstruction_maps_input_est = None

        if self.opt.no_blanket == True or opt.ctype == 'uncover':
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
        #test_y_flat[:, 160] = test_y_flat[:, 160] * test_y_flat[:, 157] / 0.06878933937454557 * 62.5 + test_y_flat[:, 160] * test_y_flat[:, 158] / 0.0828308574658067 * 78.4 #GROUND TRUTH M SYNTH IS OFF
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

        if opt.no_blanket == True:
            dat_pc = test_dat_m_slp['pc'] + test_dat_f_slp['pc']
        else:
            dat_pc = test_dat_m_slp_nb['pc'] + test_dat_f_slp_nb['pc']


        if self.opt.savefig == True and opt.no_blanket == True:
            dat_color_render = test_dat_m_slp_lg['overhead_colorcam_noblanket'] + test_dat_f_slp_lg['overhead_colorcam_noblanket']
            dat_color_occl_render = test_dat_m_slp_lg['overhead_colorcam_noblanket'] + test_dat_f_slp_lg['overhead_colorcam_noblanket']
            dat_depth_render = test_dat_m_slp_lg['overhead_depthcam_noblanket'] + test_dat_f_slp_lg['overhead_depthcam_noblanket']
        elif self.opt.savefig == True and opt.no_blanket == False:
            dat_color_render = test_dat_m_slp_nb_lg['overhead_colorcam_noblanket'] + test_dat_f_slp_nb_lg['overhead_colorcam_noblanket']
            if opt.ctype == "uncover":
                dat_depth_render = test_dat_m_slp_nb_lg['overhead_depthcam_noblanket'] + test_dat_f_slp_nb_lg['overhead_depthcam_noblanket']
                dat_color_occl_render = test_dat_m_slp_nb_lg['overhead_colorcam_noblanket'] + test_dat_f_slp_nb_lg['overhead_colorcam_noblanket']
            else:
                dat_depth_render = test_dat_m_slp_lg['overhead_depthcam'] + test_dat_f_slp_lg['overhead_depthcam']
                dat_color_occl_render = test_dat_m_slp_lg['overhead_colorcam'] + test_dat_f_slp_lg['overhead_colorcam']
        else:
            dat_color_render = None
            dat_color_occl_render = None
            dat_depth_render = None



        #for im_num in range(29, 100):
        for im_num in range(self.opt.pose_num, np.shape(test_x)[0]):#self.color_all.shape[0]):


            print("TESTING IM NUM ", im_num)

            #PRESSURE
            self.pressure = test_x[im_num, 0, :, :]

            self.depth = test_x[im_num:im_num+1, 1:, :, :]

            pc_autofil_red = dat_pc[im_num]

            if self.opt.savefig == True:
                self.color_im_render = dat_color_render[im_num]
                self.color_im_occl_render = dat_color_occl_render[im_num]
                self.depth_im_render = dat_depth_render[im_num]

                print(np.shape(dat_color_render))
                print(np.shape(dat_depth_render))


            self.estimate_pose(self.pressure, pc_autofil_red, model, model2, im_num)


            self.point_cloud_array = None
            #sleep(100)

        if self.opt.viz == '3D' and self.opt.savefig == False:
            if self.opt.pimgerr == True:
                if self.opt.pmr == False:
                    dir = FILEPATH + 'data_BP/results/results_pressure/' + NETWORK_1 + '_' + opt.ctype
                else:
                    dir = FILEPATH + 'data_BP/results/results_pressure/' + NETWORK_2 + '_' + opt.ctype
            else:
                if self.opt.pmr == True or self.CTRL_PNL['depth_out_unet'] == True:
                    dir = FILEPATH + 'data_BP/results/3D_quantitative_PAMI_v2vP/'+NETWORK_2+'_'+opt.ctype
                else:
                    dir = FILEPATH + 'data_BP/results/3D_quantitative_PAMI_v2vP/'+NETWORK_1+'_'+opt.ctype

            if not os.path.exists(dir):
                os.mkdir(dir)

            participant_num = '%05d' % (opt.p_idx)

            pkl.dump(self.RESULTS_DICT, open(dir+'/results_slp_3D_'+participant_num+'.p', 'wb'))
        else:
            self.save_pck_results(self.joint_error_list, self.hd_th_dist_list, self.wt_gt_est_ht_gt_est_list, self.pmat_gt_est_list)



    def compute_htwt_error(self, INPUT_DICT, OUTPUT_DICT):
        weight_kg_gt = float(INPUT_DICT['batch_weight_kg'].detach().cpu().numpy())
        weight_kg_est = float(OUTPUT_DICT['batch_weight_kg_est'].detach().cpu().numpy())
        height_m_gt = float(INPUT_DICT['batch_height'].detach().cpu().numpy())
        height_m_est = float(OUTPUT_DICT['batch_height_est'].detach().cpu().numpy())

        print('ground truth weight kg:', weight_kg_gt)
        print('estimated weight kg:', weight_kg_est)
        print('ground truth height m:', height_m_gt)
        print('estimated height m:', height_m_est)

        betas_gt = np.array(INPUT_DICT['batch_betas'].detach().cpu().numpy()[0])
        betas_est = np.array(OUTPUT_DICT['batch_betas_est_post_clip'].detach().cpu().numpy()[0])

        #print(betas_gt, "BETAS GT")
        #print(betas_est, "BETAS GT")

        gender = 'f'
        model_path = '/home/henry/git/SMPL_python_v.1.0.0/smpl/models/basicModel_'+gender+'_lbs_10_207_0_v1.0.0.pkl'
        mf = load_model(model_path)

        human_mesh_face_all = [np.array(mf.f)]

        for shape_param in range(10):
            mf.betas[shape_param] = float(betas_gt[shape_param])

        human_mesh_vtx_all_mod = [np.array(mf.r)]
        import trimesh
        smpl_trimesh_mod = trimesh.Trimesh(vertices=human_mesh_vtx_all_mod[0], faces=human_mesh_face_all[0])
        vol_mod = smpl_trimesh_mod.volume
        print(vol_mod, 'vol mod gt')



        for shape_param in range(10):
            mf.betas[shape_param] = float(betas_est[shape_param])

        human_mesh_vtx_all_mod = [np.array(mf.r)]

        smpl_trimesh_mod = trimesh.Trimesh(vertices=human_mesh_vtx_all_mod[0], faces=human_mesh_face_all[0])
        vol_mod = smpl_trimesh_mod.volume
        print(vol_mod, 'vol mod est')

        return [weight_kg_gt, weight_kg_est, height_m_gt, height_m_est]



    def compute_3D_error(self, target_3D, score_3D):
        score_3D = score_3D[:24, :]
        xy_diff = target_3D - score_3D
        eulc_err = np.linalg.norm(xy_diff, axis = 1)
        print(eulc_err, 'eulc err')

        return eulc_err, target_3D, score_3D


    def save_pck_results(self, all_error, hd_th_dist_list, wt_gt_est_ht_gt_est_list, pmat_gt_est_list):
        all_error = np.array(all_error)
        hd_th_dist_list = np.array(hd_th_dist_list).flatten()


        all_error = all_error.flatten()

        pck_dict = {}

        #if np.shape(all_error_head)[0] == save_idx: #then plot stuff
        pck_dict['total_allerror'] = all_error
        pck_dict['hd_th_dist'] = hd_th_dist_list
        pck_dict['wt_gt_est_ht_gt_est_list'] = wt_gt_est_ht_gt_est_list
        pck_dict['pmat_gt_est_list'] = pmat_gt_est_list
        pck_dict['smpl_verts_est'] = self.smpl_verts_est_list
        pck_dict['smpl_verts_gt'] = self.smpl_verts_gt_list

        #if opt.blanket == True:
        #pkl.dump(pck_dict, open('/home/henry/git/sim_camera_resting_scene/data_BR/'+NETWORK_2+'_pck_71to80_slp_'+opt.ctype+'.p', 'wb'))\
        if opt.mod == 1:
            pkl.dump(pck_dict, open(FILEPATH + 'data_BP/results/quant_'+NETWORK_1+'_81to102_slp_'+opt.ctype+'.p', 'wb'))
        elif opt.mod == 2:
            pkl.dump(pck_dict, open(FILEPATH + 'data_BP/results/quant_'+NETWORK_2+'_81to102_slp_'+opt.ctype+'.p', 'wb'))
        #else:
        #    pkl.dump(pck_dict, open('/home/henry/git/sim_camera_resting_scene/data_BR/'+NETWORK_1+'_pck_last12_slp_uncover.p', 'wb'))


    def get_SMPL_verts(self, batch1, INPUT_DICT):
        betas_gt = torch.mean(batch1[:, 72:82], dim=0).numpy()
        angles_gt = torch.mean(batch1[:, 82:154], dim=0).numpy()
        root_shift_est_gt = torch.mean(batch1[:, 154:157], dim=0).numpy()
        root_shift_est_gt[1] *= -1
        root_shift_est_gt[2] *= -1

        R_root = kinematics_lib_br.matrix_from_dir_cos_angles(angles_gt[0:3])
        flip_root_euler = np.pi
        flip_root_R = kinematics_lib_br.eulerAnglesToRotationMatrix([flip_root_euler, 0.0, 0.0])
        angles_gt[0:3] = kinematics_lib_br.dir_cos_angles_from_matrix(np.matmul(flip_root_R, R_root))


        if INPUT_DICT['batch_gender'].data.numpy()[0][0] == 1:
            for beta in range(betas_gt.shape[0]):
                self.m_female.betas[beta] = betas_gt[beta]
            for angle in range(angles_gt.shape[0]):
                self.m_female.pose[angle] = angles_gt[angle]

            smpl_verts_gt = np.array(self.m_female.r)
            for s in range(root_shift_est_gt.shape[0]):
                smpl_verts_gt[:, s] += (root_shift_est_gt[s] - float(self.m_female.J_transformed[0, s]))

            smpl_verts_gt = np.concatenate((-(smpl_verts_gt[:, 1:2]),# - 0.286 + 0.0143),
                                            smpl_verts_gt[:, 0:1],# - 0.286 + 0.0143,
                                            smpl_verts_gt[:, 2:3]), axis=1)

            print('FEMALE')
        elif INPUT_DICT['batch_gender'].data.numpy()[0][0] == 0:
            for beta in range(betas_gt.shape[0]):
                self.m_male.betas[beta] = betas_gt[beta]
            for angle in range(angles_gt.shape[0]):
                self.m_male.pose[angle] = angles_gt[angle]

            smpl_verts_gt = np.array(self.m_male.r)
            for s in range(root_shift_est_gt.shape[0]):
                smpl_verts_gt[:, s] += (root_shift_est_gt[s] - float(self.m_male.J_transformed[0, s]))

            print(smpl_verts_gt)
            smpl_verts_gt = np.concatenate((-(smpl_verts_gt[:, 1:2]),# - 0.286 + 0.0143),
                                            smpl_verts_gt[:, 0:1],# - 0.286 + 0.0143,
                                            smpl_verts_gt[:, 2:3]), axis=1)

            print('MALE')
        return smpl_verts_gt


    def estimate_pose(self, pmat, pc_autofil_red, model, model2, im_num):

        bedangle = 0

        mat_size = (64, 27)

        pmat_stack = PreprocessingLib().preprocessing_create_pressure_only_stack([pmat], mat_size, self.CTRL_PNL)[0]
        pmat_stack = np.expand_dims(np.array(pmat_stack), 0)
        pmat_stack = torch.Tensor(pmat_stack)


        batch1 = self.test_y_tensor[im_num:im_num+1, :]


        batch = []
        depth_stack = torch.Tensor(self.depth)


        batch.append(torch.cat((pmat_stack, depth_stack), dim = 1))
        batch.append(batch1)


        batch0 = batch[0].clone()

        NUMOFOUTPUTDIMS = 3
        NUMOFOUTPUTNODES_TRAIN = 24
        self.output_size_train = (NUMOFOUTPUTNODES_TRAIN, NUMOFOUTPUTDIMS)

        self.CTRL_PNL['recon_map_input_est'] = False
        self.CTRL_PNL['adjust_ang_from_est'] = False
        self.CTRL_PNL['mod'] = 1
        if opt.pmr == True and opt.mod == 2:
            self.CTRL_PNL['compute_forward_maps'] = True
        else:
            self.CTRL_PNL['compute_forward_maps'] = False

        scores, INPUT_DICT, OUTPUT_DICT = UnpackDepthBatchLib().unpack_batch(batch, is_training=False, model=model, model_smpl_pmr = self.model_smpl_pmr,
                                                                             model_CAL = model_CAL, model_betanet = model_betanet,
                                                                             CTRL_PNL = self.CTRL_PNL)
        INPUT_DICT_mod1 = INPUT_DICT.copy()
        if opt.pmr == True:
            self.CTRL_PNL['compute_forward_maps'] = False
        if model2 is not None:
            self.CTRL_PNL['mod'] = 2

            print("Using model 2")

            if opt.pmr == True:
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
            scores, INPUT_DICT, OUTPUT_DICT = UnpackDepthBatchLib().unpack_batch(batch_cor, is_training=False, model=model2, model_smpl_pmr = self.model_smpl_pmr,
                                                                                 model_CAL = model_CAL, model_betanet = model_betanet,
                                                                                 CTRL_PNL = self.CTRL_PNL)

        self.CTRL_PNL['first_pass'] = False



        #OUTPUT_DICT['batch_betas_est'].cpu()
        wt_gt_est_ht_gt_est = self.compute_htwt_error(INPUT_DICT, OUTPUT_DICT)
        self.wt_gt_est_ht_gt_est_list.append(wt_gt_est_ht_gt_est)


        #if self.CTRL_PNL['slp'] != 'real':
        # print betas_est, root_shift_est, angles_est
        if self.CTRL_PNL['dropout'] == True:
            # print OUTPUT_DICT['verts'].shape
            smpl_verts = np.mean(OUTPUT_DICT['verts'], axis=0)
            dropout_variance = np.std(OUTPUT_DICT['verts'], axis=0)
            dropout_variance = np.linalg.norm(dropout_variance, axis=1)
        else:
            smpl_verts = OUTPUT_DICT['verts'][0, :, :]
            dropout_variance = None

        smpl_verts = np.concatenate((smpl_verts[:, 1:2], smpl_verts[:, 0:1],  -smpl_verts[:, 2:3]),  axis=1)
        smpl_faces = np.array(self.m.f)

        camera_point = [1.09898028, 0.46441343, -CAM_BED_DIST]

        SHOW_SMPL_EST = True
        if SHOW_SMPL_EST == False:
            smpl_verts *= 0.001


        smpl_verts_gt = self.get_SMPL_verts(batch1, INPUT_DICT)

        self.smpl_verts_est_list.append(smpl_verts)
        self.smpl_verts_gt_list.append(smpl_verts_gt)

        if self.opt.viz == '2D':
            viz_type = "2D"
        elif self.opt.viz == '3D':
            viz_type = "3D"
        else:
            viz_type = None


        self.tar_sample = INPUT_DICT['batch_targets']

        self.tar_sample = self.tar_sample.view(-1, 24, 3)
        #self.tar_sample[:, :, 1] -= 100.

        self.tar_sample = self.tar_sample[0, :].squeeze() / 1000
        sc_sample = OUTPUT_DICT['batch_targets_est'].clone()
        sc_sample = sc_sample[0, :].squeeze() / 1000
        sc_sample = torch.cat((sc_sample, torch.Tensor([0.0, 0.0, 0.0])), dim=0)
        sc_sample = sc_sample.view([self.output_size_train[0] + 1, self.output_size_train[1]])




        #print sc_sample, 'SCORE SAMPLE'
        th_hd = [self.tar_sample.cpu().numpy()[12, 0:2], self.tar_sample.cpu().numpy()[15, 0:2]]

        all_error, target3D, score3D = self.compute_3D_error(self.tar_sample.cpu().numpy(), sc_sample.cpu().numpy())

        self.joint_error_list.append(all_error)
        self.target_list.append(target3D)
        self.score_list.append(score3D)


        print(np.mean(self.joint_error_list[-1]), 'average error for pose', im_num) #0.13761432 69 #0.14558357 70
        shape_err_list = np.shape(self.joint_error_list)
        if im_num == 44:
            print(np.shape(np.array(self.joint_error_list)[-45:, :]), np.mean(np.array(self.joint_error_list)[-45:, :]), 'mean for participant')


        im_display_idx = 0
        self.VIZ_DICT = {}
        self.VIZ_DICT = VisualizationLib().get_depthnet_viz_maps(im_display_idx, INPUT_DICT, OUTPUT_DICT, self.VIZ_DICT, self.CTRL_PNL)
        self.VIZ_DICT = VisualizationLib().get_fcn_recon_viz_maps(im_display_idx, INPUT_DICT, OUTPUT_DICT,  self.VIZ_DICT, self.CTRL_PNL)
        self.CTRL_PNL['recon_map_input_est'] = False

        if self.opt.pmr == True or self.CTRL_PNL['depth_out_unet'] == True:
            pmatV = self.VIZ_DICT['p_img'].data.numpy()
            pmatV_est = self.VIZ_DICT['p_img_est'].data.numpy()
            #pmatV_est = OUTPUT_DICT['batch_mdm_est'].clone().data.numpy() * -1

            if self.opt.pimgerr == True:
                self.RESULTS_DICT['pmat'].append(pmatV)
                self.RESULTS_DICT['pmat_est'].append(pmatV_est)

            weight_kg_gt = float(INPUT_DICT['batch_weight_kg'].detach().cpu().numpy())
            weight_kg_est = float(OUTPUT_DICT['batch_weight_kg_est'].detach().cpu().numpy())

            #pmatV = pmatV#/np.sum(pmatV)
            #pmatV_est = pmatV_est#/np.sum(pmatV_est)

            #print(np.max(pmatV), np.sum(pmatV), weight_kg_gt, 'max pmatV, sum pmatV, and weight')
            print(np.max(pmatV_est), np.sum(pmatV_est), weight_kg_est, 'max pmatVest, sum pmatVest, and est weight')

            #pmatV = pmatV*((weight_kg_gt * 9.81) / (np.sum(pmatV)* 0.0264*0.0286)) * (1/133.322) #we already do this when loading the GT stuff
            #pmatV_est = pmatV_est*((weight_kg_est * 9.81) / (np.sum(pmatV_est)* 0.0264*0.0286)) * (1/133.322)
            #print('hereeee')

            print(np.max(pmatV), np.sum(pmatV), weight_kg_gt, 'max pmatV, sum pmatV, and weight, mmHg normalized')
            print(np.max(pmatV_est), np.sum(pmatV_est), weight_kg_est, 'max pmatVest, sum pmatVest, and est weight, mmHg normalized')

            print(np.mean(np.square(np.abs(pmatV - pmatV_est))), np.mean(np.abs(pmatV - pmatV_est)))
            #ct = 0
            #for i in [0, 5, 10, 15, 20, 25, 30, 35, 40]:
            #    pmatV_est[ct:ct+4, 0:4] = i
            #    ct += 4
            self.pmat_gt_est_list.append([pmatV, pmatV_est])
            self.VIZ_DICT['p_img_est'] =pmatV_est*1.


            print("Pimg error, mmHg squared", np.mean(np.square(np.array(pmatV)-np.array(pmatV_est)) ) )
            print("Pimg error, kPa squared", 133.32*133.32*(1/1000000)*np.mean(np.square(np.array(pmatV)-np.array(pmatV_est)) ) )




        if viz_type == "2D":
            VisualizationLib().visualize_depth_net(VIZ_DICT = self.VIZ_DICT,
                                                      targets_raw = self.tar_sample.cpu(), scores_net1 = sc_sample.cpu(),
                                                      block=True, max_depth = 2200, is_testing = True)


        elif viz_type == "3D":
            depth = INPUT_DICT_mod1['batch_images'][0, int(INPUT_DICT_mod1['batch_images'].size()[1]) - 2, :].clone().numpy().reshape(128, 54)
            self.depth_im_render = zoom(depth, 3.435, order = 1)


            participant_savename = str('%05d' % (self.opt.p_idx))
            #if self.opt.blanket == True:
            participant_savename += ('_'+opt.ctype)



            if self.opt.savefig == False:
                self.RESULTS_DICT = self.pyRender.render_mesh_pc_bed_pyrender_everything(smpl_verts, smpl_faces, camera_point,
                                                                                         self.RESULTS_DICT, smpl_verts_gt = smpl_verts_gt,
                                                                                         pc=pc_autofil_red, pmat=pmatV, pmat_est = pmatV_est,
                                                                                         targets=self.tar_sample.view(72).cpu(),
                                                                                         scores=sc_sample.cpu())


                input("Press Enter to continue...")

            else:
                self.RESULTS_DICT = self.pyRender.render_mesh_pc_bed_pyrender_everything(smpl_verts, smpl_faces, self.RESULTS_DICT,
                                                                                        #smpl_verts_gt = smpl_verts_gt,
                                                                                        pmat=pmatV, pmat_est = pmatV_est,
                                                                                        color_im_occl=self.color_im_occl_render,
                                                                                        color_im=self.color_im_render,
                                                                                        depth_im=self.depth_im_render,
                                                                                        current_pose_type_ct=str(im_num),
                                                                                        participant=participant_savename)

            self.point_cloud_array = None






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

    p.add_option('--savefig', action='store_true', dest='savefig', default=False,
                 help='Use blankets.')

    p.add_option('--perlimbpeak', action='store_true', dest='perlimbpeak', default=False,
                 help='Evaluate peak pressure per limb with threshold.')

    opt, args = p.parse_args()


    if opt.savefig == True:
        opt.viz = '3D'
    if opt.ctype not in ['uncover', 'cover1', 'cover2']:
        print("need to specify valid ctype of specify `uncover` or `cover1` or `cover2`.")
        sys.exit()


    if opt.hd == True:
        dana_lab_path = '/media/henry/multimodal_data_2/data/SLP/danaLab/'
    else:
        dana_lab_path = FILEPATH + 'data_BP/SLP/danaLab/'


    if opt.p_idx == 0:
        test_crit = 'last12'
    else:
        test_crit = None

    if test_crit == "all_subjects":
        all_subj_str_list = []
        for i in range(1, 103):
            all_subj_str_list.append('%05d' % (i))
    elif test_crit == "last12":
        all_subj_str_list = []
        #for i in range(69, 81):
        #for i in range(71, 81):
        for i in range(81, 103):
        #for i in range(1, 91):
            all_subj_str_list.append('%05d' % (i))
    else:
        all_subj_str_list = ['%05d' % (opt.p_idx)]


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

    print(testing_database_file_m)
    print(testing_database_file_f)

    if opt.no_blanket == True:
        opt.ctype = 'uncover'


    V3D = Viz3DPose(opt)


    if opt.losstype == 'anglesDC':
        NETWORK_1 = "anglesDC_"
        NETWORK_2 = "anglesDC_"
    elif opt.losstype == 'direct':
        NETWORK_1 = "direct_"
        NETWORK_2 = "direct_"

    if opt.small == True:
        NETWORK_1 += "46000ct_"
        NETWORK_2 += "46000ct_"
    elif opt.slp == "real":
        #NETWORK_1 += "9315ct_"
        #NETWORK_2 += "9315ct_"
        NETWORK_1 += "10665ct_"
        NETWORK_2 += "10665ct_"
    elif opt.slp == "synth":
        #NETWORK_1 += "85114ct_"
        #NETWORK_2 += "85114ct_"
        NETWORK_1 += "97495ct_"
        NETWORK_2 += "97495ct_"
    elif opt.slp == "mixed":
        NETWORK_1 += "183114ct_"
        NETWORK_2 += "183114ct_"
    elif opt.slp == "mixedreal":
        #NETWORK_1 += "94429ct_"
        #NETWORK_2 += "94429ct_"
        NETWORK_1 += "108160ct_"
        NETWORK_2 += "108160ct_"
    else:
        NETWORK_1 += "184000ct_"
        NETWORK_2 += "184000ct_"


    if opt.go200 == True:
        NETWORK_1 += "128b_x1pm"
        NETWORK_2 += "128b_x1pm"
    elif opt.pmr == True:
        if opt.mod == 1:
            NETWORK_1 += "128b_x1pm_0.5rtojtdpth"

        elif opt.mod == 2:
            NETWORK_1 += "128b_x1pm"#_0.5rtojtdpth"
            NETWORK_2 += "128b_x1pm_0.5rtojtdpth_depthestin_angleadj"
    elif opt.X_is == 'B' and opt.mod == 2:
        NETWORK_1 += "128b_x1pm"
        NETWORK_2 += "128b_x1pm_depthestin_angleadj"
    else:
        NETWORK_1 += "128b_x1pm"
        NETWORK_2 += "128b_x1pm_angleadj"

    if opt.no_reg_angles == False:
        NETWORK_1 += '_rgangs'
        NETWORK_2 += '_rgangs'

    if opt.no_loss_betas == False:
        NETWORK_1 += '_lb'
        NETWORK_2 += '_lb'

    if opt.v2v == True:
        if opt.mod == 1:
            NETWORK_1 += '_lv2v'
        NETWORK_2 += '_lv2v'

    if opt.noloss_htwt == True:
        NETWORK_1 += '_nlhw'
        NETWORK_2 += '_nlhw'

    if opt.no_blanket == False or opt.slp == 'real':
        NETWORK_1 += "_slpb"
        NETWORK_2 += "_slpb"
    else:
        NETWORK_1 += "_slpnb"
        NETWORK_2 += "_slpnb"

    if opt.htwt == True:
        NETWORK_1 += "_htwt"
        NETWORK_2 += "_htwt"

    if opt.no_depthnoise == False:
        NETWORK_1 += "_dpns"
        NETWORK_2 += "_dpns"

    if opt.slpnoise == True:
        NETWORK_1 += "_slpns"
        NETWORK_2 += "_slpns"

    if opt.no_loss_root == False:
        NETWORK_1 += "_rt"
        NETWORK_2 += "_rt"

    if opt.half_shape_wt == True:
        NETWORK_1 += "_hsw"
        NETWORK_2 += "_hsw"


    if opt.X_is == 'B':
        NETWORK_1 += "_dou"
        NETWORK_2 += "_dou"
    elif opt.X_is == 'W':
        pass
    else:
        print('you need to select a valid X_is. choose "W" for white box net or "B" for black box net.')
        sys.exit()



    NETWORK_1 += "_100e"
    NETWORK_2 += "_40e"


    if opt.X_is == 'B':
        if opt.small == True:
            filename1 = FILEPATH + "data_BP/convnets/resnetunet34_1_" + NETWORK_1 + "_0.0001lr.pt"
            filename2 = FILEPATH + "data_BP/convnets/resnetunet34_2_" + NETWORK_2 + "_0.0001lr.pt"
        else:
            filename1 = FILEPATH + "data_BP/convnets/resnetunet34_1_" + NETWORK_1 + "_0.0001lr.pt"
            filename2 = FILEPATH + "data_BP/convnets/resnetunet34_2_" + NETWORK_2 + "_0.0001lr.pt"

        if opt.mod == 1:
            filename2 = None
    else:
        if opt.small == True:
            filename1 = FILEPATH + "data_BP/convnets/resnet34_1_" + NETWORK_1 + "_0.0001lr.pt"#
            filename2 = FILEPATH + "data_BP/convnets/resnet34_2_" + NETWORK_2 + "_0.0001lr.pt"#_noiseyCorr.pt"
        elif opt.slp == "real":
            filename1 = FILEPATH + "data_BP/convnets/resnet34_1_" + NETWORK_1 + "_0.0001lr.pt"
            filename2 = FILEPATH + "data_BP/convnets/resnet34_2_" + NETWORK_2 + "_0.0001lr.pt"#_noiseyCorr.pt"
        else:
            filename1 = FILEPATH + "data_BP/convnets/resnet34_1_" + NETWORK_1 + "_0.0001lr.pt"
            filename2 = FILEPATH + "data_BP/convnets/resnet34_2_" + NETWORK_2 + "_0.0001lr.pt"#_noiseyCorr.pt"

        if opt.pmr == False:
            filename2 = None



    if GPU == True:
        for i in range(0, 8):
            try:
                model = torch.load(filename1, map_location={'cuda:' + str(i): 'cuda:0'})
                model = model.cuda().eval()
                print ("Network 1 loaded.")
                break
            except:
                pass
        if filename2 is not None and opt.mod == 2:
            for i in range(0, 8):
                try:
                    model2 = torch.load(filename2, map_location={'cuda:' + str(i): 'cuda:0'})
                    model2 = model2.cuda().eval()
                    print ("Network 2 loaded.")
                    break
                except:
                    pass
        else:
            model2 = None
    else:
        if sys.version_info.major < 3:
            print(sys.version_info.major)
            model = torch.load(filename1, map_location='cpu')
        else:
            print(sys.version_info.major)
            pkl.load = partial(pkl.load, encoding='latin1')
            pkl.Unpickler = partial(pkl.Unpickler, encoding='latin1')
            model = torch.load(filename1, map_location='cpu', pickle_module=pkl)
            print("got here")

        model = model.eval()
        print ("Network 1 loaded.")
        if filename2 is not None and opt.mod == 2:
            if sys.version_info.major < 3:
                model2 = torch.load(filename2, map_location='cpu')
            else:
                model2 = torch.load(filename2, map_location='cpu', pickle_module=pkl)

            model2 = model2.eval()
            print ("Network 2 loaded.")
        else:
            model2 = None


    if opt.X_is == 'W':
        if opt.slp == 'synth':
            model_CAL = torch.load(FILEPATH + 'data_BP/convnets/CAL_97495ct_128b_500e_0.0001lr.pt',map_location='cpu')
        else:
            model_CAL = torch.load(FILEPATH + 'data_BP/convnets/CAL_10665ct_128b_500e_0.0001lr.pt',map_location='cpu')
    else:
        model_CAL = None
    model_betanet = torch.load(FILEPATH + 'data_BP/convnets/betanet_108160ct_128b_volfrac_500e_0.0001lr.pt', map_location='cpu')
    #python evaluate_depthreal_slp.py  --depthnoise  --p_idx 91 --loss_root --rgangs --small --pmr  --viz '3D' --blanket
    F_eval = V3D.evaluate_data(testing_database_file_f, testing_database_file_m, model, model2)



