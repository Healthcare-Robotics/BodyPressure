
#!/usr/bin/env python

#Bodies at Rest: Code to visualize real dataset.
#(c) Henry M. Clever
#Major updates made for CVPR release: December 10, 2019

#python evaluate_depthreal_slp.py --viz '2D' --depthnoise  --p_idx 1 --loss_root --pcsum --small --cnn 'resnetunet' --blanket


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

import sys
sys.path.insert(0, '../lib_py')
sys.path.insert(-1,'/home/henry/git/smpl/smpl_webuser3')
sys.path.insert(0, '/home/henry/git/sim_camera_resting_scene/DPNet/')
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
print(sys.path)

import fixedwt_smpl_pmr_net as fixedwt_smpl_pmr
import lib_pyrender_depth as libPyRender
import lib_pyrender_depth_savefig as libPyRenderSave

import optparse

from visualization_lib_br import VisualizationLib
from preprocessing_lib_br import PreprocessingLib
from tensorprep_lib_br import TensorPrepLib
from unpack_depth_batch_lib_br import UnpackDepthBatchLib
import kinematics_lib_br as kinematics_lib_br


try:
    from smpl.smpl_webuser.serialization import load_model
except:
    print("importing load model3",  sys.path)
    from smpl.smpl_webuser3.serialization import load_model

import os


x = load_pickle('/home/henry/git/smplify_public/output_00001/0001.pkl')
print (x)


#volumetric pose gen libraries
from time import sleep
from scipy.stats import mode
import os.path as osp
import imutils
from scipy.ndimage.filters import gaussian_filter

from scipy.ndimage.interpolation import zoom

import matplotlib.cm as cm #use cm.jet(list)

DATASET_CREATE_TYPE = 1

from slp_prep_lib_br import SLPPrepLib
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
COVER_TYPE = 'cover2'
MARKERS_GT_TYPE = '3D'

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
        else:
            self.pyRender = libPyRender.pyRenderMesh(render = pyrender3D)
            #self.pyRender = libPyRender.pyRenderMesh(render = False)

        if MARKERS_GT_TYPE == '3D':
            model_path_m = '/home/henry/git/SMPL_python_v.1.0.0/smpl/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl'
            model_path_f = '/home/henry/git/SMPL_python_v.1.0.0/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'

            self.m_male = load_model(model_path_m)
            self.m_female = load_model(model_path_f)


        self.weight_lbs = 0
        self.height_in  = 0

        self.index_queue = []

        self.reset_pose = False

        self.pressure = None

        self.CTRL_PNL = {}
        self.CTRL_PNL['slp'] = opt.slp
        self.CTRL_PNL['nosmpl'] = opt.nosmpl
        self.CTRL_PNL['batch_size'] = 1
        self.CTRL_PNL['loss_vector_type'] = opt.losstype
        self.CTRL_PNL['CNN'] = opt.cnn
        self.CTRL_PNL['mod'] = opt.mod
        self.CTRL_PNL['verbose'] = False
        self.CTRL_PNL['num_epochs'] = 101
        self.CTRL_PNL['incl_inter'] = True
        self.CTRL_PNL['shuffle'] = False
        self.CTRL_PNL['incl_ht_wt_channels'] = opt.htwt
        self.CTRL_PNL['loss_root'] = opt.loss_root
        self.CTRL_PNL['pimg_cntct_sum'] = opt.pimg_cntct_sum
        self.CTRL_PNL['omit_pimg_cntct_sobel'] = opt.omit_pimg_cntct_sobel
        self.CTRL_PNL['incl_pmat_cntct_input'] = False
        self.CTRL_PNL['num_input_channels'] = 3
        self.CTRL_PNL['GPU'] = GPU
        self.CTRL_PNL['dtype'] = dtype
        self.CTRL_PNL['repeat_real_data_ct'] = 1
        self.CTRL_PNL['regr_angles'] = 1
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
        self.CTRL_PNL['depth_noise'] = opt.depthnoise
        self.CTRL_PNL['cal_noise_amt'] = 0.2
        self.CTRL_PNL['output_only_prev_est'] = False
        self.CTRL_PNL['double_network_size'] = False
        self.CTRL_PNL['first_pass'] = True
        self.CTRL_PNL['align_procr'] = False
        self.CTRL_PNL['depth_in'] = True
        self.CTRL_PNL['depth_out_unet'] = opt.depth_out_unet
        self.CTRL_PNL['onlyhuman_labels'] = False
        self.CTRL_PNL['slp_real'] = True
        self.CTRL_PNL['train_only_betanet'] = False
        self.CTRL_PNL['train_only_adjQP'] = False
        self.CTRL_PNL['compute_forward_maps'] = True
        self.CTRL_PNL['v2v'] = opt.v2v
        self.CTRL_PNL['x_y_offset_synth'] = [-7, -45]#[200, 200]#


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



    def evaluate_data(self, testing_database_file_f, testing_database_file_m, model, model2):


        #if len(testing_database_file_m) > 0 and len(testing_database_file_f) == 0:
        #    model_path = '../../../git/SMPL_python_v.1.0.0/smpl/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl'
        #elif len(testing_database_file_f) > 0 and len(testing_database_file_m) == 0:
        model_path = '../../../git/SMPL_python_v.1.0.0/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
        #else:
        #    sys.exit("can only test f or m at one time, not both.")
        self.m = load_model(model_path)



        if self.opt.blanket == False:
            if self.opt.savefig == True:
                color_load = 'uncover'
                test_dat_f_slp_lg = SLPPrepLib().load_slp_files_to_database(testing_database_file_f, dana_lab_path, PM = 'uncover', depth = 'uncover', color = color_load, mass_ht_list=all_f_subj_mass_ht_list, image_zoom = 4.585/2, markers_gt_type = MARKERS_GT_TYPE, use_pc = True)
                test_dat_m_slp_lg = SLPPrepLib().load_slp_files_to_database(testing_database_file_m, dana_lab_path, PM = 'uncover', depth = 'uncover', color = color_load, mass_ht_list=all_m_subj_mass_ht_list, image_zoom = 4.585/2, markers_gt_type = MARKERS_GT_TYPE, use_pc = True)
            else: pass
            test_dat_f_slp = SLPPrepLib().load_slp_files_to_database(testing_database_file_f, dana_lab_path, PM = 'uncover', depth = 'uncover', color = None, mass_ht_list=all_f_subj_mass_ht_list, markers_gt_type = MARKERS_GT_TYPE, use_pc = True)
            test_dat_m_slp = SLPPrepLib().load_slp_files_to_database(testing_database_file_m, dana_lab_path, PM = 'uncover', depth = 'uncover', color = None, mass_ht_list=all_m_subj_mass_ht_list, markers_gt_type = MARKERS_GT_TYPE, use_pc = True)


        else:
            if self.opt.savefig == True:
                test_dat_f_slp_lg = SLPPrepLib().load_slp_files_to_database(testing_database_file_f, dana_lab_path, PM = COVER_TYPE, depth = COVER_TYPE, color = COVER_TYPE, mass_ht_list=all_f_subj_mass_ht_list, image_zoom = 4.585/2, markers_gt_type = MARKERS_GT_TYPE, use_pc = True)
                test_dat_m_slp_lg = SLPPrepLib().load_slp_files_to_database(testing_database_file_m, dana_lab_path, PM = COVER_TYPE, depth = COVER_TYPE, color = COVER_TYPE, mass_ht_list=all_m_subj_mass_ht_list, image_zoom = 4.585/2, markers_gt_type = MARKERS_GT_TYPE, use_pc = True)

                test_dat_f_slp_nb_lg = SLPPrepLib().load_slp_files_to_database(testing_database_file_f, dana_lab_path, PM = 'uncover', depth = 'uncover', color = 'uncover', mass_ht_list=all_f_subj_mass_ht_list, image_zoom = 4.585/2, markers_gt_type = MARKERS_GT_TYPE, use_pc = True)
                test_dat_m_slp_nb_lg = SLPPrepLib().load_slp_files_to_database(testing_database_file_m, dana_lab_path, PM = 'uncover', depth = 'uncover', color = 'uncover', mass_ht_list=all_m_subj_mass_ht_list, image_zoom = 4.585/2, markers_gt_type = MARKERS_GT_TYPE, use_pc = True)

            else: pass
            test_dat_f_slp = SLPPrepLib().load_slp_files_to_database(testing_database_file_f, dana_lab_path, PM = COVER_TYPE, depth = COVER_TYPE, color = None, mass_ht_list=all_f_subj_mass_ht_list, markers_gt_type = MARKERS_GT_TYPE, use_pc = True)
            test_dat_m_slp = SLPPrepLib().load_slp_files_to_database(testing_database_file_m, dana_lab_path, PM = COVER_TYPE, depth = COVER_TYPE, color = None, mass_ht_list=all_m_subj_mass_ht_list, markers_gt_type = MARKERS_GT_TYPE, use_pc = True)

            test_dat_f_slp_nb = SLPPrepLib().load_slp_files_to_database(testing_database_file_f, dana_lab_path, PM = 'uncover', depth = 'uncover', color = None, mass_ht_list=all_f_subj_mass_ht_list, markers_gt_type = MARKERS_GT_TYPE, use_pc = True)
            test_dat_m_slp_nb = SLPPrepLib().load_slp_files_to_database(testing_database_file_m, dana_lab_path, PM = 'uncover', depth = 'uncover', color = None, mass_ht_list=all_m_subj_mass_ht_list, markers_gt_type = MARKERS_GT_TYPE, use_pc = True)



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

        if self.opt.blanket == False or COVER_TYPE == 'uncover':
            test_x = TensorPrepLib().prep_depth_input_images(test_x, test_dat_f_slp, test_dat_m_slp, None, None, start_map_idx = depth_in_idx, depth_type = 'no_blanket')
        else:
            test_x = TensorPrepLib().prep_depth_input_images(test_x, test_dat_f_slp, test_dat_m_slp, None, None, start_map_idx = depth_in_idx, depth_type = 'all_meshes')



        self.test_x_tensor = torch.Tensor(test_x)


        test_y_flat = []  # Initialize the ground truth listhave
        test_y_flat = SLPPrepLib().prep_labels_slp(test_y_flat, test_dat_f_slp, num_repeats = 1,
                                                    z_adj = -0.075, gender = "f", is_synth = True, markers_gt_type = MARKERS_GT_TYPE, x_y_adjust_mm = [X_ADJ_MM, Y_ADJ_MM]) #not sure if we should us is_synth true or false???
        test_y_flat = SLPPrepLib().prep_labels_slp(test_y_flat, test_dat_m_slp, num_repeats = 1,
                                                    z_adj = -0.075, gender = "m", is_synth = True, markers_gt_type = MARKERS_GT_TYPE, x_y_adjust_mm = [X_ADJ_MM, Y_ADJ_MM])

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

        if opt.blanket == False:
            dat_pc = test_dat_m_slp['pc'] + test_dat_f_slp['pc']
        else:
            dat_pc = test_dat_m_slp_nb['pc'] + test_dat_f_slp_nb['pc']


        if self.opt.savefig == True and opt.blanket == False:
            dat_color_render = test_dat_m_slp_lg['overhead_colorcam_noblanket'] + test_dat_f_slp_lg['overhead_colorcam_noblanket']
            dat_depth_render = test_dat_m_slp_lg['overhead_depthcam_noblanket'] + test_dat_f_slp_lg['overhead_depthcam_noblanket']
        elif self.opt.savefig == True and opt.blanket == True:
            dat_color_render = test_dat_m_slp_nb_lg['overhead_colorcam_noblanket'] + test_dat_f_slp_nb_lg['overhead_colorcam_noblanket']
            if COVER_TYPE == "uncover":
                dat_depth_render = test_dat_m_slp_nb_lg['overhead_depthcam_noblanket'] + test_dat_f_slp_nb_lg['overhead_depthcam_noblanket']
            else:
                dat_depth_render = test_dat_m_slp_lg['overhead_depthcam'] + test_dat_f_slp_lg['overhead_depthcam']
        else:
            dat_color_render = None
            dat_depth_render = None



        #for im_num in range(29, 100):
        for im_num in range(0, np.shape(test_x)[0]):#self.color_all.shape[0]):


            print("TESTING IM NUM ", im_num)

            #PRESSURE
            self.pressure = test_x[im_num, 0, :, :]


            #because we used a sheet on the bed the overall pressure is lower than calibration, which was done without a sheet. bump it up here.
            bedsheet_norm_factor = float(1)


            self.pressure = np.clip(self.pressure*bedsheet_norm_factor, 0, 100)


            self.depth = test_x[im_num:im_num+1, 1:, :, :]


            #now do 3D rendering
            pmat = np.clip(self.pressure.reshape(MAT_SIZE), a_min=0, a_max=100)




            pc_autofil_red = dat_pc[im_num]

            if self.opt.savefig == True:
                self.color_im_render = dat_color_render[im_num]
                self.depth_im_render = dat_depth_render[im_num]

                print(np.shape(dat_color_render))
                print(np.shape(dat_depth_render))

            #camera_point = [1.09898028, 0.46441343, -CAM_BED_DIST] #[dist from foot of bed, dist from left side of mat, dist normal]

            #
            self.estimate_pose(pmat, pc_autofil_red, model, model2, im_num)

            #self.pyRender.render_3D_data(camera_point, pmat = pmat, pc = pc_autofil_red)

            self.point_cloud_array = None
            #sleep(100)

        #self.compute_pck_slp()
        if self.opt.viz == '3D' and self.opt.savefig == False:
            if self.opt.pimgerr == True:
                if self.opt.pmr == False:
                    dir = FILEPATH_PREFIX + '/results/results_pressure/' + NETWORK_1 + '_' + COVER_TYPE
                else:
                    dir = FILEPATH_PREFIX + '/results/results_pressure/' + NETWORK_2 + '_' + COVER_TYPE
            else:
                if self.opt.pmr == False:
                    dir = FILEPATH_PREFIX + '/results/results_3DVPE/'+NETWORK_1+'_'+COVER_TYPE
                else:
                    dir = FILEPATH_PREFIX + '/results/results_3DVPE/'+NETWORK_2+'_'+COVER_TYPE

            if not os.path.exists(dir):
                os.mkdir(dir)

            participant_num = '%05d' % (opt.p_idx)

            pkl.dump(self.RESULTS_DICT, open(dir+'/results_slp_3D_'+participant_num+'.p', 'wb'))
        else:
            self.save_pck_results(self.joint_error_list, self.hd_th_dist_list)




    def compute_2D_error(self, target, score):


        target_2D = np.stack((target[1, 0:2],
                            target[2, 0:2],
                            target[4, 0:2],
                            target[5, 0:2],
                            target[7, 0:2],
                            target[8, 0:2],
                            target[12, 0:2],
                            target[15, 0:2],
                            target[16, 0:2],
                            target[17, 0:2],
                            target[18, 0:2],
                            target[19, 0:2],
                            target[20, 0:2],
                            target[21, 0:2]), axis = 0)

        score_2D = np.stack((score[1, 0:2],
                            score[2, 0:2],
                            score[4, 0:2],
                            score[5, 0:2],
                            score[7, 0:2],
                            score[8, 0:2],
                            score[12, 0:2],
                            score[15, 0:2],
                            score[16, 0:2],
                            score[17, 0:2],
                            score[18, 0:2],
                            score[19, 0:2],
                            score[20, 0:2],
                            score[21, 0:2]), axis = 0)

        xy_diff = target_2D - score_2D
        eulc_err = np.linalg.norm(xy_diff, axis = 1)
        print(eulc_err)
        return eulc_err, target_2D, score_2D


    def compute_3D_error(self, target_3D, score_3D):
        score_3D = score_3D[:24, :]
        xy_diff = target_3D - score_3D
        eulc_err = np.linalg.norm(xy_diff, axis = 1)
        print(eulc_err, 'eulc err')

        return eulc_err, target_3D, score_3D



    def compute_pck_slp(self):

        vis2D_list = list(np.ones((np.shape(self.target_list)[0], np.shape(self.target_list)[1], 1)))
        print(np.shape(vis2D_list))
        print(len(vis2D_list))
        #print(vis2D_list.shape)


        preds_ori = np.array(self.score_list) #input list joints Bx14x2
        joints_ori = np.array(self.target_list) #input list joints Bx14x2
        joints_vis = np.array(vis2D_list) #input list just a bunch of ones that is Bx14x1
        l_std_ori_all = np.array(self.hd_th_dist_list)# np.concatenate(self.hd_th_dist_list, axis=0) #input list norm of joints ori head and thorax. B


        pck_new_dict = {}
        pck_new_dict['preds_ori'] = preds_ori
        pck_new_dict['joints_ori'] = joints_ori
        pck_new_dict['joints_vis'] = joints_vis
        pck_new_dict['l_std_ori_all'] = l_std_ori_all


        print(np.shape(preds_ori), np.shape(joints_ori), np.shape(joints_vis), np.shape(l_std_ori_all))

        err_nmd = SLPPrepLib().distNorm(preds_ori, joints_ori, l_std_ori_all)
        ticks = np.linspace(0, 0.5, 11)  # 11 ticks
        pck_all = SLPPrepLib().pck(err_nmd, joints_vis, ticks=ticks)

        pck_new_dict['pck'] = pck_all

        #if self.opt.pmr == True:
        #    pkl.dump(pck_new_dict, open('/home/henry/git/sim_camera_resting_scene/data_BR/'+NETWORK_2+'_pck_last12_slp_'+COVER_TYPE+'.p', 'wb'))
        #else:
        #    pkl.dump(pck_new_dict, open('/home/henry/git/sim_camera_resting_scene/data_BR/'+NETWORK_1+'_pck_last12_slp_'+COVER_TYPE+'.p', 'wb'))

        print(np.shape(pck_all))
        print(pck_all)



    def save_pck_results(self, all_error, hd_th_dist_list):
        all_error = np.array(all_error)
        hd_th_dist_list = np.array(hd_th_dist_list).flatten()


        if MARKERS_GT_TYPE == '2D':
            all_error_hips = all_error[:, 0:2].flatten()
            all_error_knees = all_error[:, 2:4].flatten()
            all_error_ankles = all_error[:, 4:6].flatten()
            all_error_head = all_error[:, 7:8].flatten()
            all_error_shoulders = all_error[:, 8:10].flatten()
            all_error_elbows = all_error[:, 10:12].flatten()
            all_error_wrists = all_error[:, 12:14].flatten()
            all_error = all_error.flatten()
        else:

            all_error_hips = all_error[:, 1:3].flatten()
            all_error_knees = all_error[:, 4:6].flatten()
            all_error_ankles = all_error[:, 7:9].flatten()
            all_error_head = all_error[:, 15:16].flatten()
            all_error_shoulders = all_error[:, 16:18].flatten()
            all_error_elbows = all_error[:, 18:20].flatten()
            all_error_wrists = all_error[:, 20:22].flatten()
            all_error = all_error.flatten()

        #if test_crit == 'all_subjects':
        #    save_idx = 45*101
        #else: #test_crit == 'last12':
        #    save_idx = 45*12
        pck_dict = {}

        #if np.shape(all_error_head)[0] == save_idx: #then plot stuff
        pck_dict['ankle_allerror'] = all_error_ankles
        pck_dict['knee_allerror'] = all_error_knees
        pck_dict['hips_allerror'] = all_error_hips
        pck_dict['shoulders_allerror'] = all_error_shoulders
        pck_dict['elbows_allerror'] = all_error_elbows
        pck_dict['wrists_allerror'] = all_error_wrists
        pck_dict['head_allerror'] = all_error_head
        pck_dict['total_allerror'] = all_error
        pck_dict['hd_th_dist'] = hd_th_dist_list

        #if opt.blanket == True:
        pkl.dump(pck_dict, open('/home/henry/git/sim_camera_resting_scene/data_BR/'+NETWORK_2+'_pck_71to80_slp_'+COVER_TYPE+'_'+str(X_ADJ_MM) + 'mmx_'+ str(Y_ADJ_MM) +'mmy.p', 'wb'))
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

        print(betas_gt, "BETAS GT")

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

            # joint_cart_gt = np.array(self.m_female.J_transformed).reshape(24, 3)
            # for s in range(root_shift_est_gt.shape[0]):
            #    joint_cart_gt[:, s] += (root_shift_est_gt[s] - float(self.m_female.J_transformed[0, s]))
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

            # joint_cart_gt = np.array(self.m_male.J_transformed).reshape(24, 3)
            # for s in range(root_shift_est_gt.shape[0]):
            #    joint_cart_gt[:, s] += (root_shift_est_gt[s] - float(self.m_male.J_transformed[0, s]))
            print('MALE')
        return smpl_verts_gt


    def estimate_pose(self, pmat, pc_autofil_red, model, model2, im_num):

        bedangle = 0

        mat_size = (64, 27)

        #pmat = np.fliplr(np.flipud(np.clip(pmat.reshape(MAT_SIZE) * float(self.CTRL_PNL['pmat_mult']), a_min=0, a_max=100)))
        pmat = np.clip(pmat.reshape(MAT_SIZE) * float(self.CTRL_PNL['pmat_mult']), a_min=0, a_max=100)

        #if self.CTRL_PNL['cal_noise'] == False:
        #    pmat = gaussian_filter(pmat, sigma=0.5)

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
        if opt.pmr == True:
            self.CTRL_PNL['compute_forward_maps'] = True

        scores, INPUT_DICT, OUTPUT_DICT = UnpackDepthBatchLib().unpack_batch(batch, is_training=False, model=model, model_smpl_pmr = self.model_smpl_pmr,
                                                                             model_adjQP = model_adjQP, model_betanet = model_betanet,
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
                                                                                 model_adjQP = model_adjQP, model_betanet = model_betanet,
                                                                                 CTRL_PNL = self.CTRL_PNL)

        self.CTRL_PNL['first_pass'] = False





        if self.CTRL_PNL['slp'] != 'real':
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


        if MARKERS_GT_TYPE == '3D':
            smpl_verts_gt = self.get_SMPL_verts(batch1, INPUT_DICT)
            #est = OUTPUT_DICT['batch_angles_est'].cpu().numpy()[0]
            #gt = torch.mean(batch1[:, 82:154], dim=0).numpy()
            #for i in range(72):
            #    print("angle comparison", i, est[i], gt[i])
        else:
            smpl_verts_gt = None



        if self.opt.viz == '2D':
            viz_type = "2D"
        elif self.opt.viz == '3D':
            viz_type = "3D"
        else:
            viz_type = None
        #viz_type = "2D"


        self.tar_sample = INPUT_DICT['batch_targets']

        self.tar_sample = self.tar_sample.view(-1, 24, 3)
        #self.tar_sample[:, :, 1] -= 100.

        self.tar_sample = self.tar_sample[0, :].squeeze() / 1000
        sc_sample = OUTPUT_DICT['batch_targets_est'].clone()
        sc_sample = sc_sample[0, :].squeeze() / 1000
        sc_sample = torch.cat((sc_sample, torch.Tensor([0.0, 0.0, 0.0])), dim=0)
        sc_sample = sc_sample.view([self.output_size_train[0] + 1, self.output_size_train[1]])



        if self.CTRL_PNL['slp'] != 'real' and MARKERS_GT_TYPE == '2D':
            sc_sample[15, :] = torch.Tensor(OUTPUT_DICT['verts'][:, 336, :])
            #sc_sample[15, :] = torch.Tensor(OUTPUT_DICT['verts'][336, :])


        #print sc_sample, 'SCORE SAMPLE'
        th_hd = [self.tar_sample.cpu().numpy()[12, 0:2], self.tar_sample.cpu().numpy()[15, 0:2]]

        #print(th_hd)
        #print(th_hd[0] - th_hd[1])
        #print()

        if MARKERS_GT_TYPE == '2D':
            all_error, target2D, score2D = self.compute_2D_error(self.tar_sample.cpu().numpy(), sc_sample.cpu().numpy())

            self.joint_error_list.append(all_error)
            self.target_list.append(target2D)
            self.score_list.append(score2D)
            self.hd_th_dist_list.append(np.linalg.norm(th_hd[0] - th_hd[1]))

        if MARKERS_GT_TYPE == '3D':
            all_error, target3D, score3D = self.compute_3D_error(self.tar_sample.cpu().numpy(), sc_sample.cpu().numpy())

            self.joint_error_list.append(all_error)
            self.target_list.append(target3D)
            self.score_list.append(score3D)


        print(np.mean(self.joint_error_list[-1]), 'average error for pose', im_num) #0.13761432 69 #0.14558357 70
        shape_err_list = np.shape(self.joint_error_list)
        if im_num == 44:
            print(np.shape(np.array(self.joint_error_list)[-45:, :]), np.mean(np.array(self.joint_error_list)[-45:, :]), 'mean for participant')


        #if self.CTRL_PNL['slp'] != 'real':
        #if viz_type == "2D" or viz_type == "3D":
        im_display_idx = 0
        self.VIZ_DICT = {}
        self.VIZ_DICT = VisualizationLib().get_depthnet_viz_maps(im_display_idx, INPUT_DICT, OUTPUT_DICT, self.VIZ_DICT, self.CTRL_PNL)
        self.VIZ_DICT = VisualizationLib().get_fcn_recon_viz_maps(im_display_idx, INPUT_DICT, OUTPUT_DICT,  self.VIZ_DICT, self.CTRL_PNL)
        self.CTRL_PNL['recon_map_input_est'] = False

        if self.opt.pmr == True:
            pmatV = self.VIZ_DICT['p_img'].data.numpy()#ndimage.zoom(pmatV, 0.5, order=1)
            if self.CTRL_PNL['slp'] != 'real':
                pmatV_est = self.VIZ_DICT['p_img_est'].data.numpy()

            if self.opt.pimgerr == True:
                self.RESULTS_DICT['pmat'].append(pmatV)
                self.RESULTS_DICT['pmat_est'].append(pmatV_est)

            pmatV = pmatV/np.sum(pmatV)
            if self.CTRL_PNL['slp'] != 'real':
                pmatV_est = pmatV_est/np.sum(pmatV_est)

        #print(OUTPUT_DICT['batch_weight_kg_est'] , 'est weight')



        if viz_type == "2D":
            VisualizationLib().visualize_depth_net(VIZ_DICT = self.VIZ_DICT,
                                                      targets_raw = self.tar_sample.cpu(), scores_net1 = sc_sample.cpu(),
                                                      block=True, max_depth = 2200)


        elif viz_type == "3D":
            depth = INPUT_DICT_mod1['batch_images'][0, int(INPUT_DICT_mod1['batch_images'].size()[1]) - 2, :].clone().numpy().reshape(128, 54)
            self.depth_im_render = zoom(depth, 3.435, order = 1)


            print(np.min(pc_autofil_red[:, 2]), np.max(pc_autofil_red[:, 2]))

            print(pc_autofil_red.shape)
            #pc_autofil_red = pc_autofil_red[pc_autofil_red[:, 2] > -0.7, :]
            #pc_autofil_red = pc_autofil_red[pc_autofil_red[:, 2] < -0.015, :]

            print(np.min(pmatV), np.max(pmatV), np.min(pmatV_est), np.max(pmatV_est))
            # render everything

            participant_savename = str('%05d' % (self.opt.p_idx))
            #if self.opt.blanket == True:
            participant_savename += ('_'+COVER_TYPE)



            USE_PREFILTERED_PC = False
            if USE_PREFILTERED_PC == True:
                im_num_save = ('%06d' % (im_num + 1))
                p_idx_save = ('%05d' % (self.opt.p_idx))
                pc_dir = '/home/henry/data/SLP/danaLab/' + p_idx_save + '/pointCloud/uncover/'
                pc_autofil_red = np.load(pc_dir + im_num_save + '_v2.npy')
                pc_autofil_red[:, 0] += 0.0286



            print(smpl_verts_gt)

            if self.opt.savefig == False:
                self.RESULTS_DICT = self.pyRender.render_mesh_pc_bed_pyrender_everything(smpl_verts, smpl_faces, camera_point,
                                                                                         bedangle, self.RESULTS_DICT,
                                                                                         smpl_verts_gt = smpl_verts_gt,
                                                                                         pc=pc_autofil_red, pmat=pmatV, pmat_est = pmatV_est,
                                                                                         smpl_render_points=False,
                                                                                         markers=[[0.0, 0.0, 0.0],
                                                                                                  [0.0, 1.0, 0.0],
                                                                                                  [0.0, 0.0, 0.0],
                                                                                                  [0.0, 0.0, 0.0]],
                                                                                         dropout_variance=dropout_variance,
                                                                                         targets=self.tar_sample.view(72).cpu(),
                                                                                         scores=sc_sample.cpu())
                input("Press Enter to continue...")


            else:
                self.RESULTS_DICT = self.pyRender.render_mesh_pc_bed_pyrender_everything(smpl_verts, smpl_faces, camera_point,
                                                                                        bedangle, self.RESULTS_DICT,
                                                                                        #smpl_verts_gt = smpl_verts_gt,
                                                                                        pc=pc_autofil_red, pmat=pmatV, pmat_est = pmatV_est,
                                                                                        smpl_render_points=False,
                                                                                        markers=[[0.0, 0.0, 0.0],
                                                                                                 [0.0, 1.5, 0.0],
                                                                                                 [0.0, 0.0, 0.0],
                                                                                                 [0.0, 0.0, 0.0]],
                                                                                        dropout_variance=dropout_variance,
                                                                                        color_im=self.color_im_render,
                                                                                        depth_im=self.depth_im_render,
                                                                                        tf_corners=None,
                                                                                        current_pose_type_ct=str(im_num),
                                                                                        participant=participant_savename)

            self.point_cloud_array = None


            #self.save_processed_pc(im_num, self.RESULTS_DICT)
        #print(self.RESULTS_DICT, "RESULTS DICT")
        #time.sleep(100)





    def save_processed_pc(self, im_num, RESULTS_DICT):

        im_num_save = ('%06d' % (im_num + 1))

        p_idx_save = ('%05d' % (self.opt.p_idx))

        pc = RESULTS_DICT['pc_red']
        pc_norm = RESULTS_DICT['pc_red_norm']

        pc_save_dir = '/home/henry/data/SLP/danaLab/'+p_idx_save+'/pointCloud/'
        if not os.path.exists(pc_save_dir):
            os.mkdir(pc_save_dir)

        pc_save_dir += 'uncover/'
        if not os.path.exists(pc_save_dir):
            os.mkdir(pc_save_dir)

        save_arr = np.stack((pc, pc_norm))
        print(save_arr.shape)

        np.save(pc_save_dir+im_num_save+'.npy', save_arr)

        print(im_num_save, p_idx_save)






if __name__ ==  "__main__":

    import optparse
    from optparse_lib import get_depthnet_options

    p = optparse.OptionParser()

    p = get_depthnet_options(p)

    p.add_option('--mod', action='store', type = 'int', dest='mod', default=1,
                 help='Choose a network.')

    p.add_option('--p_idx', action='store', type='int', dest='p_idx', default=0,
                 help='Choose a participant. Enter a number from 1 to 100.')

    p.add_option('--viz', action='store', dest='viz', default='None',
                 help='Visualize training. specify `2D` or `3D`.')

    p.add_option('--pimgerr', action='store_true', dest='pimgerr', default=False,
                 help='Compute pressure image error.')

    p.add_option('--savefig', action='store_true', dest='savefig', default=False,
                 help='Use blankets.')

    opt, args = p.parse_args()


    if opt.hd == True:
        dana_lab_path = '/media/henry/multimodal_data_2/data/SLP/danaLab/'
    else:
        dana_lab_path = '/home/henry/data/SLP/danaLab/'


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
        for i in range(71, 81):
        #for i in range(1, 91):
            all_subj_str_list.append('%05d' % (i))
    else:
        all_subj_str_list = ['%05d' % (opt.p_idx)]


    phys_arr = np.load('/home/henry/data/SLP/danaLab/physiqueData.npy')
    phys_arr[:, [2, 0]] = phys_arr[:, [0, 2]]
    testing_database_file_f = []
    testing_database_file_m = []
    all_f_subj_mass_ht_list = []
    all_m_subj_mass_ht_list = []
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

    if opt.blanket == False:
        COVER_TYPE = 'uncover'



    if opt.hd == False:
        #FILEPATH_PREFIX = "../data_BR"
        FILEPATH_PREFIX = "/home/henry/git/sim_camera_resting_scene/data_BR"
    else:
        FILEPATH_PREFIX = "/media/henry/multimodal_data_2/data_BR"



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
        NETWORK_1 += "9315ct_"
        NETWORK_2 += "9315ct_"
    elif opt.slp == "synth":
        NETWORK_1 += "85114ct_"
        NETWORK_2 += "85114ct_"
    elif opt.slp == "mixed":
        NETWORK_1 += "183114ct_"
        NETWORK_2 += "183114ct_"
    elif opt.slp == "mixedreal":
        NETWORK_1 += "94429ct_"
        NETWORK_2 += "94429ct_"
    else:
        NETWORK_1 += "184000ct_"
        NETWORK_2 += "184000ct_"


    if opt.go200 == True:
        NETWORK_1 += "128b_x1pm"
        NETWORK_2 += "128b_x1pm"
    elif opt.pmr == True:
        NETWORK_1 += "128b_x1pm"#_0.5rtojtdpth"
        NETWORK_2 += "128b_x1pm_0.5rtojtdpth_depthestin_angleadj"
    elif opt.depth_out_unet == True and opt.mod == 2:
        NETWORK_1 += "128b_x1pm"
        NETWORK_2 += "128b_x1pm_depthestin_angleadj"
    else:
        NETWORK_1 += "128b_x1pm"
        NETWORK_2 += "128b_x1pm_angleadj"

    if opt.reg_angles == True:
        NETWORK_1 += '_rgangs'
        NETWORK_2 += '_rgangs'

    if opt.loss_betas == True:
        NETWORK_1 += '_lb'
        NETWORK_2 += '_lb'

    if opt.v2v == True:
        NETWORK_2 += '_lv2v'

    if opt.noloss_htwt == True:
        NETWORK_1 += '_nlhw'
        NETWORK_2 += '_nlhw'

    if opt.blanket == True or opt.slp == 'real':
        NETWORK_1 += "_slpb"
        NETWORK_2 += "_slpb"
    else:
        NETWORK_1 += "_slpnb"
        NETWORK_2 += "_slpnb"

    if opt.htwt == True:
        NETWORK_1 += "_htwt"
        NETWORK_2 += "_htwt"

    if opt.depthnoise == True:
        NETWORK_1 += "_dpns"
        NETWORK_2 += "_dpns"

    if opt.slpnoise == True:
        NETWORK_1 += "_slpns"
        NETWORK_2 += "_slpns"

    if opt.loss_root == True:
        NETWORK_1 += "_rt"
        NETWORK_2 += "_rt"

    if opt.half_shape_wt == True:
        NETWORK_1 += "_hsw"
        NETWORK_2 += "_hsw"

    if opt.depth_out_unet == True:
        NETWORK_1 += "_dou"
        NETWORK_2 += "_dou"

    NETWORK_1 += "_100e"
    NETWORK_2 += "_40e"

    # if opt.go200 == False:
    #    filename1 = FILEPATH_PREFIX+"/convnets/resnet50_1_anglesDC_" + NETWORK_1 + "_100e_0.0001lr.pt"
    #    filename2 = FILEPATH_PREFIX+"/convnets/resnet50_2_anglesDC_" + NETWORK_2 + "_100e_0.0001lr.pt"
    # else:


    if opt.cnn == 'resnetunet':
        if opt.small == True:
            filename1 = FILEPATH_PREFIX + "/convnets/resnetunet34_1_" + NETWORK_1 + "_0.0001lr.pt"
            filename2 = FILEPATH_PREFIX + "/convnets/resnetunet34_2_" + NETWORK_2 + "_0.0001lr.pt"
        else:
            filename1 = FILEPATH_PREFIX + "/convnets/resnetunet34_1_" + NETWORK_1 + "_0.0001lr.pt"
            filename2 = FILEPATH_PREFIX + "/convnets/resnetunet34_2_" + NETWORK_2 + "_0.0001lr.pt"

        if opt.mod == 1:
            filename2 = None
        #filename1_new = FILEPATH_PREFIX + "/convnets/CVPR2021/resnetunet34_1_anglesDC_" + NETWORK_1 + "_0.0001lr_new.pt"
    else:
        if opt.small == True:
            filename1 = FILEPATH_PREFIX + "/convnets/resnet34_1_" + NETWORK_1 + "_0.0001lr.pt"#
            filename2 = FILEPATH_PREFIX + "/convnets/resnet34_2_" + NETWORK_2 + "_0.0001lr.pt"#_noiseyCorr.pt"
        elif opt.slp == "real":
            filename1 = FILEPATH_PREFIX + "/convnets/resnet34_1_" + NETWORK_1 + "_0.0001lr.pt"
            filename2 = FILEPATH_PREFIX + "/convnets/resnet34_2_" + NETWORK_2 + "_0.0001lr.pt"#_noiseyCorr.pt"
        else:
            filename1 = FILEPATH_PREFIX + "/convnets/resnet34_1_" + NETWORK_1 + "_0.0001lr.pt"
            filename2 = FILEPATH_PREFIX + "/convnets/resnet34_2_" + NETWORK_2 + "_0.0001lr.pt"#_noiseyCorr.pt"

        if opt.pmr == False:
            filename2 = None

    #filename_loss = load_pickle(FILEPATH_PREFIX + "/convnets/resnet34_losses_1_" + NETWORK_1 + "_0.0001lr.p")
    #filename_loss = load_pickle(FILEPATH_PREFIX + "/convnets/betanet_losses_1_anglesDC_184000ct_128b_x1pm_0.5rtojtdpth_rgangs_slpb_rt_500e_0.0001lr.p")
    #VisualizationLib().make_popup_2D_plot(np.linspace(0, 500, 14500), filename_loss['train_loss'], np.linspace(0, 500, 14500), filename_loss['val_loss'], )
    #VisualizationLib().make_popup_2D_plot(np.linspace(0, 500, 14500), filename_loss['train_loss'], np.linspace(0, 500, 14500), filename_loss['val_loss'], )



    if GPU == True:
        for i in range(0, 8):
            try:
                model = torch.load(filename1, map_location={'cuda:' + str(i): 'cuda:0'})
                model = model.cuda().eval()
                print ("Network 1 loaded.")
                break
            except:
                pass
        if filename2 is not None:
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
            model = torch.load(filename1, map_location='cpu')#), pickle_module=pkl)

        model = model.eval()
        print ("Network 1 loaded.")
        if filename2 is not None:
            if sys.version_info.major < 3:
                model2 = torch.load(filename2, map_location='cpu')
            else:
                model2 = torch.load(filename2, map_location='cpu', pickle_module=pkl)
                
            model2 = model2.eval()
            print ("Network 2 loaded.")
        else:
            model2 = None


    if opt.cnn == 'resnet':
        model_adjQP = torch.load(FILEPATH_PREFIX+ '/convnets/adjQP_1_anglesDC_184000ct_128b_500e_0.0001lr.pt',map_location='cpu')
    else:
        model_adjQP = None
    model_betanet = torch.load(FILEPATH_PREFIX+ '/convnets/betanet_1_anglesDC_184000ct_128b_volfrac_500e_0.0001lr.pt', map_location='cpu')
    #python evaluate_depthreal_slp.py  --depthnoise  --p_idx 91 --loss_root --rgangs --small --pmr  --viz '3D' --blanket


    #for X_ADJ_MM in [8, 10, 12, 14, 16]:
    #    for Y_ADJ_MM in [6, 8, 10, 12, 14]:
    for X_ADJ_MM in [18, 20, 22, 24, 26]:
        for Y_ADJ_MM in [6, 8, 10, 12, 14]:
            V3D = Viz3DPose(opt)
            F_eval = V3D.evaluate_data(testing_database_file_f, testing_database_file_m, model, model2)



