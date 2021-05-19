
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
sys.path.insert(-1,FILEPATH+'smpl/smpl_webuser3')
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
#sys.path.remove('/home/henry/git/sim_camera_resting_scene/DPNet/')
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


import lib_basic_3Dviz as libPyBasic3D
import lib_basic_2Dviz as libPyBasic2D

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

# Use for CPU
GPU = False
dtype = torch.FloatTensor

class Viz3DPose():
    def __init__(self, opt):

        self.opt = opt

        if opt.viz == '2D':
            self.pyRender = libPyBasic2D.pyRenderMesh(render = True)
        else:
            self.pyRender = libPyBasic3D.pyRenderMesh(render = True)

        model_path_m = FILEPATH+'smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
        model_path_f = FILEPATH+'smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
        self.m_male = load_model(model_path_m)
        self.m_female = load_model(model_path_f)

        self.CTRL_PNL = {}
        self.CTRL_PNL['batch_size'] = 1
        self.CTRL_PNL['verbose'] = False
        self.CTRL_PNL['mesh_recon_map_labels'] = False #can only be true if we have 100% synthetic data for training
        self.CTRL_PNL['mesh_recon_map_labels_test'] = False #can only be true is we have 100% synth for testing
        self.CTRL_PNL['recon_map_input_est'] = False  #do this if we're working in a two-part regression
        self.CTRL_PNL['x_y_offset_synth'] = [12, -35]#[-7, -45]#[200, 200]#
        self.CTRL_PNL['clean_slp_depth'] = False


    def evaluate_data(self, testing_database_file_f, testing_database_file_m):


        if self.opt.viz == '2D':
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


        test_x = np.zeros((len_f + len_m, x_map_ct, 64, 27)).astype(np.float32)
        #allocate pressure images
        test_x = TensorPrepLib().prep_images(test_x, test_dat_f_slp, test_dat_m_slp, None, None, filter_sigma = 0.5, start_map_idx = pmat_gt_idx)


        self.mesh_reconstruction_maps = None
        self.reconstruction_maps_input_est = None

        if opt.ctype == 'uncover':
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


        if opt.ctype == 'uncover':
            dat_pc = test_dat_m_slp_nb['pc'] + test_dat_f_slp_nb['pc']
        else:
            dat_pc = test_dat_m_slp['pc'] + test_dat_f_slp['pc']


        if self.opt.viz == '2D':
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



        for im_num in range(opt.pose_num-1, np.shape(test_x)[0]):#self.color_all.shape[0]):


            #PRESSURE
            self.pressure = test_x[im_num, 0, :, :]
            self.depth = test_x[im_num:im_num+1, 1:, :, :]

            pc_autofil_red = dat_pc[im_num]

            if self.opt.viz == '2D':
                self.color_im_render = dat_color_render[im_num]
                self.color_im_occl_render = dat_color_occl_render[im_num]
                self.depth_im_render = dat_depth_render[im_num]

            bedangle = 0

            mat_size = (64, 27)

            pmat_stack = PreprocessingLib().preprocessing_create_pressure_only_stack([self.pressure], mat_size, self.CTRL_PNL)[0]
            pmat_stack = np.expand_dims(np.array(pmat_stack), 0)
            pmat_stack = torch.Tensor(pmat_stack)

            batch1 = self.test_y_tensor[im_num:im_num + 1, :]

            batch = []
            depth_stack = torch.Tensor(self.depth)

            batch.append(torch.cat((pmat_stack, depth_stack), dim=1))
            batch.append(batch1)

            NUMOFOUTPUTDIMS = 3
            NUMOFOUTPUTNODES_TRAIN = 24
            self.output_size_train = (NUMOFOUTPUTNODES_TRAIN, NUMOFOUTPUTDIMS)

            self.CTRL_PNL['recon_map_input_est'] = False
            self.CTRL_PNL['mod'] = 1

            smpl_faces = np.array(self.m_female.f)

            camera_point = [1.09898028, 0.46441343, -CAM_BED_DIST]

            smpl_verts_gt = self.get_SMPL_verts(batch1)

            self.tar_sample = torch.mean(batch1[:, 0:72], dim=0)
            self.tar_sample = self.tar_sample.view(-1, 24, 3)
            self.tar_sample = self.tar_sample[0, :].squeeze() / 1000

            participant_savename = str('%05d' % (self.opt.p_idx))
            participant_savename += ('_' + opt.ctype)

            if self.opt.viz == '3D':
                self.pyRender.render_mesh_pc_bed_pyrender_everything(smpl_verts_gt, smpl_faces,
                                                                     camera_point,
                                                                     bedangle,
                                                                     smpl_verts_gt=smpl_verts_gt,
                                                                     pc=pc_autofil_red,
                                                                     pmat=self.pressure,
                                                                     pmat_est=self.pressure,
                                                                     smpl_render_points=False,
                                                                     markers=[[0.0, 0.0, 0.0],
                                                                              [0.0, 1.0, 0.0],
                                                                              [0.0, 0.0, 0.0],
                                                                              [0.0, 0.0, 0.0]],
                                                                     dropout_variance=None,
                                                                     targets=self.tar_sample.view(
                                                                         72).cpu(),
                                                                     scores=None)

                input("Press Enter to continue...")

            else:
                depth_im_render = zoom(self.depth_im_render, 3.435, order=1)
                self.pyRender.render_mesh_pc_bed_pyrender_everything(smpl_verts_gt, smpl_faces,
                                                                     pmat=self.pressure,
                                                                     pmat_est=self.pressure,
                                                                     color_im_occl=self.color_im_occl_render,
                                                                     color_im=self.color_im_render,
                                                                     depth_im=depth_im_render,
                                                                     current_pose_type_ct=str(
                                                                         im_num),
                                                                     participant=participant_savename)

            self.point_cloud_array = None



    def get_SMPL_verts(self, batch1):
        betas_gt = torch.mean(batch1[:, 72:82], dim=0).numpy()
        angles_gt = torch.mean(batch1[:, 82:154], dim=0).numpy()
        root_shift_est_gt = torch.mean(batch1[:, 154:157], dim=0).numpy()
        gender_gt = batch1[:, 157:159]
        root_shift_est_gt[1] *= -1
        root_shift_est_gt[2] *= -1

        R_root = kinematics_lib_br.matrix_from_dir_cos_angles(angles_gt[0:3])
        flip_root_euler = np.pi
        flip_root_R = kinematics_lib_br.eulerAnglesToRotationMatrix([flip_root_euler, 0.0, 0.0])
        angles_gt[0:3] = kinematics_lib_br.dir_cos_angles_from_matrix(np.matmul(flip_root_R, R_root))


        if int(gender_gt.numpy()[0][0]) == 1:
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
        elif int(gender_gt.numpy()[0][0]) == 0:
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





if __name__ ==  "__main__":

    import optparse
    from optparse_lib import get_depthnet_options

    p = optparse.OptionParser()


    p.add_option('--p_idx', action='store', type='int', dest='p_idx', default=0,
                 help='Choose a participant. Enter a number from 1 to 102.')

    p.add_option('--pose_num', action='store', type='int', dest='pose_num', default=0,
                 help='Choose a pose index. Enter a number between 1 and 45.')

    p.add_option('--ctype', action='store', dest='ctype', default='None',
                 help='Visualize training. specify `uncover` or `cover1` or `cover2`.')

    p.add_option('--viz', action='store', dest='viz', default='2D',
                 help='Visualize training. specify `2D` or `3D`.')

    opt, args = p.parse_args()


    if opt.ctype not in ['uncover', 'cover1', 'cover2']:
        print("need to specify valid ctype of `uncover` or `cover1` or `cover2`.")
        sys.exit()

    if opt.p_idx == 0:
        print("need to specify valid p_idx. choose an index between 1 and 102.")
        sys.exit()

    if opt.pose_num == 0:
        print("need to specify valid pose_num. choose a pose num between 1 and 45.")
        sys.exit()


    dana_lab_path = FILEPATH + 'data_BP/SLP/danaLab/'

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
        if some_subj == '00007':
            print("can't visualize subject 7")
            sys.exit()
        if int(gender_bin) == 0:
            all_f_subj_mass_ht_list.append([phys_arr[int(some_subj) - 1][0], phys_arr[int(some_subj) - 1][1]])
            testing_database_file_f.append(some_subj)
        else:
            all_m_subj_mass_ht_list.append([phys_arr[int(some_subj) - 1][0], phys_arr[int(some_subj) - 1][1]])
            testing_database_file_m.append(some_subj)

    V3D = Viz3DPose(opt)

    F_eval = V3D.evaluate_data(testing_database_file_f, testing_database_file_m)



