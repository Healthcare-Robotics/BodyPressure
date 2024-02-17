
#!/usr/bin/env python

#Bodies at Rest: Code to visualize real dataset.
#(c) Henry M. Clever
#Major updates made for CVPR release: December 10, 2019

#python evaluate_depthreal_slp.py --viz '2D' --depthnoise  --p_idx 1 --loss_root --pcsum --small --cnn 'resnetunet' --blanket



txtfile = open("/home/ganyong/Githubwork/Examples/BodyPressure/FILEPATH.txt")
FILEPATH = txtfile.read().replace("\n", "")
txtfile.close()


import sys
sys.path.insert(0, '../lib_py')
sys.path.insert(-1,FILEPATH+'smpl/smpl_webuser3')
sys.path.insert(-1,FILEPATH)
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
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

from lib_py.visualization_lib_bp import VisualizationLib
from lib_py.preprocessing_lib_bp import PreprocessingLib
from lib_py.tensorprep_lib_bp import TensorPrepLib
from lib_py.unpack_depth_batch_lib_bp import UnpackDepthBatchLib
import lib_py.kinematics_lib_bp as kinematics_lib_br
from lib_py.slp_prep_lib_bp import SLPPrepLib


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


    def evaluate_data(self, test_files_f, test_files_m):

        test_dat_f_synth = TensorPrepLib(opt=self.opt).load_files_to_database(test_files_f, creation_type='synth', reduce_data=False, depth_in=True)
        test_dat_m_synth = TensorPrepLib(opt=self.opt).load_files_to_database(test_files_m, creation_type='synth', reduce_data=False, depth_in=True)

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
            len_f =  np.shape(test_dat_f_synth['images'])[0]
        except:
            len_f = 0
        try:
            len_m =  np.shape(test_dat_m_synth['images'])[0]
        except:
            len_m = 0


        test_x = np.zeros((len_f + len_m, x_map_ct, 64, 27)).astype(np.float32)
        #allocate pressure images
        test_x = TensorPrepLib(opt=self.opt).prep_images(test_x, None, None, test_dat_f_synth, test_dat_m_synth, filter_sigma = 0.5, start_map_idx = pmat_gt_idx)


        self.mesh_reconstruction_maps = None
        self.reconstruction_maps_input_est = None

        test_x_nobl = TensorPrepLib(opt=self.opt).prep_depth_input_images(test_x, None, None, test_dat_f_synth, test_dat_m_synth, start_map_idx = depth_in_idx, depth_type = 'no_blanket', mix_bl_nobl = False)
        test_x = TensorPrepLib(opt=self.opt).prep_depth_input_images(test_x, None, None, test_dat_f_synth, test_dat_m_synth, start_map_idx = depth_in_idx, depth_type = 'all_meshes', mix_bl_nobl = False)


        test_y_flat = []  # Initialize the ground truth listhave
        for gender_synth in [["f", test_dat_f_synth], ["m", test_dat_m_synth]]:
            test_y_flat = TensorPrepLib(opt=self.opt).prep_labels(test_y_flat, gender_synth[1],
                                                        z_adj = -0.075, gender = gender_synth[0], is_synth = True,
                                                        loss_vector_type = 'anglesDC',
                                                        initial_angle_est = False, x_y_adjust_mm = self.CTRL_PNL['x_y_offset_synth'])


        test_y_flat = np.array(test_y_flat)
        self.test_y_tensor = torch.Tensor(test_y_flat)


        if self.opt.viz == '2D':
            try:
                dat_depth_render_uncover = test_dat_m_synth['overhead_depthcam_noblanket']
            except:
                dat_depth_render_uncover = []
            if test_dat_f_synth['overhead_depthcam_noblanket'] is not None:
                dat_depth_render_uncover += test_dat_f_synth['overhead_depthcam_noblanket']

            try:
                dat_depth_render = test_dat_m_synth['overhead_depthcam']
            except:
                dat_depth_render = []
            if test_dat_f_synth['overhead_depthcam'] is not None:
                dat_depth_render += test_dat_f_synth['overhead_depthcam']

        else:
            dat_depth_render_uncover = None
            dat_depth_render = None



        for im_num in range(opt.pose_num-1, np.shape(test_x)[0]):#self.color_all.shape[0]):

            #PRESSURE
            self.pressure = test_x[im_num, 0, :, :]
            self.depth = test_x[im_num:im_num+1, 1:, :, :]

            if self.opt.viz == '2D':
                print('LOADING DEPTH IM RENDER')
                self.depth_im_render = dat_depth_render[im_num]
                self.depth_im_render_uncover = dat_depth_render_uncover[im_num]

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

            participant_savename = str('%02d' % (self.opt.filenum)) + '_' + str('%05d' % (self.opt.pose_num))

            if self.opt.viz == '3D':
                self.pyRender.render_mesh_pc_bed_pyrender_everything(smpl_verts_gt, smpl_faces,
                                                                     camera_point,
                                                                     bedangle,
                                                                     smpl_verts_gt=smpl_verts_gt,
                                                                     pc=None,
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
                depth_im_render_uncover = zoom(self.depth_im_render_uncover, 3.435, order=1)
                self.pyRender.render_mesh_pc_bed_pyrender_everything(smpl_verts_gt, smpl_faces,
                                                                     pmat=self.pressure,
                                                                     pmat_est=self.pressure,
                                                                     color_im_occl=None,
                                                                     color_im=None,
                                                                     depth_im=depth_im_render,
                                                                     depth_im2=depth_im_render_uncover,
                                                                     current_pose_type_ct=str(im_num),
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
    p = get_depthnet_options(p)

    p.add_option('--mod', action='store', type='int', dest='mod', default=1,
                 help='')


    p.add_option('--filenum', action='store', type='int', dest='filenum', default=0,
                 help='Choose a participant. Enter a number from 1 to 18.')

    p.add_option('--pose_num', action='store', type='int', dest='pose_num', default=0,
                 help='Choose a pose index. Enter a number between 1 and 10000.')

    p.add_option('--viz', action='store', dest='viz', default='2D',
                 help='Visualize training. specify `2D` or `3D`.')

    opt, args = p.parse_args()


    if opt.filenum == 0:
        print("need to specify valid p_idx. choose an index between 1 and 18.")
        sys.exit()

    if opt.pose_num == 0:
        print("need to specify valid pose_num. choose a pose num between 1 and 10000ish.")
        sys.exit()

    from filename_input_lib_bp import FileNameInputLib
    FileNameInputLib1 = FileNameInputLib(opt, depth=False)
    test_database_file_f, test_database_file_m = FileNameInputLib1.get_slpsynth_pressurepose(True, '')  # _nonoise')

    test_files_f = [test_database_file_f]
    test_files_m = [test_database_file_m]

    FileNameInputLib2 = FileNameInputLib(opt, depth=True)
    test_database_file_depth_f, test_database_file_depth_m = FileNameInputLib2.get_slpsynth_pressurepose(True, '')

    test_files_f.append(test_database_file_depth_f)
    test_files_m.append(test_database_file_depth_m)

    V3D = Viz3DPose(opt)

    if opt.filenum >= 1 and opt.filenum <= 9:
        female_synth_idx = (opt.filenum-1)
        F_eval = V3D.evaluate_data([[test_files_f[0][female_synth_idx]], [test_files_f[1][female_synth_idx]]], [[],[]])
        # break

    elif opt.filenum >= 10 and opt.filenum <= 18:
        male_synth_idx = (opt.filenum-10)
        F_eval = V3D.evaluate_data([[],[]], [[test_files_m[0][male_synth_idx]], [test_files_m[1][male_synth_idx]]])
        # break






