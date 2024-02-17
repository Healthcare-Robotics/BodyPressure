#!/usr/bin/env python
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

try:
    import cPickle as pickle
except:
    import pickle as pickle

import random
from scipy import ndimage
import scipy.stats as ss
import scipy.io as sio
import torch
# from scipy.misc import imresize
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.filters import gaussian_filter
from lib_py.preprocessing_lib_bp import PreprocessingLib
import lib_py.kinematics_lib_bp as kinematics_lib_br

import cv2

import imageio
# PyTorch libraries
import argparse

MAT_WIDTH = 0.762  # metres
MAT_HEIGHT = 1.854  # metres
MAT_HALF_WIDTH = MAT_WIDTH / 2
NUMOFTAXELS_X = 84  # 73 #taxels
NUMOFTAXELS_Y = 47  # 30
INTER_SENSOR_DISTANCE = 0.0286  # metres
LOW_TAXEL_THRESH_X = 0
LOW_TAXEL_THRESH_Y = 0
HIGH_TAXEL_THRESH_X = (NUMOFTAXELS_X - 1)
HIGH_TAXEL_THRESH_Y = (NUMOFTAXELS_Y - 1)


txtfile = open("/home/ganyong/Githubwork/Examples/BodyPressure/FILEPATH.txt")
FILEPATH = txtfile.read().replace("\n", "")
txtfile.close()


def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

try:
    import cPickle as pkl


    def load_pickle(filename):
        with open(filename, 'rb') as f:
            return pkl.load(f)
except:
    import pickle as pkl


    def load_pickle(filename):
        with open(filename, 'rb') as f:
            return pkl.load(f, encoding='latin1')
try:
    from smpl.smpl_webuser.serialization import load_model
except:
    from smpl.smpl_webuser3.serialization import load_model


class SLPPrepLib():

    def load_slp_files_to_database(self, database_file, danaLabPath, PM=None, depth=None, color=None, mass_ht_list=None,
                                   image_zoom=0.665, filter_pc=True, markers_gt_type='2D', use_pc=True, depth_out_unet = False, pm_adjust_mm = [0, 0]):
        # load in the training or testing files.  This may take a while.
        # print "GOT HERE*!!", database_file
        filter_pc = False
        if markers_gt_type == '3D':
            self.m_female = load_model(
                # '../../../git/SMPL_python_v.1.0.0/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl')
                '/home/ganyong/Githubwork/Examples/BodyPressure/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl')
            self.m_male = load_model(
                # '../../../git/SMPL_python_v.1.0.0/smpl/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl')
                '/home/ganyong/Githubwork/Examples/BodyPressure/smpl/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl')

        self.filter_pc = filter_pc

        dat = {}
        if markers_gt_type == '2D':
            dat['markers_xy_m'] = []
        elif markers_gt_type == '3D':
            dat['markers_xyz_m'] = []
            dat['body_shape'] = []
            dat['joint_angles'] = []
            dat['root_xyz_shift'] = []

        dat['images'] = []
        dat['pc'] = []
        dat['body_mass'] = []
        dat['body_height'] = []
        if color == 'uncover':
            dat['overhead_colorcam_noblanket'] = []
        elif color is not None:
            dat['overhead_colorcam'] = []

        if depth == 'uncover':
            dat['overhead_depthcam_noblanket'] = []
        elif depth is not None:
            dat['overhead_depthcam'] = []

        if depth_out_unet == True:
            dat['overhead_depthcam_onlyhuman'] = []


        # load_num_per_part = 5 #45
        load_num_per_part = 45
        pm_adjust_cm = [0, 0]
        for i in range(2):
            if pm_adjust_mm[i] < 0:
                pm_adjust_cm[i] = int(float(pm_adjust_mm[i])/10. - 0.5)
            elif pm_adjust_mm[i] >= 0:
                pm_adjust_cm[i] = int(float(pm_adjust_mm[i])/10. + 0.5)


        pm_adjust_cm[0] = int(-pm_adjust_cm[0])
        pm_adjust_cm[1] = int(-pm_adjust_cm[1])

        # print(len(database_file), len(mass_ht_list), 'SUBJ MASS LIST')
        mass_ht_list_ct = 0

        #np.save(FILEPATH+'data_BP/slp_real_cleaned/depth_uncover_cleaned_0to102.npy', np.zeros((102, 45, 128, 54)).astype(np.float32))
        #np.save(FILEPATH+'data_BP/slp_real_cleaned/depth_cover1_cleaned_0to102.npy', np.zeros((102, 45, 128, 54)).astype(np.float32))
        #np.save(FILEPATH+'data_BP/slp_real_cleaned/depth_cover2_cleaned_0to102.npy', np.zeros((102, 45, 128, 54)).astype(np.float32))
        #np.save(FILEPATH+'data_BP/slp_real_cleaned/depth_onlyhuman_0to102.npy', np.zeros((102, 45, 128, 54)).astype(np.uint16))
        #np.save(FILEPATH+'data_BP/slp_real_cleaned/pressure_recon_Pplus_gt_0to102.npy', np.zeros((102, 45, 64, 27)).astype(np.int32))
        #np.save(FILEPATH+'data_BP/slp_real_cleaned/pressure_recon_C_Pplus_gt_0to102.npy', np.zeros((102, 45, 64, 27)).astype(np.bool))
        #np.save(FILEPATH+'data_BP/slp_real_cleaned/O_T_slp_0to102.npy', np.zeros((102, 45, 3)).astype(np.float64))
        #np.save(FILEPATH+'data_BP/slp_real_cleaned/slp_T_cam_0to102.npy', np.zeros((102, 45, 3)).astype(np.float64))

        if depth is not None:
            depth_cleaned_0to102 = np.load(FILEPATH+'data_BP/slp_real_cleaned/depth_'+depth+'_cleaned_0to102.npy')
            depth_onlyhuman_0to102 = np.load(FILEPATH+'data_BP/slp_real_cleaned/depth_onlyhuman_0to102.npy')
            #pressure_recon_Pplus_gt_0to102 = np.load(FILEPATH+'data_BP/slp_real_cleaned/pressure_recon_Pplus_gt_0to102.npy')
            #pressure_recon_C_Pplus_gt_0to102 = np.load(FILEPATH+'data_BP/slp_real_cleaned/pressure_recon_C_Pplus_gt_0to102.npy')
            O_T_slp_0to102 = np.load(FILEPATH+'data_BP/slp_real_cleaned/O_T_slp_0to102.npy')
            slp_T_cam_0to102 = np.load(FILEPATH+'data_BP/slp_real_cleaned/slp_T_cam_0to102.npy')


        #np.save(FILEPATH+'data_BP/slp_real_cleaned/depth_uncover_cleaned_0to102.npy', depth_uncover_cleaned_0to102)
        #np.save(FILEPATH+'data_BP/slp_real_cleaned/depth_cover1_cleaned_0to102.npy', depth_cover1_cleaned_0to102)
        #np.save(FILEPATH+'data_BP/slp_real_cleaned/depth_cover2_cleaned_0to102.npy', depth_cover2_cleaned_0to102)
        #np.save(FILEPATH+'data_BP/slp_real_cleaned/depth_onlyhuman_0to102.npy', depth_onlyhuman_0to102)
        #np.save(FILEPATH+'data_BP/slp_real_cleaned/pressure_recon_Pplus_gt_0to102.npy', pressure_recon_Pplus_gt_0to102)
        #np.save(FILEPATH+'data_BP/slp_real_cleaned/pressure_recon_C_Pplus_gt_0to102.npy', pressure_recon_C_Pplus_gt_0to102)
        #np.save(FILEPATH+'data_BP/slp_real_cleaned/O_T_slp_0to102.npy', O_T_slp_0to102)
        #np.save(FILEPATH+'data_BP/slp_real_cleaned/slp_T_cam_0to102.npy', slp_T_cam_0to102)




        for some_subject in database_file:
            # print(some_subject)

            self.some_subject = some_subject
            self.danaLabPath = danaLabPath

            # sys.exit()
            for i in range(1, 1 + load_num_per_part):
                dat['body_mass'].append(mass_ht_list[mass_ht_list_ct][0])
                dat['body_height'].append(mass_ht_list[mass_ht_list_ct][1] / 100.)

            mass_ht_list_ct += 1

            # here load the pressure images, depth images, and joints
            if PM is not None:
                pth_PM = os.path.join(danaLabPath + some_subject + '/PMarray/' + PM + '/')
                for i in range(1, 1 + load_num_per_part):
                    PM_arr = np.load(pth_PM + '{:06d}.npy'.format(i)).astype(np.float)

                    #PM_arr = PM_arr[1:191, 3:80]  # cut off the edges because the original pressure mat is like 1.90 x 0.77 while this one is 1.92 x 0.84.
                    if pm_adjust_cm[1] <= -1:
                        PM_arr = PM_arr[1-pm_adjust_cm[1]:192, 0:84]  # cut off the edges because the original pressure mat is like 1.90 x 0.77 while this one is 1.92 x 0.84.
                        if np.shape(PM_arr)[0] < 190:
                            PM_arr = np.concatenate((PM_arr, np.zeros((190-np.shape(PM_arr)[0], np.shape(PM_arr)[1]))), axis = 0)

                    elif pm_adjust_cm[1] == 0: #this is if you have 0 through 192 or 1 through 191
                        #print('got here')
                        PM_arr = PM_arr[1:191, :]

                    elif pm_adjust_cm[1] >= 1:
                        PM_arr = PM_arr[0:191-pm_adjust_cm[1], 0:84]
                        if np.shape(PM_arr)[0] < 190:
                            PM_arr = np.concatenate((np.zeros((190-np.shape(PM_arr)[0], np.shape(PM_arr)[1])), PM_arr), axis = 0)


                    if pm_adjust_cm[0] <= -4:
                        PM_arr = PM_arr[:, 3-pm_adjust_cm[0]:84]
                        if np.shape(PM_arr)[1] < 77:
                            PM_arr = np.concatenate((PM_arr, np.zeros((np.shape(PM_arr)[0], 77-np.shape(PM_arr)[1]))), axis = 1)

                    elif pm_adjust_cm[0] >= -3 and pm_adjust_cm[0] <= 2:
                        #for a -2 you want it like [6:83]
                        #for a 3 you want it like [1:78]
                        PM_arr = PM_arr[:, 3-pm_adjust_cm[0]:80-pm_adjust_cm[0]]

                    elif pm_adjust_cm[0] >= 3:
                        PM_arr = PM_arr[:, 0:80-pm_adjust_cm[0]]
                        if np.shape(PM_arr)[1] < 77:
                            PM_arr = np.concatenate((np.zeros((np.shape(PM_arr)[0], 77-np.shape(PM_arr)[1])), PM_arr), axis = 1)

                    PM_arr = gaussian_filter(PM_arr, sigma = 0.5/0.345)

                    PM_arr = zoom(PM_arr, (0.335, 0.355), order=1)

                    PM_arr = PM_arr * ((dat['body_mass'][-1] * 9.81) / (np.sum(PM_arr) * 0.0264 * 0.0286)) * (1 / 133.322) #normalize by body mass and convert to mmHg

                    # print(PM_arr.shape, "PM ARR SHAPE")

                    dat['images'].append(PM_arr)
                    # print(np.max(dat['images'][-1]), 'max pim')

            if depth is not None:
                pth_PM = os.path.join(danaLabPath + some_subject + '/depthRaw/' + depth + '/')

                self.calibrate_depth = True
                # print(some_subject, i)

                if True:  # load existing cleaned depth image
                    #depth_arr_set45 = np.load(FILEPATH + 'data_BP/slp_real_cleaned/' + some_subject + '/depth_'+depth+'_cleaned.npy')
                    #O_T_slp_set45 = np.load(FILEPATH + 'data_BP/slp_real_cleaned/' + some_subject + '/O_T_slp.npy')
                    #slp_T_cam_set45 = np.load(FILEPATH + 'data_BP/slp_real_cleaned/' + some_subject + '/slp_T_cam.npy')
                    depth_arr_set45 = np.array(depth_cleaned_0to102[int(some_subject)-1, :])
                    O_T_slp_set45 = np.array(O_T_slp_0to102[int(some_subject)-1, :])
                    slp_T_cam_set45 = np.array(slp_T_cam_0to102[int(some_subject)-1, :])
                    for i in range(1, 1 + load_num_per_part):


                        depth_arr = np.array(depth_arr_set45[i-1, :])
                        self.O_T_slp = np.array(O_T_slp_set45[i-1, :])
                        self.slp_T_cam = np.array(slp_T_cam_set45[i-1, :])


                        if use_pc == True:
                            depth_arr_in = np.load(pth_PM + '{:06d}.npy'.format(i))
                            PTr_A2B = self.get_PTr_A2B(modA='depth', modB='PM')
                            pc = self.convert_depth_2_pc(depth_arr_in, depth, i)
                            dat['pc'].append(pc)

                        if depth == 'uncover':
                            try:
                                dat['overhead_depthcam_noblanket'].append(depth_arr)
                            except:
                                dat['overhead_depthcam_noblanket'] = []
                                dat['overhead_depthcam_noblanket'].append(depth_arr)
                        elif depth == 'cover2' or depth == 'cover1':
                            try:
                                dat['overhead_depthcam'].append(depth_arr)
                            except:
                                dat['overhead_depthcam'] = []
                                dat['overhead_depthcam'].append(depth_arr)

                else:
                    depth_arr_list45 = []
                    depth_arr_onlyhuman_list45 = []
                    O_T_slp_list45 = []
                    slp_T_cam_list45 = []
                    for i in range(1, 1 + load_num_per_part):


                        depth_arr_in = np.load(pth_PM + '{:06d}.npy'.format(i))

                        PTr_A2B = self.get_PTr_A2B(modA='depth', modB='PM')
                        # sz_B = [84, 192]
                        # sz_A = [424, 512]
                        # sz_A = [155, 345]

                        # print(np.shape(depth_arr_in), np.std(depth_arr_in), np.mean(depth_arr_in), np.max(depth_arr_in))

                        # depth_arr_lg = cv2.warpPerspective(depth_arr_in, PTr_A2B, tuple(sz_A)) #this is wrong. PTr_A2B isn't designed for a tuplie this size
                        # print(np.shape(depth_arr_lg), np.std(depth_arr_lg), np.mean(depth_arr_lg), np.max(depth_arr_lg))

                        pc = self.convert_depth_2_pc(depth_arr_in, depth, i)
                        dat['pc'].append(pc)

                        sz_B = [84, 192]
                        # sz_B = [56, 128]
                        depth_arr = cv2.warpPerspective(depth_arr_in, PTr_A2B, tuple(sz_B)).astype(np.int16)
                        # depth_arr = depth_arr[1:191, 3:80] #cut off the edges because the original pressure mat is like 1.90 x 0.77 while this one is 1.92 x 0.84.
                        # print(depth_arr.shape, "DEPTH ARR SHAPE")

                        depth_arr = zoom(depth_arr, image_zoom, order=1)
                        # print(depth_arr.shape, "DEPTH ARR SHAPE")

                        # depth_arr = depth_arr[32:160, 14:70]
                        depth_arr = depth_arr[:, 1:-1]  # - (2101 - 1660)

                        depth_arr = \
                        PreprocessingLib().clean_depth_images([0, torch.Tensor(depth_arr).unsqueeze(0).unsqueeze(0)])[1].clone().squeeze().cpu().numpy()

                        if depth == 'uncover':
                            try:
                                dat['overhead_depthcam_noblanket'].append(depth_arr)
                            except:
                                dat['overhead_depthcam_noblanket'] = []
                                dat['overhead_depthcam_noblanket'].append(depth_arr)
                        elif depth == 'cover2' or depth == 'cover1':
                            try:
                                dat['overhead_depthcam'].append(depth_arr)
                            except:
                                dat['overhead_depthcam'] = []
                                dat['overhead_depthcam'].append(depth_arr)

                        #depth_arr = np.load('/home/henry/data/01_init_poses/slp_gt_updated/uncover_' + some_subject + '/images/image_%06d/'% (i)+'depth_'+depth+'_cleaned.npy')
                        depth_arr_onlyhuman = np.load('/home/henry/data/01_init_poses/slp_gt_updated/uncover_' + some_subject + '/images/image_%06d/'% (i)+'depth_humanonly.npy')
                        #self.O_T_slp = np.load('/home/henry/data/01_init_poses/slp_gt_updated/uncover_' + some_subject + '/images/image_%06d/O_T_slp.npy' % (i))
                        #self.slp_T_cam = np.load('/home/henry/data/01_init_poses/slp_gt_updated/uncover_' + some_subject + '/images/image_%06d/slp_T_cam.npy' % (i))

                        depth_arr_list45.append(depth_arr)
                        depth_arr_onlyhuman_list45.append(depth_arr_onlyhuman)
                        O_T_slp_list45.append(self.O_T_slp)
                        slp_T_cam_list45.append(self.slp_T_cam)

                    np.save(FILEPATH + 'data_BP/slp_real_cleaned/' + some_subject + '/depth_'+depth+'_cleaned.npy' , np.array(depth_arr_list45))
                    np.save(FILEPATH + 'data_BP/slp_real_cleaned/' + some_subject + '/depth_onlyhuman.npy' , np.array(depth_arr_onlyhuman_list45))
                    np.save(FILEPATH + 'data_BP/slp_real_cleaned/' + some_subject + '/O_T_slp.npy', np.array(O_T_slp_list45))
                    np.save(FILEPATH + 'data_BP/slp_real_cleaned/' + some_subject + '/slp_T_cam.npy', np.array(slp_T_cam_list45))



                # regardless of what covering we use we should ALSO load the human only label if we are training the res U smpl net
                if depth_out_unet == True:
                    #depth_arr_onlyhuman_set45 = np.load(FILEPATH + 'data_BP/slp_real_cleaned/' + some_subject + '/depth_onlyhuman.npy')
                    depth_arr_onlyhuman_set45 = np.array(depth_onlyhuman_0to102[int(some_subject)-1, :])

                    for i in range(1, 1 + load_num_per_part):
                        dat['overhead_depthcam_onlyhuman'].append(depth_arr_onlyhuman_set45[i-1, :])






            if color is not None:
                pth_PM = os.path.join(danaLabPath + some_subject + '/RGB/' + color + '/')
                for i in range(1, 1 + load_num_per_part):
                    color_arr_in = imageio.imread(pth_PM + 'image_{:06d}.png'.format(i))

                    PTr_A2B = self.get_PTr_A2B(modA='RGB', modB='PM')
                    # sz_B = [84, 192]
                    # sz_A = [424, 512]
                    # sz_A = [155, 345]


                    sz_B = [84, 192]
                    # sz_B = [56, 128]
                    color_arr = cv2.warpPerspective(color_arr_in, PTr_A2B, tuple(sz_B)).astype(np.int16)
                    # color_arr = color_arr[1:191, 3:80] #cut off the edges because the original pressure mat is like 1.90 x 0.77 while this one is 1.92 x 0.84.

                    # print(color_arr[:, :, 0:1].shape)
                    color_arr_r = zoom(color_arr[:, :, 0], image_zoom, order=1)
                    # print(color_arr_r.shape)
                    color_arr_g = zoom(color_arr[:, :, 1], image_zoom, order=1)
                    # print(color_arr_g.shape)
                    color_arr_b = zoom(color_arr[:, :, 2], image_zoom, order=1)
                    # print(color_arr_b.shape)

                    color_arr = np.stack((color_arr_r, color_arr_g, color_arr_b), axis=2)

                    # depth_arr = depth_arr[32:160, 14:70]
                    color_arr = color_arr[:, 1:-1]  # - (2101 - 1660)

                    # print(color_arr.shape)

                    if color == 'uncover':
                        try:
                            dat['overhead_colorcam_noblanket'].append(color_arr)
                        except:
                            dat['overhead_colorcam_noblanket'] = []
                            dat['overhead_colorcam_noblanket'].append(color_arr)
                    elif color == 'cover2' or color == 'cover1':
                        try:
                            dat['overhead_colorcam'].append(color_arr)
                        except:
                            dat['overhead_colorcam'] = []
                            dat['overhead_colorcam'].append(color_arr)

            if markers_gt_type == '2D':
                try:
                    dat['markers_xy_m'] = np.concatenate(
                        (dat['markers_xy_m'], self.get_SLP_2D_markers()[0:load_num_per_part, :, :]), axis=0)
                except:
                    dat['markers_xy_m'] = self.get_SLP_2D_markers()[0:load_num_per_part, :, :]

                print(np.shape(dat['markers_xy_m']), 'shape markers 2D')

            elif markers_gt_type == '3D':
                try:
                    body_shape, joint_angles, root_xyz_shift, markers_xyz_m = self.get_SLP_3D_params()
                    dat['body_shape'] = np.concatenate((dat['body_shape'], body_shape), axis=0)
                    dat['joint_angles'] = np.concatenate((dat['joint_angles'], joint_angles), axis=0)
                    dat['root_xyz_shift'] = np.concatenate((dat['root_xyz_shift'], root_xyz_shift), axis=0)
                    dat['markers_xyz_m'] = np.concatenate((dat['markers_xyz_m'], markers_xyz_m), axis=0)
                except:
                    dat['body_shape'], dat['joint_angles'], dat['root_xyz_shift'], dat[
                        'markers_xyz_m'] = self.get_SLP_3D_params()

            #print(some_subject, np.shape(dat['images']))
            # print(np.shape(dat['overhead_depthcam_noblanket']))
        # print(np.shape(dat['overhead_depthcam']))

        return dat

    def load_slp_gt_maps_est_maps(self, files, dat_slp, data_fp_suffix, depth_out_unet = False):
        # print(files)
        if depth_out_unet == False:
            items_to_transfer_est = ['angles_est','root_xyz_est','betas_est','root_atan2_est','mdm_est','cm_est','bed_vertical_shift_est']
            items_to_transfer_gt = ['mesh_depth','mesh_contact']
        else:
            items_to_transfer_est = ['angles_est','root_xyz_est','betas_est','root_atan2_est','dimg_est', 'pimg_est']
            items_to_transfer_gt = None

        if len(data_fp_suffix) == 0:
            items_to_transfer_est = None


        if items_to_transfer_est is not None:
            for item in items_to_transfer_est:
                dat_slp[item] = []
        if items_to_transfer_gt is not None:
            for item in items_to_transfer_gt:
                dat_slp[item] = []

        pressure_recon_Pplus_gt_0to102 = np.load(FILEPATH+'data_BP/slp_real_cleaned/pressure_recon_Pplus_gt_0to102.npy')
        pressure_recon_C_Pplus_gt_0to102 = np.load(FILEPATH+'data_BP/slp_real_cleaned/pressure_recon_C_Pplus_gt_0to102.npy')

        for some_subj in files:
            if items_to_transfer_est is not None:
                est_maps = load_pickle(FILEPATH + 'data_BP/mod1est_real/'+some_subj+data_fp_suffix+'.p')
                for item in items_to_transfer_est:
                    for i in range(0, 45):
                        dat_slp[item].append(est_maps[item][i])

            if items_to_transfer_gt is not None:
                #gt_maps = load_pickle(FILEPATH + 'data_BP/slp_real_cleaned/'+some_subj+'/recon_pressure_maps_gt_Q-.p')
                #for item in items_to_transfer_gt:
                for i in range(0, 45):
                    dat_slp['mesh_depth'].append(pressure_recon_Pplus_gt_0to102[int(some_subj)-1, i, :, :])
                    dat_slp['mesh_contact'].append(pressure_recon_C_Pplus_gt_0to102[int(some_subj)-1, i, :, :])
                    #dat_slp[item].append(gt_maps[item][i])



        for some_subj in files:
            if items_to_transfer_est is not None:
                est_maps = load_pickle(FILEPATH + 'data_BP/mod1est_real/'+some_subj+data_fp_suffix+'.p')
                for item in items_to_transfer_est:
                    for i in range(45, 90):
                        dat_slp[item].append(est_maps[item][i])

            if items_to_transfer_gt is not None:
                #gt_maps = load_pickle(FILEPATH + 'data_BP/slp_real_cleaned/'+some_subj+'/recon_pressure_maps_gt_Q-.p')
                #for item in items_to_transfer_gt:
                for i in range(0, 45):
                    dat_slp['mesh_depth'].append(pressure_recon_Pplus_gt_0to102[int(some_subj)-1, i, :, :])
                    dat_slp['mesh_contact'].append(pressure_recon_C_Pplus_gt_0to102[int(some_subj)-1, i, :, :])
                    #dat_slp[item].append(gt_maps[item][i])


        for some_subj in files:
            if items_to_transfer_est is not None:
                est_maps = load_pickle(FILEPATH + 'data_BP/mod1est_real/'+some_subj+data_fp_suffix+'.p')
                for item in items_to_transfer_est:
                    for i in range(90, 135):
                        dat_slp[item].append(est_maps[item][i])

            if items_to_transfer_gt is not None:
                #gt_maps = load_pickle(FILEPATH + 'data_BP/slp_real_cleaned/'+some_subj+'/recon_pressure_maps_gt_Q-.p')
                #for item in items_to_transfer_gt:
                for i in range(0, 45):
                    dat_slp['mesh_depth'].append(pressure_recon_Pplus_gt_0to102[int(some_subj)-1, i, :, :])
                    dat_slp['mesh_contact'].append(pressure_recon_C_Pplus_gt_0to102[int(some_subj)-1, i, :, :])
                    #dat_slp[item].append(gt_maps[item][i])


        return dat_slp


    def get_bbox(self, joint_img, rt_margin=1.2, rt_xy=0):
        '''
        get the bounding box from joint gt min max, with a margin ratio.
        :param joint_img:
        :param rt_margin:
        :param rt_xy:   the ratio of x/y . 0 for original bb size. most times 1 for square input patch. Can be gotten from sz_pch[0]/sz_pch[1].
        :return:
        '''
        bb = np.zeros((4))
        xmin = np.min(joint_img[:, 0])
        ymin = np.min(joint_img[:, 1])
        xmax = np.max(joint_img[:, 0])
        ymax = np.max(joint_img[:, 1])
        width = xmax - xmin - 1
        height = ymax - ymin - 1
        if rt_xy:
            c_x = (xmin + xmax) / 2.
            c_y = (ymin + ymax) / 2.
            aspect_ratio = rt_xy
            w = width
            h = height
            if w > aspect_ratio * h:
                h = w / aspect_ratio
            elif w < aspect_ratio * h:
                w = h * aspect_ratio
            bb[2] = w * rt_margin
            bb[3] = h * rt_margin
            bb[0] = c_x - bb[2] / 2.
            bb[1] = c_y - bb[3] / 2.
        else:
            bb[0] = (xmin + xmax) / 2. - width / 2 * rt_margin
            bb[1] = (ymin + ymax) / 2. - height / 2 * rt_margin
            bb[2] = width * rt_margin
            bb[3] = height * rt_margin

        return bb

    def pixel2cam(self, pixel_coord, f, c):
        pixel_coord = pixel_coord.astype(float)
        # print(pixel_coord.shape, np.min(pixel_coord[:,0]), np.max(pixel_coord[:,0]), np.min(pixel_coord[:,1]), np.max(pixel_coord[:,1]), np.min(pixel_coord[:,2]), np.max(pixel_coord[:,2]))

        if self.filter_pc == True:
            pixel_coord = pixel_coord[pixel_coord[:, 0] < 141, :]

            pixel_coord2h = pixel_coord[pixel_coord[:, 1] <= 0, :]

            pixel_coord2h = pixel_coord2h[0:1][:]

            pixel_coord3h = pixel_coord[pixel_coord[:, 1] >= 10, :]

            pixel_coord = np.concatenate((pixel_coord2h, pixel_coord3h), axis=0)
            pixel_coord2v = pixel_coord[pixel_coord[:, 0] <= 0, :]
            # pixel_coord2h = pixel_coord2h[0:1][:]
            pixel_coord3v = pixel_coord[pixel_coord[:, 0] >= 10, :]

            pixel_coord = np.concatenate((pixel_coord2v, pixel_coord3v), axis=0)
            # pixel_coord = pixel_coord[pixel_coord[:, 1] < 331, :]

            pixel_coord = np.concatenate((pixel_coord, [[100, 0, 2600]]), axis=0)
            # print(pixel_coord.shape, np.min(pixel_coord[:,0]), np.max(pixel_coord[:,0]), np.min(pixel_coord[:,1]), np.max(pixel_coord[:,1]), np.min(pixel_coord[:,2]), np.max(pixel_coord[:,2]))
            # pixel_coord[:, 1] -= 100.
            # print(pixel_coord.shape, np.min(pixel_coord[:,0]), np.max(pixel_coord[:,0]), np.min(pixel_coord[:,1]), np.max(pixel_coord[:,1]), np.min(pixel_coord[:,2]), np.max(pixel_coord[:,2]))

            max_list = []
            num_to_edit = 20
            for i in range(num_to_edit):
                pixel_coordz = pixel_coord[pixel_coord[:, 1] == 330 - i, :]
                # print(np.max(pixel_coordz[-1, :]))
                max_list.append(np.max(pixel_coordz[-1, :]))
                # if np.max(pixel_coordz[-1, :]) < 2200: break

            cutoff = 0
            for i in range(num_to_edit):
                if max_list[i] > 2200:
                    cutoff = int(i)
            # print(max_list)

            pixel_coord = pixel_coord[pixel_coord[:, 1] < 330 - cutoff, :]

        # pixel_coord = pixel_coord[pixel_coord[:, 1] < 335, :]

        # pixel_coord = np.concatenate((pixel_coord, [[0.0, 0.0, 0.0]]), axis = 0)

        jt_cam = np.zeros_like(pixel_coord)
        jt_cam[..., 0] = (pixel_coord[..., 0] - c[0]) / f[0] * pixel_coord[..., 2]
        jt_cam[..., 1] = (pixel_coord[..., 1] - c[1]) / (f[1]) * pixel_coord[..., 2]
        jt_cam[..., 2] = pixel_coord[..., 2]

        # print(jt_cam.shape, np.min(jt_cam[:,0]), np.max(jt_cam[:,0]), np.min(jt_cam[:,1]), np.max(jt_cam[:,1]), np.min(jt_cam[:,2]), np.max(jt_cam[:,2]), 'jt cam shape')

        return jt_cam

    def get_ptc(self, depth, f, c, bb=None):
        '''
        get the list of the point cloud in flatten order, row -> column order.
        :param depth: 2d array with real depth value.
        :param f:
        :param c:
        :param bb: if cropping the image and only show the bb area, default none.
        :return: np array of vts list
        '''
        h, w = depth.shape
        vts = []  # lift for rst
        if bb is None:
            rg_r = (0, h)
            rg_c = (0, w)
        else:
            rg_r = (bb[1], bb[1] + bb[3])
            rg_c = (bb[0], bb[0] + bb[2])

        # print(rg_r[0], rg_r[1])
        # print(rg_c[0], rg_c[1])
        # print(depth.shape)

        for i in range(rg_r[0], rg_r[1]):
            for j in range(rg_c[0], rg_c[1]):
                next_vert = [j, i, depth[i, j]]
                vts.append(next_vert)
        vts = np.array(vts)
        # print('call ut_get ptc')
        vts_cam = self.pixel2cam(vts, f, c)

        if False:
            print('vts 0 to 5', vts[:5])
            print('after to cam is', vts_cam[:5])
        return vts_cam  # make to np array

    def convert_depth_2_pc(self, depth_arr, depth, i):

        if depth == 'uncover':
            cov_idx = 0
        elif depth == 'cover1':
            cov_idx = 1
        elif depth == 'cover2':
            cov_idx = 2

        self.c_d = [208.1, 259.7]  # z/f = x_m/x_p so m or mm doesn't matter
        self.f_d = [367.8, 367.8]

        if self.calibrate_depth == True:

            depth_Tr = self.genPTr_dict(['depth'])['depth'][0]  # self.get_PTr_A2B(modA='depth', modB='PM')
            # print(self.get_PTr_A2B(modA='depth', modB='PM'))
            depth_Tr[0:2, 0:3] = depth_Tr[0:2, 0:3] / (192. / 345.)

            # print('depthTR', depth_Tr)
            # print(np.matmul(depth_Tr, np.array([self.c_d[0], self.c_d[1], 1.0]).T))
            cd_modified = np.matmul(depth_Tr, np.array([self.c_d[0], self.c_d[1], 1.0]).T)
            cd_modified = cd_modified / cd_modified[2]

            depth_arr_mod = cv2.warpPerspective(depth_arr, depth_Tr, tuple([155, 345])).astype(
                np.int16)  # size of depth arr input is
            depth_arr_mod[0, 0] = 2101

            ptc = self.get_ptc(depth_arr_mod, self.f_d, cd_modified[0:2], None) / 1000
            # ptc = self.get_ptc(depth_arr_mod, self.f_d, self.c_d, None)/1000

        else:
            ptc = self.get_ptc(depth_arr, self.f_d, self.c_d, None) / 1000

        if self.filter_pc == True:
            rot_angle_fixed = np.deg2rad(3.0)
            # rot_angle_fixed = np.deg2rad(10.0)
            ptc[:, 0] = (ptc[:, 0]) * np.cos(rot_angle_fixed) - (ptc[:, 2]) * np.sin(rot_angle_fixed)
            ptc[:, 2] = (ptc[:, 0]) * np.sin(rot_angle_fixed) + (ptc[:, 2]) * np.cos(rot_angle_fixed)

            rot_angle = np.deg2rad(2.5)  #
            ptc[:, 1] = (ptc[:, 1]) * np.cos(rot_angle) - (ptc[:, 2]) * np.sin(rot_angle)
            ptc[:, 2] = (ptc[:, 1]) * np.sin(rot_angle) + (ptc[:, 2]) * np.cos(rot_angle)

            ptc = ptc[ptc[:, 2] < 2.103]

            ptc[:, 1] = (ptc[:, 1]) * np.cos(-rot_angle) - (ptc[:, 2]) * np.sin(-rot_angle)
            ptc[:, 2] = (ptc[:, 1]) * np.sin(-rot_angle) + (ptc[:, 2]) * np.cos(-rot_angle)

        # print(ptc, 'point cloud')
        ptc_first_point = np.array(ptc[0])
        # print(self.ptc_first_point, '1st pt')
        # x2 = ((x1 - x0) * cos(a)) - ((y1 - y0) * sin(a)) + x0;
        # y2 = ((x1 - x0) * sin(a)) + ((y1 - y0) * cos(a)) + y0;

        # ptc = ptc[ptc[:, 0] > -0.100, :]
        # ptc = ptc[ptc[:, 0] < 0.100, :]
        # ptc = ptc[ptc[:, 1] > -0.100, :]
        # ptc = ptc[ptc[:, 1] < 0.100, :]

        length_new_pmat = 1.92
        width_new_pmat = 0.84

        scale_diff_h = (length_new_pmat - 64 * 0.0286)
        scale_diff_w = (width_new_pmat - 27 * 0.0286)

        # this is because we set the first point in the depth image to 2101.
        ptc[:, 0] -= ptc_first_point[0]
        ptc[:, 1] -= ptc_first_point[1]
        ptc[:, 2] -= ptc_first_point[2]

        ptc[:, 0] -= (scale_diff_w)
        ptc[:, 1] -= (length_new_pmat - scale_diff_h)

        self.O_T_slp = [-scale_diff_w, -(length_new_pmat - scale_diff_h), 0.0]
        self.slp_T_cam = -np.array(ptc_first_point)

        # ptc += np.array([ 0.24229481,  1.23657246, -2.101     ])
        # [ 1.18873872  1.48349565 -2.101     ]

        ptc = np.concatenate((-ptc[:, 1:2], ptc[:, 0:1], ptc[:, 2:3]), axis=1)

        if self.filter_pc == True:
            # print(np.min(ptc[:, 0]), np.max(ptc[:, 0]), np.min(ptc[:, 1]), np.max(ptc[:, 1]), np.min(ptc[:, 2]), np.max(ptc[:, 2]))
            ptc = ptc[ptc[:, 1] > -0.01, :]  # cut off points at the edge
            ptc = ptc[ptc[:, 2] > -0.5, :]  # cut off stuff thats way up high
            ptc = ptc[ptc[:, 2] < 0.01]

        # print(np.min(ptc[:, 0]), np.max(ptc[:, 0]), np.min(ptc[:, 1]), np.max(ptc[:, 1]), np.min(ptc[:, 2]), np.max(ptc[:, 2]))
        return ptc

    def get_SLP_3D_params(self):

        phys_arr = np.load(self.danaLabPath + '/physiqueData.npy')
        phys_arr[:, [2, 0]] = phys_arr[:, [0, 2]]
        self.phys_arr = phys_arr.astype(np.float)  # all list
        #print(self.phys_arr.shape)

        #if int(self.some_subject[-3:]) > 80:
        #    save_new_xyz_root = True
        #else:
        save_new_xyz_root = False
        try:
            load_pickle(FILEPATH + 'data_BP/SLP_SMPL_fits/fits/p' + self.some_subject[-3:] + '/sd%02d.pkl' % (1))
            save_new_xyz_root = False
        except:
            save_new_xyz_root = True
            print("computing an extra set of ground truth pickle files with marker positions")


        gender = int(self.phys_arr[int(self.some_subject) - 1, 2])
        if gender == 0:
            gender = 'f'
            if save_new_xyz_root == True:
                m = self.m_female.copy()
        elif gender == 1:
            gender = 'm'
            if save_new_xyz_root == True:
                m = self.m_male.copy()
        #print('gender', gender)

        body_shape, joint_angles, root_xyz_shift, markers_xyz_m = [], [], [], []

        for pose_num in range(1, 46):

            if save_new_xyz_root == True:
                #original_pose_data = load_pickle('/home/henry/data_BP/01_init_poses/slp_gt_updated/uncover_' + self.some_subject + '/results/image_%06d/000.pkl' % (pose_num))
                original_pose_data = load_pickle(FILEPATH + 'data_BP/SLP_SMPL_fits/fits/p' + self.some_subject[-3:] + '/s%02d.pkl' % (pose_num))

                # here we need to flip the global orientation by 180 degrees
                # R_root = kinematics_lib_br.matrix_from_dir_cos_angles(original_pose_data['global_orient'])
                # flip_root_euler = np.pi
                # flip_root_R = kinematics_lib_br.eulerAnglesToRotationMatrix([flip_root_euler, 0.0, 0.0])
                # original_pose_data['global_orient'] = kinematics_lib_br.dir_cos_angles_from_matrix(np.matmul(R_root, flip_root_R))

                body_shape.append(original_pose_data['betas'])
                joint_angles.append(list(original_pose_data['global_orient']) + list(original_pose_data['body_pose']))

                for i in range(10):
                    m.betas[i] = float(body_shape[-1][i])

                for i in range(72):
                    m.pose[i] = float(joint_angles[-1][i])

                mJtransformed = np.array(m.J_transformed)

                original_pose_data['O_T_slp'] = np.array(self.O_T_slp)
                original_pose_data['slp_T_cam'] = np.array(self.slp_T_cam)
                # original_pose_data['slp_T_cam'] = np.array([ 0.18665951,  1.04323772 + 0.03, -2.101 + 0.03     ])
                original_pose_data['cam_T_Bo'] = np.array(original_pose_data['transl'])
                original_pose_data['Bo_T_Br'] = np.array(mJtransformed[0, :])

                original_pose_data['markers_xyz_m'] = np.array(mJtransformed)
                original_pose_data['markers_xyz_m'] += np.array(original_pose_data['O_T_slp']) + np.array(original_pose_data['slp_T_cam']) + np.array(original_pose_data['cam_T_Bo'])
                original_pose_data['markers_xyz_m'][:, 1] *= -1
                original_pose_data['markers_xyz_m'][:, 2] *= -1

                # [-0.0678 -1.8304  0.    ] [ 0.24229481  1.23657246 -2.101     ] [0.22207807 0.19696784 2.0123627 ] [-0.00181301 -0.26298164  0.0336967 ]

                pkl.dump(original_pose_data, open(FILEPATH + 'data_BP/SLP_SMPL_fits/fits/p' + self.some_subject[-3:] + '/sd%02d.pkl' % (pose_num), 'wb'))
                #pkl.dump(original_pose_data, open(
                #    '/home/henry/data/01_init_poses/slp_gt_updated/uncover_' + self.some_subject + '/results/image_%06d/001.pkl' % (
                #        pose_num), 'wb'))
                print("saved", pose_num)
            else:
                #original_pose_data = load_pickle(
                #    '/home/henry/data/01_init_poses/slp_gt_updated/uncover_' + self.some_subject + '/results/image_%06d/001.pkl' % (
                #        pose_num))
                original_pose_data = load_pickle(FILEPATH + 'data_BP/SLP_SMPL_fits/fits/p' + self.some_subject[-3:] + '/sd%02d.pkl' % (pose_num))

                # original_pose_data['slp_T_cam'] = np.array([ 0.18665951,  1.04323772 + 0.03, -2.101 + 0.03    ])
                original_pose_data['slp_T_cam'] = np.array(self.slp_T_cam)
                # print(self.slp_T_cam, "SLP T CAM") #[ 0.24229481  1.23657246 -2.101     ]

                # here we need to flip the global orientation by 180 degrees
                # print(original_pose_data['global_orient'], 'global orient')

                R_root = kinematics_lib_br.matrix_from_dir_cos_angles(original_pose_data['global_orient'])
                flip_root_euler = np.pi
                flip_root_R = kinematics_lib_br.eulerAnglesToRotationMatrix([flip_root_euler, 0.0, 0.0])
                original_pose_data['global_orient'] = kinematics_lib_br.dir_cos_angles_from_matrix(np.matmul(flip_root_R, R_root))

                body_shape.append(original_pose_data['betas'])
                joint_angles.append(list(original_pose_data['global_orient']) + list(original_pose_data['body_pose']))

                # original_pose_data['markers_xyz_m'][0, :] = np.array(original_pose_data['O_T_slp']) * np.array([1, -1, -1])
                # original_pose_data['markers_xyz_m'][1, :] = np.array(original_pose_data['O_T_slp']+original_pose_data['slp_T_cam']) * np.array([1, -1, -1])
                # original_pose_data['markers_xyz_m'][2, :] = np.array(original_pose_data['O_T_slp']+original_pose_data['slp_T_cam']+original_pose_data['cam_T_Bo']) * np.array([1, -1, -1])
                # original_pose_data['markers_xyz_m'][3, :] = np.array(original_pose_data['O_T_slp']+original_pose_data['slp_T_cam']+original_pose_data['cam_T_Bo']+original_pose_data['Bo_T_Br']) * np.array([1, -1, -1])
                # original_pose_data['markers_xyz_m'][4:, :] *= 0

            markers_xyz_m.append(original_pose_data['markers_xyz_m'])

            # print(original_pose_data['O_T_slp'], original_pose_data['slp_T_cam'], original_pose_data['cam_T_Bo'], original_pose_data['Bo_T_Br'])
            if self.calibrate_depth == True:
                root_xyz_shift.append(
                    original_pose_data['O_T_slp'] + original_pose_data['slp_T_cam'] + original_pose_data['cam_T_Bo'] +
                    original_pose_data[
                        'Bo_T_Br'])  # np.array(original_pose_data['transl']))# + np.array([0.0, 0.0, -2.0]))
            else:
                root_xyz_shift.append(
                    original_pose_data['O_T_slp'] + original_pose_data['cam_T_Bo'] + original_pose_data[
                        'Bo_T_Br'])  # np.array(original_pose_data['transl']))# + np.array([0.0, 0.0, -2.0]))
            # root_xyz_shift.append(original_pose_data['cam_T_Bo']+original_pose_data['Bo_T_Br'])#np.array(original_pose_data['transl']))# + np.array([0.0, 0.0, -2.0]))

            root_xyz_shift[-1][1] *= -1
            root_xyz_shift[-1][2] *= -1

            #if pose_num == 1:
            #    print(original_pose_data['O_T_slp'], 'O_T_slp')
            #    print(original_pose_data['slp_T_cam'], 'slp_T_cam')
            #    print(original_pose_data['cam_T_Bo'], 'cam_T_Bo')
            #    print(original_pose_data['Bo_T_Br'], 'Bo_T_Br')


        # for item in original_pose_data:
        #    print(pose_num, item, np.shape(original_pose_data[item]))
        # print(np.shape(markers_xyz_m), np.shape(root_xyz_shift))
        # print(np.array(markers_xyz_m)[:, 0, :])
        # print(np.array(root_xyz_shift))


        #print(np.shape(body_shape), np.shape(joint_angles), np.shape(root_xyz_shift), np.shape(markers_xyz_m), '&&&&&&&&&&&&&&&&&&&&&&')

        # sys.exit()
        return body_shape, joint_angles, root_xyz_shift, markers_xyz_m

    def get_SLP_2D_markers(self):

        dct_li_PTr = self.genPTr_dict(['RGB', 'IR', 'depth', 'PM'])
        phys_arr = np.load(self.danaLabPath + '/physiqueData.npy')
        phys_arr[:, [2, 0]] = phys_arr[:, [0, 2]]
        # wt(kg)
        # height,
        # gender(0 femal, 1 male),
        # bust (cm),waist ,
        # hip,
        # right upper arm ,
        # right lower arm,
        # righ upper leg,
        # right lower leg
        self.phys_arr = phys_arr.astype(np.float)  # all list

        joints_gt_RGB_t = sio.loadmat(self.danaLabPath + self.some_subject + '/joints_gt_RGB.mat')[
            'joints_gt']  # 3 x n_jt x n_frm -> n_jt x 3

        # print(np.shape(joints_gt_RGB_t))

        joints_gt_RGB_t = joints_gt_RGB_t.transpose([2, 1, 0])
        joints_gt_RGB_t = joints_gt_RGB_t - 1  # to 0 based

        # homography RGB to depth
        PTr_RGB = dct_li_PTr['RGB'][0]  # default to PM
        PTr_depth = dct_li_PTr['depth'][0]
        PTr_RGB2depth = np.dot(np.linalg.inv(PTr_depth), PTr_RGB)
        PTr_RGB2depth = PTr_RGB2depth / np.linalg.norm(PTr_RGB2depth)

        joints_gt_depth_t = np.array(list(
            map(lambda x: cv2.perspectiveTransform(np.array([x]), PTr_RGB2depth)[0], joints_gt_RGB_t[:, :, :2])))
        joints_gt_depth_t = np.concatenate([joints_gt_depth_t, joints_gt_RGB_t[:, :, 2, None]], axis=2)

        joints_gt_PM_t = np.array(list(
            map(lambda x: cv2.perspectiveTransform(np.array([x]), PTr_RGB)[0], joints_gt_RGB_t[:, :, :2])))
        joints_gt_PM_t = np.concatenate([joints_gt_PM_t, joints_gt_RGB_t[:, :, 2, None]], axis=2)

        # print(joints_gt_RGB_t[0])
        # print(joints_gt_depth_t[0])
        # print(joints_gt_PM_t[0])

        pth_cali = os.path.join(self.danaLabPath, self.some_subject, 'PMcali.npy')
        self.caliPM = np.load(pth_cali)  # 3 x 45

        self.li_bb_sq_depth = np.array(list(map(lambda x: self.get_bbox(x, rt_xy=1), joints_gt_depth_t)))

        return joints_gt_PM_t

    def genPTr_dict(self, mod_li):
        '''
        loop idx_li, loop mod_li then generate dictionary {mod[0]:PTr_li[...], mod[1]:PTr_li[...]}
        history: 6/3/20: add 'PM' as eye matrix for simplicity
        :param subj_li:
        :param mod_li:
        :return:
        '''

        PTr_dct_li_src = {}  # a dict
        for modNm in mod_li:  # initialize the dict_li
            PTr_dct_li_src[modNm] = []  # make empty list  {md:[], md2:[]...}

        # print(self.danaLabPath, self.some_subject)

        for mod in mod_li:  # add mod PTr
            if 'PM' not in mod:
                pth_PTr = os.path.join(self.danaLabPath, self.some_subject, 'align_PTr_{}.npy'.format(mod))
                PTr = np.load(pth_PTr)
            else:
                PTr = np.eye(3)  # fill PM with identical matrix
            # print(mod, PTr)
            PTr_dct_li_src[mod].append(PTr)
        return PTr_dct_li_src

    def get_PTr_A2B(self, modA, modB):
        dct_li_PTr = self.genPTr_dict(['RGB', 'IR', 'depth', 'PM'])

        PTrA = dct_li_PTr[modA][0]  # subj -1 for the li index
        PTrB = dct_li_PTr[modB][0]
        PTr_A2B = np.dot(np.linalg.inv(PTrB), PTrA)
        PTr_A2B = PTr_A2B / np.linalg.norm(PTr_A2B)  # normalize

        return PTr_A2B

    def prep_labels_slp(self, y_flat, dat, num_repeats, z_adj, gender, is_synth, markers_gt_type='2D',  initial_angle_est = False, cnn_type = 'resnet', x_y_adjust_mm = [0, 0]):
        if gender == "f":
            g1 = 1
            g2 = 0
        elif gender == "m":
            g1 = 0
            g2 = 1
        if is_synth == True:
            s1 = 1
        else:
            s1 = 0

        if dat is not None:

            if markers_gt_type == '2D':

                for entry in range(len(dat['markers_xy_m'])):

                    markers = np.concatenate((np.array(3 * [0]),
                                              dat['markers_xy_m'][entry][3, :],  # L hip
                                              dat['markers_xy_m'][entry][2, :],  # R hip
                                              np.array(3 * [0]),
                                              dat['markers_xy_m'][entry][4, :],  # L KNEE
                                              dat['markers_xy_m'][entry][1, :],  # R KNEE
                                              np.array(3 * [0]),
                                              dat['markers_xy_m'][entry][5, :],  # L ANKLE
                                              dat['markers_xy_m'][entry][0, :],  # R ANKLE
                                              np.array(9 * [0]),
                                              dat['markers_xy_m'][entry][12, :],  # NECK
                                              np.array(6 * [0]),
                                              dat['markers_xy_m'][entry][13, :],  # HEAD
                                              # fixed_head_markers,
                                              dat['markers_xy_m'][entry][9, :],  # L SHOULD
                                              dat['markers_xy_m'][entry][8, :],  # R SHOULD
                                              dat['markers_xy_m'][entry][10, :],  # L ELBOW
                                              dat['markers_xy_m'][entry][7, :],  # R ELBOW
                                              dat['markers_xy_m'][entry][11, :],  # L WRIST
                                              dat['markers_xy_m'][entry][6, :],  # R WRIST
                                              np.array(6 * [0])), axis=0).reshape(24, 3)

                    length_new_pmat = 1.92
                    width_new_pmat = 0.84

                    scale_diff_h = (length_new_pmat - 64 * 0.0286)
                    scale_diff_w = (width_new_pmat - 27 * 0.0286)

                    markers = markers * 0.01

                    markers[:, 0] = markers[:, 0] - scale_diff_w
                    markers[:, 1] = (-markers[:, 1] + length_new_pmat - scale_diff_h)

                    print(markers[:, 1])
                    markers *= 1000.
                    c = np.concatenate((markers.reshape(72),
                                        np.array(85 * [0]),
                                        [g1], [g2], [s1],
                                        [dat['body_mass'][entry]],
                                        [dat['body_height'][entry]],),
                                       axis=0)  # [x1], [x2], [x3]: female synth: 1, 0, 1.
                    y_flat.append(c)

            elif markers_gt_type == '3D':
                for entry in range(len(dat['markers_xyz_m'])):

                    markers_curr = np.array(dat['markers_xyz_m'][entry]) + np.array([x_y_adjust_mm[0]/1000., x_y_adjust_mm[1]/1000., 0.0])
                    markers_curr *= 1000.

                    if gender == "f":
                        c = np.concatenate((markers_curr.reshape(72),
                                            dat['body_shape'][entry][0:10],
                                            dat['joint_angles'][entry][0:72],
                                            dat['root_xyz_shift'][entry][0:3],
                                            [g1], [g2], [s1],
                                            [dat['body_mass'][entry]],
                                            [dat['body_height'][entry]],),
                                           axis=0)  # [x1], [x2], [x3]: female synth: 1, 0, 1.
                    elif gender == "m":
                        c = np.concatenate((markers_curr.reshape(72),
                                            dat['body_shape'][entry][0:10],
                                            dat['joint_angles'][entry][0:72],
                                            dat['root_xyz_shift'][entry][0:3],
                                            [g1], [g2], [s1],
                                            [dat['body_mass'][entry]],
                                            [dat['body_height'][entry]],),
                                           axis=0)  # [x1], [x2], [x3]: female synth: 1, 0, 1.
                    if initial_angle_est == True:
                        c = np.concatenate((c,
                                            dat['betas_est'][entry][0:10],
                                            dat['angles_est'][entry][0:72],
                                            dat['root_xyz_est'][entry][0:3],
                                            dat['root_atan2_est'][entry][0:6]), axis = 0)
                        if cnn_type == 'resnetunet':
                            c = np.concatenate((c, [0]), axis = 0)
                        else:
                            c = np.concatenate((c, dat['bed_vertical_shift_est'][entry][0:1]), axis = 0)



                    y_flat.append(c)
        return y_flat

    def distNorm(self, pos_pred_src, pos_gt_src, l_std):
        '''
        claculate the normalized distance  between pos_red_src to pos_gt_src
        :param pos_pred_src: the predict pose  nx 2(3)
        :param pos_gt_src:   the target pose
        :param l_std:
        :return: N x n_jt  normalized dist
        '''
        uv_error = pos_pred_src - pos_gt_src
        uv_err = np.linalg.norm(uv_error, axis=2)  # N x n_jt
        head_size = l_std[..., None]  # add last dim  # N x 1
        return uv_err / head_size  # get the

    def pck(self, errs, joints_vis, ticks):
        '''
        from the distance, calculate the pck value at each ticks.
        if don't want to use mask, simply set all vlaue of joints_vis to 1.
        :param errs: errors.  better to be normalized.  N x n_jt
        :param joints_vis:  visibility. Give all 1 if you want to count all. N xn_jt
        :param ticks:  the ticks need to be evaluated.
        :return: n_jt x n_ticks
        '''
        joints_vis = joints_vis.squeeze()  # N x 14 ?
        cnts = np.sum(joints_vis, axis=0)  # n_jt
        # print('cnts shape', cnts.shape)
        n_jt = errs.shape[1]
        li_pck = []
        jnt_ratio = cnts / np.sum(cnts).astype(np.float64)
        for i in range(len(ticks)):  # from 0
            pck_t = np.zeros(n_jt + 1)  # for last mean
            thr = ticks[i]
            hits = np.sum((errs <= thr) * joints_vis, axis=0)  # n_jt       60x14x1?
            # print('hits shape', hits.shape)
            # print('cnts shape', cnts.shape)
            pck_t[:n_jt] = hits / cnts  # n_jt     14 to 60
            pck_t[-1] = np.sum(pck_t[:-1] * jnt_ratio)  #
            li_pck.append(pck_t)
        pck_all = np.array(li_pck) * 100  # 11 x 14   to  %
        # print('pck all shape', pck_all.shape)
        # print('pck all T shape', pck_all.T.shape)
        return pck_all.T  # n_jt x n_ticks





