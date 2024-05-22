#!/usr/bin/env python
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import *

txtfile = open("../FILEPATH.txt")
FILEPATH = txtfile.read().replace("\n", "")
txtfile.close()


import sys
import os
import time
import matplotlib.gridspec as gridspec
import math


#PyTorch libraries
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable


#import tf.transformations as tft
from smpl.smpl_webuser.serialization import load_model

# import hrl_lib.util as ut
import cPickle as pickle
# from hrl_lib.util import load_pickle
def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


import cPickle as pkl
import random
from scipy import ndimage

np.set_printoptions(threshold=sys.maxsize)

MAT_WIDTH = 0.762 #metres
MAT_HEIGHT = 1.854 #metres
MAT_HALF_WIDTH = MAT_WIDTH/2
NUMOFTAXELS_X = 74#73 #taxels
NUMOFTAXELS_Y = 27#30
NUMOFOUTPUTDIMS = 3
NUMOFOUTPUTNODES = 10
INTER_SENSOR_DISTANCE = 0.0286#metres
LOW_TAXEL_THRESH_X = 0
LOW_TAXEL_THRESH_Y = 0
HIGH_TAXEL_THRESH_X = (NUMOFTAXELS_X - 1)
HIGH_TAXEL_THRESH_Y = (NUMOFTAXELS_Y - 1)


def visualize_pressure_map(p_map, targets_raw=None, scores_raw=None, p_map_val=None, targets_val=None, scores_val=None, block=False, title=' '):
    # p_map_val[0, :, :] = p_map[1, : ,:]

    try:
        p_map = p_map[0, :, :]  # select the original image matrix from the intermediate amplifier matrix and the height matrix
    except:
        pass

    plt.close()
    plt.pause(0.0001)

    fig = plt.figure()
    mngr = plt.get_current_fig_manager()
    # to put it into the upper left corner for example:
    # mngr.window.setGeometry(50, 100, 840, 705)

    plt.pause(0.0001)

    # set options
    if p_map_val is not None:
        try:
            p_map_val = p_map_val[0, :, :]  # select the original image matrix from the intermediate amplifier matrix and the height matrix
        except:
            pass
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        xlim = [-10.0, 37.0]
        ylim = [74.0, -10.0]
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)
        ax1.set_facecolor('cyan')
        ax2.set_facecolor('cyan')
        ax1.imshow(p_map, interpolation='nearest', cmap=
        plt.cm.jet, origin='upper', vmin=0, vmax=100)
        ax2.imshow(p_map_val, interpolation='nearest', cmap=
        plt.cm.jet, origin='upper', vmin=0, vmax=100)
        ax1.set_title('Training Sample \n Targets and Estimates')
        ax2.set_title('Validation Sample \n Targets and Estimates')


    else:
        ax1 = fig.add_subplot(1, 1, 1)
        #xlim = [-2.0, 49.0]
        #ylim = [86.0, -2.0]
        xlim = [-10.0, 37.0]
        ylim = [74.0, -10.0]
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        ax1.set_facecolor('cyan')
        ax1.imshow(p_map, interpolation='nearest', cmap=
        plt.cm.jet, origin='upper', vmin=0, vmax=100)
        ax1.set_title('Validation Sample \n Targets and Estimates \n' + title)

    # Visualize targets of training set
    if targets_raw is not None:
        if len(np.shape(targets_raw)) == 1:
            targets_raw = np.reshape(targets_raw, (len(targets_raw) / 3, 3))
        target_coord = targets_raw[:, :2] / INTER_SENSOR_DISTANCE
        target_coord[:, 0] -= 10
        target_coord[:, 1] -= (NUMOFTAXELS_X - 1)
        target_coord[:, 1] *= -1.0
        ax1.plot(target_coord[:, 0], target_coord[:, 1], marker='o', linestyle='None', markerfacecolor='green',
                 markeredgecolor='black', ms=8)
    plt.pause(0.0001)

    # Visualize estimated from training set
    if scores_raw is not None:
        if len(np.shape(scores_raw)) == 1:
            scores_raw = np.reshape(scores_raw, (len(scores_raw) / 3, 3))
        target_coord = scores_raw[:, :2] / INTER_SENSOR_DISTANCE
        target_coord[:, 1] -= (NUMOFTAXELS_X - 1)
        target_coord[:, 1] *= -1.0
        ax1.plot(target_coord[:, 0], target_coord[:, 1], marker='o', linestyle='None', markerfacecolor='yellow',
                 markeredgecolor='black', ms=8)
    plt.pause(0.0001)

    # Visualize targets of validation set
    if targets_val is not None:
        if len(np.shape(targets_val)) == 1:
            targets_val = np.reshape(targets_val, (len(targets_val) / 3, 3))
        target_coord = targets_val[:, :2] / INTER_SENSOR_DISTANCE
        target_coord[:, 0] -= 10
        target_coord[:, 1] -= (NUMOFTAXELS_X - 1)
        target_coord[:, 1] *= -1.0
        ax2.plot(target_coord[:, 0], target_coord[:, 1], marker='o', linestyle='None', markerfacecolor='green',
                 markeredgecolor='black', ms=8)
    plt.pause(0.0001)

    # Visualize estimated from training set
    if scores_val is not None:
        if len(np.shape(scores_val)) == 1:
            scores_val = np.reshape(scores_val, (len(scores_val) / 3, 3))
        target_coord = scores_val[:, :2] / INTER_SENSOR_DISTANCE
        target_coord[:, 1] -= (NUMOFTAXELS_X - 1)
        target_coord[:, 1] *= -1.0
        ax2.plot(target_coord[:, 0], target_coord[:, 1], marker='o', linestyle='None', markerfacecolor='yellow',
                 markeredgecolor='black', ms=8)
    plt.pause(0.50001)

    raw_input()
    plt.show(block=block)


def fix_mv_mf_hv_hf_bv_bf_pmat_k():

    #all_data_names = [#["f", "lay", 1944, "1to10"],
    #                  ["f", "lay", 2210, "11to20"],
    #                  ["f", "lay", 2201, "21to30"],
    #                  ["f", "lay", 2194, "31to40"]]
    #all_data_names = [#["f", "lside", 1857, "1to10"],
    #                  ["f", "lside", 2087, "11to20"],
    #                  ["f", "lside", 2086, "21to30"],
    #                  ["f", "lside", 2106, "31to40"]]
    #all_data_names = [#["f", "rside", 1805, "1to10"],
    #                  ["f", "rside", 2001, "11to20"]]#,
    #                  ["f", "rside", 1922, "21to30"],
    #                  ["f", "rside", 1949, "31to40"]]
    #all_data_names = [#["m", "lay", 1946, "1to10"],
    #                  ["m", "lay", 2192, "11to20"],
    #                  #["m", "lay", 2178, "21to30"],
    #                  ["m", "lay", 2177, "31to40"]]
    #all_data_names = [#["m", "lside", 1731, "1to10"],
    #                  ["m", "lside", 2007, "11to20"],
    #                  ["m", "lside", 2002, "21to30"],
    #                  ["m", "lside", 2021, "31to40"]]
    #all_data_names = [#["m", "rside", 1704, "1to10"],
    #                  ["m", "rside", 1927, "11to20"],
    #                  ["m", "rside", 1844, "21to30"]]
    #                  ["m", "rside", 1902, "31to40"]]



    #all_data_names = [#["f", "lay", 2198, "41to50", "train"],
    #                  ["f", "lay", 2197, "51to60", "train"],
    #                  ["f", "lay", 2213, "61to70", "train"]]#,
    #                  ["f", "lay", 2184, "71to80", "train"]]
    #all_data_names = [#["f", "lside", 2091, "41to50", "train"],
    #                  ["f", "lside", 2053, "51to60", "train"],
    #                  ["f", "lside", 2014, "61to70", "train"]]#,
    #                  ["f", "lside", 2058, "71to80", "train"]]
    #all_data_names = [#["f", "rside", 1976, "41to50", "train"],
                     # ["f", "rside", 2043, "51to60", "train"],
    #                  ["f", "rside", 1987, "61to70", "train"]]#,
    #                  ["f", "rside", 2010, "71to80", "train"]]
    #all_data_names = [#["m", "lay", 2195, "41to50", "train"],
    #                  ["m", "lay", 2199, "51to60", "train"],
    #                  ["m", "lay", 2203, "61to70", "train"]]#,
    #                  ["m", "lay", 2188, "71to80", "train"]]
    #all_data_names = [#["m", "lside", 2049, "41to50", "train"],
    #                  ["m", "lside", 1952, "51to60", "train"],
    #                  ["m", "lside", 1934, "61to70", "train"]]#,
    #                  ["m", "lside", 2002, "71to80", "train"]]
    #all_data_names = [#["m", "rside", 1904, "41to50", "train"],
    #                  ["m", "rside", 1973, "51to60", "train"],
    #                  ["m", "rside", 1940, "61to70", "train"]]#,
    #                  ["m", "rside", 1939, "71to80", "train"]]


    for gpsn in all_data_names:
        gender = gpsn[0]
        posture = gpsn[1]
        num_resting_poses = gpsn[2]
        subj_nums = gpsn[3]

        filename = 'slp_' + subj_nums + '_' + posture + '_' + gender + '_' + str(num_resting_poses) +'_filtered'

        resting_pose_data_list = np.load('/home/henry/data/02_resting_poses/slp_filtered/resting_pose_'+filename +'.npy', allow_pickle=True)

        training_database_pmat_height_list = list(np.load('/home/henry/data/03a_pmat_height/slp/pmat_height_'+ filename + '.npy', allow_pickle=True))

        #if  np.shape(training_database_pmat_height_list)[0] <= len(resting_pose_data_list):continue

        print len(resting_pose_data_list), np.shape(training_database_pmat_height_list)[0]

        list1_start = 0
        list1_end = np.shape(training_database_pmat_height_list)[0] - 100
        list2_start = np.shape(training_database_pmat_height_list)[0] - len(resting_pose_data_list) + np.shape(training_database_pmat_height_list)[0] - 100
        list2_end = np.shape(training_database_pmat_height_list)[0]

        #list1_end = 1900
        print(list1_start, list1_end, list2_start, list2_end)



        mv = list(np.load("/home/henry/data/04_resting_meshes/slp/"+filename+"/mv.npy", allow_pickle = True))
        mv = mv[list1_start:list1_end] + mv[list2_start:list2_end]
        np.save("/home/henry/data/04_resting_meshes/slp/"+filename+"/mv.npy", np.array(mv).astype(np.float32))

        mf = list(np.load("/home/henry/data/04_resting_meshes/slp/"+filename+"/mf.npy", allow_pickle = True))
        mf = mf[list1_start:list1_end] + mf[list2_start:list2_end]
        np.save("/home/henry/data/04_resting_meshes/slp/"+filename+"/mf.npy", np.array(mf).astype(np.int32))

        hv = list(np.load("/home/henry/data/04_resting_meshes/slp/"+filename+"/hv.npy", allow_pickle = True))
        hv = hv[list1_start:list1_end] + hv[list2_start:list2_end]
        np.save("/home/henry/data/04_resting_meshes/slp/"+filename+"/hv.npy", np.array(hv).astype(np.float32))

        hf = list(np.load("/home/henry/data/04_resting_meshes/slp/"+filename+"/hf.npy", allow_pickle = True))
        hf = hf[list1_start:list1_end] + hf[list2_start:list2_end]
        np.save("/home/henry/data/04_resting_meshes/slp/"+filename+"/hf.npy", np.array(hf).astype(np.int32))

        bv = list(np.load("/home/henry/data/04_resting_meshes/slp/"+filename+"/bv.npy", allow_pickle = True))
        bv = bv[list1_start:list1_end] + bv[list2_start:list2_end]
        np.save("/home/henry/data/04_resting_meshes/slp/"+filename+"/bv.npy", np.array(bv))

        bf = list(np.load("/home/henry/data/04_resting_meshes/slp/"+filename+"/bf.npy", allow_pickle = True))
        bf = bf[list1_start:list1_end] + bf[list2_start:list2_end]
        np.save("/home/henry/data/04_resting_meshes/slp/"+filename+"/bf.npy", np.array(bf))

        pmat = list(np.load("/home/henry/data/03a_pmat_height/slp/pmat_height_"+filename+".npy", allow_pickle = True))
        pmat = pmat[list1_start:list1_end] + pmat[list2_start:list2_end]
        np.save("/home/henry/data/03a_pmat_height/slp/pmat_height_"+filename+".npy", np.array(pmat))

        k = list(np.load("/home/henry/data/04_resting_meshes/slp/"+filename+"/k.npy", allow_pickle = True))
        k = k[list1_start:list1_end] + k[list2_start:list2_end]
        np.save("/home/henry/data/04_resting_meshes/slp/"+filename+"/k.npy", np.array(k))





def reprocess_synth_data():
    # fix_angles_in_dataset()

    #all_data_names = [#["f", "lay", 1944, "1to10", "train"],
    #                  ["f", "lay", 2210, "11to20", "train"],
    #                  ["f", "lay", 2201, "21to30", "train"],
    #                  ["f", "lay", 2194, "31to40", "train"]]
    #all_data_names = [#["f", "lside", 1857, "1to10", "train"],
    #                  ["f", "lside", 2087, "11to20", "train"],
    #                  ["f", "lside", 2086, "21to30", "train"],
    #                  ["f", "lside", 2106, "31to40", "train"]]
    #all_data_names = [#["f", "rside", 1805, "1to10", "train"],
    #                  ["f", "rside", 2001, "11to20", "train"],
    #                  ["f", "rside", 1922, "21to30", "train"],
    #                  ["f", "rside", 1949, "31to40", "train"]]
    #all_data_names = [#["m", "lay", 1946, "1to10", "train"],
    #                  ["m", "lay", 2192, "11to20", "train"]]
    #                  ["m", "lay", 2178, "21to30", "train"],
    #                  ["m", "lay", 2177, "31to40", "train"]]
    #all_data_names = [#["m", "lside", 1731, "1to10", "train"],
    #                  ["m", "lside", 2007, "11to20", "train"]]
    #                  ["m", "lside", 2002, "21to30", "train"],
    #                  ["m", "lside", 2021, "31to40", "train"]]
    all_data_names = [#["m", "rside", 1704, "1to10", "train"],
    #                  ["m", "rside", 1927, "11to20", "train"]]
    #                  ["m", "rside", 1844, "21to30", "train"],
                      ["m", "rside", 1902, "31to40", "train"]]

    #all_data_names = [#["f", "lay", 2198, "41to50", "train"],
                      #["f", "lay", 2197, "51to60", "train"],
    #                  ["f", "lay", 2213, "61to70", "train"],
    #                  ["f", "lay", 2184, "71to80", "train"]]
    #all_data_names = [#["f", "lside", 2091, "41to50", "train"],
                      #["f", "lside", 2053, "51to60", "train"],
                      #["f", "lside", 2014, "61to70", "train"]]#,
    #                  ["f", "lside", 2058, "71to80", "train"]]
    #all_data_names = [#["f", "rside", 1976, "41to50", "train"],
                      #["f", "rside", 2043, "51to60", "train"],
    #                  ["f", "rside", 1987, "61to70", "train"]]#,
    #                  ["f", "rside", 2010, "71to80", "train"]]
    #all_data_names = [#["m", "lay", 2195, "41to50", "train"],
    #                  ["m", "lay", 2199, "51to60", "train"],
    #                  ["m", "lay", 2203, "61to70", "train"]]#,
    #                  ["m", "lay", 2188, "71to80", "train"]]
    #all_data_names = [#["m", "lside", 2049, "41to50", "train"],
                      #["m", "lside", 1952, "51to60", "train"],
    #                  ["m", "lside", 1934, "61to70", "train"]]#,
    #                  ["m", "lside", 2002, "71to80", "train"]]
    #all_data_names = [#["m", "rside", 1904, "41to50", "train"],
    #                  ["m", "rside", 1973, "51to60", "train"],
    #                  ["m", "rside", 1940, "61to70", "train"]]#,
    #                  ["m", "rside", 1939, "71to80", "train"]] #messed up

    amount_to_add_ct = 1895

    num_data_points = 0

    training_data_dict = {}
    training_data_dict['markers_xyz_m'] = []
    training_data_dict['root_xyz_shift'] = []
    training_data_dict['joint_angles'] = []
    training_data_dict['body_shape'] = []
    training_data_dict['body_mass'] = []
    training_data_dict['body_height'] = []
    training_data_dict['bed_angle_deg'] = []
    training_data_dict['images'] = []


    for gpsn in all_data_names:
        gender = gpsn[0]
        posture = gpsn[1]
        num_resting_poses = gpsn[2]
        subj_nums = gpsn[3]
        dattype = gpsn[4]


        model_path = '/home/henry/git/SMPL_python_v.1.0.0/smpl/models/basicModel_' + gender + '_lbs_10_207_0_v1.0.0.pkl'
        m = load_model(model_path)



        resting_pose_data_list = np.load('/home/henry/data/02_resting_poses/slp_filtered/resting_pose_slp_'
                                          + subj_nums + '_' + posture + '_' + gender + '_' + str(num_resting_poses)
                                         + '_filtered.npy', allow_pickle=True)

        training_database_pmat_height_list = np.load('/home/henry/data/03a_pmat_height/slp/pmat_height_slp_'
                                          + subj_nums + '_' + posture + '_' + gender + '_' + str(num_resting_poses)
                                         + '_filtered.npy', allow_pickle=True)



        print len(resting_pose_data_list), np.shape(training_database_pmat_height_list)[0]
        print np.shape(training_database_pmat_height_list[0])

        for resting_pose_data_ct in range(len(resting_pose_data_list)):

            resting_pose_data_ct += amount_to_add_ct

            num_data_points += 1
            resting_pose_data = resting_pose_data_list[resting_pose_data_ct]
            pmat = training_database_pmat_height_list[resting_pose_data_ct]
            capsule_angles = resting_pose_data[0].tolist()
            root_joint_pos_list = resting_pose_data[1]
            body_shape_list = resting_pose_data[2]
            body_mass = resting_pose_data[3]

            #print(np.max(pmat))

            if math.isnan(np.max(pmat)): continue


            # print "shape", body_shape_list

            #print np.shape(resting_pose_data), np.shape(pmat), np.shape(capsule_angles), np.shape(
            #    root_joint_pos_list), np.shape(body_shape_list)

            for shape_param in range(10):
                m.betas[shape_param] = float(body_shape_list[shape_param])

            m.pose[:] = np.random.rand(m.pose.size) * 0.


            training_data_dict['body_mass'].append(body_mass)
            training_data_dict['body_height'].append(np.abs(np.min(m.r[:, 1]) - np.max(m.r[:, 1])))

            #print training_data_dict['body_mass'][-1] * 2.20462, 'MASS, lbs'
            #print training_data_dict['body_height'][-1] * 3.28084, 'HEIGHT, ft'

            m.pose[0:3] = capsule_angles[0:3]
            m.pose[3:6] = capsule_angles[6:9]
            m.pose[6:9] = capsule_angles[9:12]
            m.pose[9:12] = capsule_angles[12:15]
            m.pose[12:15] = capsule_angles[15:18]
            m.pose[15:18] = capsule_angles[18:21]
            m.pose[18:21] = capsule_angles[21:24]
            m.pose[21:24] = capsule_angles[24:27]
            m.pose[24:27] = capsule_angles[27:30]
            m.pose[27:30] = capsule_angles[30:33]
            m.pose[36:39] = capsule_angles[33:36]  # neck
            m.pose[39:42] = capsule_angles[36:39]
            m.pose[42:45] = capsule_angles[39:42]
            m.pose[45:48] = capsule_angles[42:45]  # head
            m.pose[48:51] = capsule_angles[45:48]
            m.pose[51:54] = capsule_angles[48:51]
            m.pose[54:57] = capsule_angles[51:54]
            m.pose[57:60] = capsule_angles[54:57]
            m.pose[60:63] = capsule_angles[57:60]
            m.pose[63:66] = capsule_angles[60:63]

            training_data_dict['joint_angles'].append(np.array(m.pose).astype(float))
            training_data_dict['body_shape'].append(np.array(m.betas).astype(float))
            # print "dict", training_data_dict['body_shape'][-1]

            # training_data_dict['v_template'].append(np.asarray(m.v_template))
            # training_data_dict['shapedirs'].append(np.asarray(m.shapedirs))

            # print np.sum(np.array(m.v_template))
            # print np.sum(np.array(m.shapedirs))
            # print np.sum(np.zeros((np.shape(np.array(m.J_regressor)))) + np.array(m.J_regressor))

            root_shift_x = root_joint_pos_list[0] + 0.374648 + 10 * INTER_SENSOR_DISTANCE
            root_shift_y = root_joint_pos_list[1] + 0.927099 + 10 * INTER_SENSOR_DISTANCE
            # root_shift_z = height
            root_shift_z = root_joint_pos_list[2] - 0.15
            #print root_shift_z

            x_positions = np.asarray(m.J_transformed)[:, 0] - np.asarray(m.J_transformed)[0, 0] + root_shift_x
            y_positions = np.asarray(m.J_transformed)[:, 1] - np.asarray(m.J_transformed)[0, 1] + root_shift_y
            z_positions = np.asarray(m.J_transformed)[:, 2] - np.asarray(m.J_transformed)[0, 2] + root_shift_z

            if resting_pose_data_ct == 0:
                print m.betas
                print m.pose
                print "J x trans", m.J_transformed[:, 0]

            xyz_positions = np.transpose([x_positions, y_positions, z_positions])
            xyz_positions_shape = np.shape(xyz_positions)
            xyz_positions = xyz_positions.reshape(xyz_positions_shape[0] * xyz_positions_shape[1])
            training_data_dict['markers_xyz_m'].append(xyz_positions)
            training_data_dict['root_xyz_shift'].append([root_shift_x, root_shift_y, root_shift_z])
            training_data_dict['images'].append(pmat.reshape(64 * 27))

            training_data_dict['bed_angle_deg'].append(0.)

            print set, resting_pose_data_ct, len(training_data_dict['images'])
            #if resting_pose_data_ct == 249: break
            #if len(training_data_dict['images']) == 1500: break

            visualize_pressure_map(training_data_dict['images'][-1].reshape(64,27), training_data_dict['markers_xyz_m'][-1], None, None, None)



        print training_data_dict['markers_xyz_m'][0]

        #print "RECHECKING!"
        #for entry in range(len(training_data_dict['markers_xyz_m'])):
            #print entry, training_data_dict['markers_xyz_m'][entry][0:2], training_data_dict['body_shape'][entry][0:2], \
            #training_data_dict['joint_angles'][entry][0:2]

    #pickle.dump(training_data_dict, open(os.path.join(
    #    '/home/henry/data/synth/random/train_' + gender + '_' + posture + '_' + str(num_data_points) + '_' + stiffness + '_stiff.p'), 'wb'))
    pickle.dump(training_data_dict, open(os.path.join(
        '/home/henry/git/sim_camera_resting_scene/data_BR/synth/slp/'+dattype+'_slp_'
        + posture + '_' + gender + '_41to50_' + str(len(training_data_dict['images'])) + '.p'), 'wb'))





def get_depth_cont_maps_from_synth():
    #all_data_names = [#["f", "lay", 8549, "1to40", "train"]]
    #                  ["f", "lside", 8136, "1to40", "train"],
    #                  ["f", "rside", 7677, "1to40", "train"],
    #                  ["m", "lay", 8493, "1to40", "train"],
    #                  ["m", "lside", 7761, "1to40", "train"],
    #                  ["m", "rside", 7377, "1to40", "train"]]

    #all_data_names = [["f", "lay", 6608, "41to70", "train"],
    #                  ["f", "lside", 6158, "41to70", "train"],
    #                  ["f", "rside", 6006, "41to70", "train"]]

    #all_data_names = [["m", "lay", 6597, "41to70", "train"],
    #                  ["m", "lside", 5935, "41to70", "train"],
    #                  ["m", "rside", 5817, "41to70", "train"]]

    all_data_names = [["f", "lay", 2184, "71to80", "train"],
                      ["f", "lside", 2058, "71to80", "train"],
                      ["f", "rside", 2010, "71to80", "train"],
                      ["m", "lay", 2188, "71to80", "train"],
                      ["m", "lside", 2002, "71to80", "train"],
                      ["m", "rside", 1939, "71to80", "train"]]



    from visualization_lib import VisualizationLib

    filler_taxels = []
    for i in range(27):
        for j in range(64):
            filler_taxels.append([i, j, 20000])
    filler_taxels = np.array(filler_taxels)


    for gpsn in all_data_names:
        gender = gpsn[0]
        posture = gpsn[1]
        num_resting_poses = gpsn[2]
        subj_nums = gpsn[3]
        dattype = gpsn[4]

        bed_angle = np.deg2rad(1.0)

        if posture == "sit":
            bed_angle = np.deg2rad(60.0)
        elif posture == "lay":
            bed_angle = np.deg2rad(1.0)

        # training_data_dict['v_template'] = []
        # training_data_dict['shapedirs'] = []

        model_path = '/home/henry/git/SMPL_python_v.1.0.0/smpl/models/basicModel_' + gender + '_lbs_10_207_0_v1.0.0.pkl'
        m = load_model(model_path)

        filename =  '/home/henry/git/sim_camera_resting_scene/data_BR/synth/slp/' + dattype + '_slp_'\
            + posture + '_' + gender + '_'+subj_nums+'_' + str(num_resting_poses) + '.p'
        #filename =  '/home/henry/data/synth/home_poses/home_pose_'+gender+'.p'





        training_data_dict = load_pickle(filename)
        print "loaded ", filename



        betas = training_data_dict['body_shape']
        pose = training_data_dict['joint_angles']
        images = training_data_dict['images']
        training_data_dict['mesh_depth'] = []
        training_data_dict['mesh_contact'] = []
        root_xyz_shift = training_data_dict['root_xyz_shift']

        ct = 0
        for index in range(len(betas)):
            #index += 4
            for beta_idx in range(10):
                m.betas[beta_idx] = betas[index][beta_idx]
            for pose_idx in range(72):
                m.pose[pose_idx] = pose[index][pose_idx]

            #print images[index]
            images[index][images[index] > 0] += 1
            #print images[index]
            training_data_dict['images'][index] = images[index].astype(int8) #convert the original pmat to an int to save space
            #print training_data_dict['images'][index]
            curr_root_shift = np.array(root_xyz_shift[index])

            #print curr_root_shift,'currroot'
            #print m.J_transformed, 'Jest'

            joints = np.array(m.J_transformed) + curr_root_shift + np.array([0.0, 0.0, -0.075]) - np.array(m.J_transformed)[0:1, :]
            vertices = np.array(m.r) + curr_root_shift + np.array([0.0, 0.0, -0.075]) - np.array(m.J_transformed)[0:1, :]
            vertices_rot = np.copy(vertices)

            #print vertices.shape
            #print vertices[0:10, :], 'verts'

            #print curr_root_shift, 'curr shift' #[0.59753822 1.36742909 0.09295963]


            #vertices[0, :] = np.array([0.0, 1.173, -5.0])

            bend_loc = 48 * 0.0286


            #import matplotlib.pyplot as plt
            #plt.plot(-vertices[:, 1], vertices[:, 2], 'r.')
            #print vertices.dtype
            #vertices = vertices.astype(float32)

            vertices_rot[:, 1] = vertices[:, 2]*np.sin(bed_angle) - (bend_loc - vertices[:, 1])*np.cos(bed_angle) + bend_loc
            vertices_rot[:, 2] = vertices[:, 2]*np.cos(bed_angle) + (bend_loc - vertices[:, 1])*np.sin(bed_angle)

            #vertices =
            vertices_rot = vertices_rot[vertices_rot[:, 1] >= bend_loc]
            vertices = np.concatenate((vertices[vertices[:, 1] < bend_loc], vertices_rot), axis = 0)
            #print vertices.shape

            #plt.plot(-vertices[:, 1], vertices[:, 2], 'k.')

            #plt.axis([-1.8, -0.2, -0.3, 1.0])
            #plt.show()

            #print vertices.shape

            joints_taxel = joints/0.0286
            vertices_taxel = vertices/0.0286
            vertices_taxel[:, 2] *= 1000
            vertices_taxel[:, 0] *= 1.04
            vertices_taxel[:, 0] -= 10
            vertices_taxel[:, 1] -= 10

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
            print time.time() - time_orig, ct

            training_data_dict['mesh_depth'].append(mesh_matrix)
            training_data_dict['mesh_contact'].append(contact_matrix)

            #print training_data_dict['images'][index].dtype
            #print training_data_dict['mesh_depth'][index].dtype
            #print training_data_dict['mesh_contact'][index].dtype



            #print m.J_transformed

            #print np.min(mesh_matrix), np.max(mesh_matrix)

            #VisualizationLib().visualize_pressure_map(pmat, joints, None, mesh_matrix+50, joints)
            #time.sleep(5)

            #break


        filename =  '/home/henry/git/sim_camera_resting_scene/data_BR/synth/slp/' + dattype + '_slp_'\
            + posture + '_' + gender + '_'+subj_nums+'_' + str(num_resting_poses) + '.p'

        #filename =  '/home/henry/data/synth/home_poses/home_pose_'+gender+'.p'

        pickle.dump(training_data_dict, open(os.path.join(filename), 'wb'))




if __name__ == "__main__":
    #fix_mv_mf_hv_hf_bv_bf_pmat_k()
    #reprocess_synth_data() #this converts resting_poses and pmat_height to a single file
    get_depth_cont_maps_from_synth() #this takes the above result and puts additional stuff in it
    #reduce_dataset_size()
