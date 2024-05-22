
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

txtfile = open("../FILEPATH.txt")
FILEPATH = txtfile.read().replace("\n", "")
txtfile.close()


import sys
sys.path.insert(0, '../lib_py')
sys.path.insert(-1,FILEPATH+'smpl/smpl_webuser3')
sys.path.insert(-1,FILEPATH)
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
print(sys.path)

import optparse

import lib_pyrender_depth as libPyRender
from visualization_lib_bp import VisualizationLib
from preprocessing_lib_bp import PreprocessingLib
from tensorprep_lib_bp import TensorPrepLib
from unpack_depth_batch_lib_bp import UnpackDepthBatchLib


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


import matplotlib.cm as cm #use cm.jet(list)

DATASET_CREATE_TYPE = 1


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


def compute_mpjpe():

    for item in dict_uncover: print(item)

    for item in ['total_allerror']:
        print(item, np.shape(dict_uncover[item]), np.mean(dict_uncover[item])*1000, 'MPJPE uncover')
        print(item, np.shape(dict_cover1[item]), np.mean(dict_cover1[item])*1000, 'MPJPE cover1')
        print(item, np.shape(dict_cover2[item]), np.mean(dict_cover2[item])*1000, 'MPJPE cover2')

def compute_v2v():


    smpl_verts_est_uncover = np.array(dict_uncover['smpl_verts_est']) - np.array(dict_uncover['smpl_verts_gt'])
    smpl_verts_est_uncover = smpl_verts_est_uncover.reshape(smpl_verts_est_uncover.shape[0]*smpl_verts_est_uncover.shape[1], 3)
    smpl_verts_est_uncover_err = np.linalg.norm(smpl_verts_est_uncover, axis=1)
    print(np.shape(smpl_verts_est_uncover_err), np.mean(smpl_verts_est_uncover_err)*1000, 'v2v uncover')


    smpl_verts_est_cover1 = np.array(dict_cover1['smpl_verts_est']) - np.array(dict_cover1['smpl_verts_gt'])
    smpl_verts_est_cover1 = smpl_verts_est_cover1.reshape(smpl_verts_est_cover1.shape[0]*smpl_verts_est_cover1.shape[1], 3)
    smpl_verts_est_cover1_err = np.linalg.norm(smpl_verts_est_cover1, axis=1)
    print(np.shape(smpl_verts_est_cover1_err), np.mean(smpl_verts_est_cover1_err)*1000, 'v2v cover1')


    smpl_verts_est_cover2 = np.array(dict_cover2['smpl_verts_est']) - np.array(dict_cover2['smpl_verts_gt'])
    smpl_verts_est_cover2 = smpl_verts_est_cover2.reshape(smpl_verts_est_cover2.shape[0]*smpl_verts_est_cover2.shape[1], 3)
    smpl_verts_est_cover2_err = np.linalg.norm(smpl_verts_est_cover2, axis=1)
    print(np.shape(smpl_verts_est_cover2_err), np.mean(smpl_verts_est_cover2_err)*1000, 'v2v cover2')





def compute_htwt_err():
    weight_kg_gt_uncover = np.array(dict_uncover['wt_gt_est_ht_gt_est_list'])[:, 0]
    weight_kg_est_uncover = np.array(dict_uncover['wt_gt_est_ht_gt_est_list'])[:, 1]
    #print(np.mean(np.square(weight_kg_est_uncover-weight_kg_gt_uncover)), 'weight err kg uncover MSE')
    print(np.mean(np.abs(weight_kg_est_uncover-weight_kg_gt_uncover)), 'weight err kg uncover MAE')


    weight_kg_gt_cover1 = np.array(dict_cover1['wt_gt_est_ht_gt_est_list'])[:, 0]
    weight_kg_est_cover1 = np.array(dict_cover1['wt_gt_est_ht_gt_est_list'])[:, 1]
    #print(np.mean(np.square(weight_kg_est_cover1-weight_kg_gt_cover1)), 'weight err kg cover1 MSE')
    print(np.mean(np.abs(weight_kg_est_cover1-weight_kg_gt_cover1)), 'weight err kg cover1 MAE')

    weight_kg_gt_cover2 = np.array(dict_cover2['wt_gt_est_ht_gt_est_list'])[:, 0]
    weight_kg_est_cover2 = np.array(dict_cover2['wt_gt_est_ht_gt_est_list'])[:, 1]
    #print(np.mean(np.square(weight_kg_est_cover2-weight_kg_gt_cover2)), 'weight err kg cover2 MSE')
    print(np.mean(np.abs(weight_kg_est_cover2-weight_kg_gt_cover2)), 'weight err kg cover2 MAE')



    height_m_gt_uncover = np.array(dict_uncover['wt_gt_est_ht_gt_est_list'])[:, 2]
    height_m_est_uncover = np.array(dict_uncover['wt_gt_est_ht_gt_est_list'])[:, 3]
    #print(np.mean(np.square(height_m_est_uncover-height_m_gt_uncover))*1000, 'height err m uncover MSE')
    print(np.mean(np.abs(height_m_est_uncover-height_m_gt_uncover))*1000, 'height err m uncover MAE')

    height_m_gt_cover1 = np.array(dict_cover1['wt_gt_est_ht_gt_est_list'])[:, 2]
    height_m_est_cover1 = np.array(dict_cover1['wt_gt_est_ht_gt_est_list'])[:, 3]
    #print(np.mean(np.square(height_m_est_cover1-height_m_gt_cover1))*1000, 'height err m cover1 MSE')
    print(np.mean(np.abs(height_m_est_cover1-height_m_gt_cover1))*1000, 'height err m cover1 MAE')

    height_m_gt_cover2 = np.array(dict_cover2['wt_gt_est_ht_gt_est_list'])[:, 2]
    height_m_est_cover2 = np.array(dict_cover2['wt_gt_est_ht_gt_est_list'])[:, 3]
    #print(np.mean(np.square(height_m_est_cover2-height_m_gt_cover2))*1000, 'height err m cover2 MSE')
    print(np.mean(np.abs(height_m_est_cover2-height_m_gt_cover2))*1000, 'height err m cover2 MAE')




def compute_pimg_err():
    for item in ['pmat_gt_est_list']:
        print(item, np.shape(dict_uncover[item]), 'pressure img uncover')
        print(np.mean(np.square(np.array(dict_uncover[item])[:, 0, :, :]-np.array(dict_uncover[item])[:, 1, :, :])), 'MSE')
        print(np.mean(np.abs(np.array(dict_uncover[item])[:, 0, :, :]-np.array(dict_uncover[item])[:, 1, :, :])), 'MAE')
        #print(np.mean(np.square(np.array(dict_uncover[item])[:, 0, 1:, 1:]-np.array(dict_uncover[item])[:, 0, :-1, :-1])), 'MSE')
        print(np.mean(np.square(np.array(dict_uncover[item])[:, 0, :, :]-np.array(dict_uncover[item])[0, 1, :, :])), 'MSE from home pose')
        print(np.mean(np.abs(np.array(dict_uncover[item])[:, 0, :, :]-np.array(dict_uncover[item])[0, 1, :, :])), 'MAE home pose')
        #print(np.mean(np.square(np.array(dict_uncover[item])[:, 0, :, :])), 'from zero')
        print(item, np.shape(dict_cover1[item]), 'pressure img cover1')
        print(np.mean(np.square(np.array(dict_cover1[item])[:, 0, :, :]-np.array(dict_cover1[item])[:, 1, :, :])), 'MSE')
        print(np.mean(np.abs(np.array(dict_cover1[item])[:, 0, :, :]-np.array(dict_cover1[item])[:, 1, :, :])), 'MAE')
        print(item, np.shape(dict_cover2[item]), 'pressure img cover2')
        print(np.mean(np.square(np.array(dict_cover2[item])[:, 0, :, :]-np.array(dict_cover2[item])[:, 1, :, :])), 'MSE')
        print(np.mean(np.abs(np.array(dict_cover2[item])[:, 0, :, :]-np.array(dict_cover2[item])[:, 1, :, :])), 'MAE')



def compute_v2vP_err():
    dict_uncover_list_v2vP_err_abs = []
    dict_uncover_list_v2vP_err_sq = []

    set_all = [[], [], [], [], [], [], [], [], [], [], []]

    for i in range(81, 103):
        some_subject = '%05d'% (i)

        #try:
        #dict_uncover_curr = load_pickle(FILEPATH+'data_BP/results/3D_quantitative_PAMI_v2vP/anglesDC_97495ct_128b_x1pm_depthestin_angleadj_rgangs_lb_lv2v_slpb_dpns_rt_dou_40e_uncover/results_slp_3D_'+some_subject+'.p')
        dict_uncover_curr = load_pickle(FILEPATH+'data_BP/results/3D_quantitative_PAMI_v2vP/anglesDC_10665ct_128b_x1pm_0.5rtojtdpth_depthestin_angleadj_rgangs_lb_lv2v_slpb_dpns_rt_40e_uncover/results_slp_3D_'+some_subject+'.p')
        #except: continue

        sq_err = np.ndarray.flatten(np.array(dict_uncover_curr['vertex_pressure_list_sq_err']))
        gt_press = np.ndarray.flatten(np.array(dict_uncover_curr['vertex_pressure_list_GT']))



        for vert_idx in range(45*6890):
            if 0<= gt_press[vert_idx] <10:
                set_all[0].append(sq_err[vert_idx])
            elif 10<= gt_press[vert_idx] <20:
                set_all[1].append(sq_err[vert_idx])
            elif 20<= gt_press[vert_idx] <30:
                set_all[2].append(sq_err[vert_idx])
            elif 30<= gt_press[vert_idx] <40:
                set_all[3].append(sq_err[vert_idx])
            elif 40<= gt_press[vert_idx] <50:
                set_all[4].append(sq_err[vert_idx])
            elif 50<= gt_press[vert_idx] <60:
                set_all[5].append(sq_err[vert_idx])
            elif 60<= gt_press[vert_idx] <70:
                set_all[6].append(sq_err[vert_idx])
            elif 70<= gt_press[vert_idx] <80:
                set_all[7].append(sq_err[vert_idx])
            elif 80<= gt_press[vert_idx] <90:
                set_all[8].append(sq_err[vert_idx])
            elif 90<= gt_press[vert_idx] <100:
                set_all[9].append(sq_err[vert_idx])
            elif 100<= gt_press[vert_idx]:
                set_all[10].append(sq_err[vert_idx])



        print(some_subject, np.shape(dict_uncover_curr['vertex_pressure_list_abs_err']), np.mean(dict_uncover_curr['vertex_pressure_list_abs_err']), np.mean(dict_uncover_curr['vertex_pressure_list_sq_err']))

        dict_uncover_list_v2vP_err_abs.append(dict_uncover_curr['vertex_pressure_list_abs_err'])
        dict_uncover_list_v2vP_err_sq.append(dict_uncover_curr['vertex_pressure_list_sq_err'])

    print(np.mean(dict_uncover_list_v2vP_err_abs), np.mean(dict_uncover_list_v2vP_err_sq))
    for i in range(0, 11):
        print('       ', i*10, np.mean(set_all[i]))




if __name__ ==  "__main__":

    dict_uncover = load_pickle(FILEPATH+'data_BP/results/3D_quantitative_PAMI/quant_anglesDC_10665ct_128b_x1pm_0.5rtojtdpth_depthestin_angleadj_rgangs_lb_lv2v_slpb_dpns_rt_40e_81to102_slp_uncover.p')
    dict_cover1 = load_pickle(FILEPATH+'data_BP/results/3D_quantitative_PAMI/quant_anglesDC_10665ct_128b_x1pm_0.5rtojtdpth_depthestin_angleadj_rgangs_lb_lv2v_slpb_dpns_rt_40e_81to102_slp_cover1.p')
    dict_cover2 = load_pickle(FILEPATH+'data_BP/results/3D_quantitative_PAMI/quant_anglesDC_10665ct_128b_x1pm_0.5rtojtdpth_depthestin_angleadj_rgangs_lb_lv2v_slpb_dpns_rt_40e_81to102_slp_cover2.p')


    compute_mpjpe()
    compute_v2v()
    compute_htwt_err()
    compute_pimg_err()

    compute_v2vP_err()