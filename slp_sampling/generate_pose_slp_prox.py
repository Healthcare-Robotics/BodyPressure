
import numpy as np
import random
import copy
import trimesh
import pyrender
from smpl.smpl_webuser.serialization import load_model




from util import batch_global_rigid_transformation, batch_rodrigues, batch_lrotmin, reflect_pose


import sys
sys.path.insert(0, '../smplify_public_hc/code/lib')

#volumetric pose gen libraries
import lib_kinematics as libKinematics
#import lib_render as libRender
import dart_skel_sim_slp as dart_skel_sim
from time import sleep

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

#ROS
#import rospy
#import tf
DATASET_CREATE_TYPE =5

import cv2

import math
from random import shuffle
import torch
import torch.nn as nn

import tensorflow as tensorflow
import cPickle as pickle


#MISC
import time as time
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

#hmr
from hmr.src.tf_smpl.batch_smpl import SMPL



class GeneratePose():
    def __init__(self, gender, filepath_prefix = '/home/henry'):
        ## Load SMPL model (here we load the female model)
        self.filepath_prefix = filepath_prefix

        if gender == "m":
            model_path = filepath_prefix+'/git/SMPL_python_v.1.0.0/smpl/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl'
        else:
            model_path = filepath_prefix+'/git/SMPL_python_v.1.0.0/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'

        self.reset_pose = False
        self.m = load_model(model_path)
        self.scene = pyrender.Scene()
        self.first_pass = True



    def sample_body_shape(self, sampling, sigma, one_side_range):
        mu = 0
        for i in range(10):
            if sampling == "NORMAL":
                self.m.betas[i] = random.normalvariate(mu, sigma)
            elif sampling == "UNIFORM":
                self.m.betas[i]  = np.random.uniform(-one_side_range, one_side_range)



    def get_noisy_angle(self, angle, angle_min, angle_max):
        not_within_bounds = True
        mu = 0
        counter = 0
        #sigma = np.pi/16
        sigma = np.pi/12

        #print "angle to make noisy", angle, angle_min, angle_max,
        while not_within_bounds == True:

            noisy_angle = angle + random.normalvariate(mu, sigma)
            if noisy_angle > angle_min and noisy_angle < angle_max:
                #print "angle, min, max", noisy_angle, angle_min, angle_max
                not_within_bounds = False
            else:
                print "angle, min, max", noisy_angle, angle_min, angle_max
                counter += 1
                if counter > 10:
                    self.reset_pose = True
                    break
                pass

        #print "  noisy angle", noisy_angle
        return noisy_angle

    def get_noisy_angle_hard_limit(self, angle, angle_min, angle_max):
        not_within_bounds = True
        mu = 0
        counter = 0
        #sigma = np.pi/16
        sigma = np.pi/12

        noisy_angle = angle + random.normalvariate(mu, sigma)
        if noisy_angle < angle_min:
            noisy_angle = angle_min*1.
        elif noisy_angle > angle_max:
            noisy_angle = angle_max*1.

        #print "  noisy angle", noisy_angle
        return noisy_angle


    def check_for_invalid_poses_resave(self):
        verbose = False

        set_criteria_list = [["f", "lay", "1to10", "1966"],
                             ["f", "lay", "11to20", "2223"],
                             ["f", "lay", "21to30", "2227"],
                             ["f", "lay", "31to40", "2217"],
                             ["f", "lay", "41to50", "2220"],
                             ["f", "lay", "51to60", "2224"],
                             ["f", "lay", "61to70", "2229"],
                             ["f", "lay", "71to80", "2213"],
                             ["m", "lay", "1to10", "1972"],
                             ["m", "lay", "11to20", "2205"],
                             ["m", "lay", "21to30", "2195"],
                             ["m", "lay", "31to40", "2216"],
                             ["m", "lay", "41to50", "2220"],
                             ["m", "lay", "51to60", "2217"],
                             ["m", "lay", "61to70", "2220"],
                             ["m", "lay", "71to80", "2215"],
                             ["f", "lside", "1to10", "1910"],
                             ["f", "lside", "11to20", "2135"],
                             ["f", "lside", "21to30", "2140"],
                             ["f", "lside", "31to40", "2162"],
                             ["f", "lside", "41to50", "2160"],
                             ["f", "lside", "51to60", "2112"],
                             ["f", "lside", "61to70", "2091"],
                             ["f", "lside", "71to80", "2143"],
                             ["m", "lside", "1to10", "1800"],
                             ["m", "lside", "11to20", "2067"],
                             ["m", "lside", "21to30", "2046"],
                             ["m", "lside", "31to40", "2069"],
                             ["m", "lside", "41to50", "2109"],
                             ["m", "lside", "51to60", "2007"],
                             ["m", "lside", "61to70", "2013"],
                             ["m", "lside", "71to80", "2084"],
                             ["f", "rside", "1to10", "1889"],
                             ["f", "rside", "11to20", "2083"],
                             ["f", "rside", "21to30", "1989"],
                             ["f", "rside", "31to40", "2045"],
                             ["f", "rside", "41to50", "2049"],
                             ["f", "rside", "51to60", "2097"],
                             ["f", "rside", "61to70", "2079"],
                             ["f", "rside", "71to80", "2092"],
                             ["m", "rside", "1to10", "1791"],
                             ["m", "rside", "11to20", "2008"],
                             ["m", "rside", "21to30", "1916"],
                             ["m", "rside", "31to40", "1968"],
                             ["m", "rside", "41to50", "1974"],
                             ["m", "rside", "51to60", "2023"],
                             ["m", "rside", "61to70", "2017"],
                             ["m", "rside", "71to80", "2012"],
                             ]



        total_flagged_poses = 0
        total_valid_poses = 0
        reset_angles = True

        for set_criteria in set_criteria_list:
            filename = "slp_"+set_criteria[2]+"_"+set_criteria[1]+"_"+set_criteria[0]
            resting_pose_data_list = np.load("/home/henry/data/02_resting_poses/slp/resting_pose_"+filename+"_"+set_criteria[3]+".npy", allow_pickle = True)
            resting_pose_data_list_filtered = []
            flagged_poses_in_file = 0
            valid_poses_in_file = 0
            buffer_angle = 15.

            for idx in range(len(resting_pose_data_list)):

                flag_pose = False
                capsule_angles = list(np.array(resting_pose_data_list[idx][0]).astype(float))
               # print(resting_pose_data_list[idx][0])
                #hipL = capsule_angles[6:9]*180/np.pi


                if reset_angles == True:
                    hipL = np.array(libKinematics.dir_cos_angles_from_matrix(libKinematics.matrix_from_dir_cos_angles(capsule_angles[6:9])))*180/np.pi
                    resting_pose_data_list[idx][0][6:9] =  list(hipL*np.pi/180.)
                else: hipL = np.array(capsule_angles[6:9])*180/np.pi
                if hipL[0] < -140-buffer_angle or hipL[0] > 10+buffer_angle:
                    if verbose == True: print(idx, hipL[0], "left hip 0 out of range")
                    flag_pose = True
                if hipL[1] < -60-buffer_angle or hipL[1] > 120+buffer_angle:
                    if verbose == True: print(idx, hipL[1], "left hip 1 out of range")
                    flag_pose = True
                if hipL[2] < -60-buffer_angle or hipL[2] > 60+buffer_angle:
                    if verbose == True: print(idx, hipL[2], "left hip 2 out of range")
                    flag_pose = True

                #hipR = capsule_angles[9:12]*180/np.pi
                if reset_angles == True:
                    hipR = np.array(libKinematics.dir_cos_angles_from_matrix(libKinematics.matrix_from_dir_cos_angles(capsule_angles[9:12])))*180/np.pi
                    resting_pose_data_list[idx][0][9:12] =  list(hipR*np.pi/180.)
                else: hipR = np.array(capsule_angles[9:12])*180/np.pi
                if hipR[0] < -140-buffer_angle or hipR[0] > 10.+buffer_angle:
                    if verbose == True: print(idx, hipR[0], "right hip 0 out of range")
                    flag_pose = True
                if hipR[1] < -120-buffer_angle or hipR[1] > 60+buffer_angle:
                    if verbose == True: print(idx, hipR[1], "right hip 1 out of range")
                    flag_pose = True
                if hipR[2] < -60-buffer_angle or hipR[2] > 60+buffer_angle:
                    if verbose == True: print(idx, hipR[2], "right hip 2 out of range")
                    flag_pose = True

                if reset_angles == True:
                    kneeL = np.array(libKinematics.dir_cos_angles_from_matrix(libKinematics.matrix_from_dir_cos_angles(capsule_angles[12:15])))*180/np.pi
                    resting_pose_data_list[idx][0][12:15] =  list(kneeL*np.pi/180.)
                else: kneeL = np.array(libKinematics.dir_cos_angles_from_matrix(libKinematics.matrix_from_dir_cos_angles(capsule_angles[12:15])))*180/np.pi
                if kneeL[0] < -1-buffer_angle or kneeL[0] > 170.:
                    if verbose == True: print(idx, kneeL[0], "left knee 0 out of range")
                    flag_pose = True
                #print(kneeL)

                if reset_angles == True:
                    kneeR = np.array(libKinematics.dir_cos_angles_from_matrix(libKinematics.matrix_from_dir_cos_angles(capsule_angles[15:18])))*180/np.pi
                    resting_pose_data_list[idx][0][15:18] =  list(kneeR*np.pi/180.)
                else: kneeR = np.array(capsule_angles[15:18])*180/np.pi
                if kneeR[0] < -1-buffer_angle or kneeR[0] > 170.:
                    if verbose == True: print(idx, kneeR[0], "right knee 0 out of range")
                    flag_pose = True

                if reset_angles == True:
                    shdinL = np.array(libKinematics.dir_cos_angles_from_matrix(libKinematics.matrix_from_dir_cos_angles(capsule_angles[36:39])))*180/np.pi
                    resting_pose_data_list[idx][0][36:39] =  list(shdinL*np.pi/180.)
                else: shdinL = np.array(capsule_angles[36:39])*180/np.pi
                if shdinL[0] < -40-buffer_angle or shdinL[0] > 40+buffer_angle:
                    if verbose == True: print(idx, shdinL[0], "left shdin 0 out of range")
                    flag_pose = True
                if shdinL[1] < -50-buffer_angle or shdinL[1] > 20+buffer_angle:
                    if verbose == True: print(idx, shdinL[1], "left shdin 1 out of range")
                    flag_pose = True
                if shdinL[2] < -40-buffer_angle or shdinL[2] > 40+buffer_angle:
                    if verbose == True: print(idx, shdinL[2], "left shdin 2 out of range")
                    flag_pose = True


                if reset_angles == True:
                    shdinR = np.array(libKinematics.dir_cos_angles_from_matrix(libKinematics.matrix_from_dir_cos_angles(capsule_angles[39:42])))*180/np.pi
                    resting_pose_data_list[idx][0][39:42] =  list(shdinR*np.pi/180.)
                else: shdinR = np.array(capsule_angles[39:42])*180/np.pi
                if shdinR[0] < -40-buffer_angle or shdinR[0] > 40+buffer_angle:
                    if verbose == True: print(idx, shdinR[0], "right shdin 0 out of range")
                    flag_pose = True
                if shdinR[1] < -20-buffer_angle or shdinR[1] > 50+buffer_angle:
                    if verbose == True: print(idx, shdinR[1], "right shdin 1 out of range")
                    flag_pose = True
                if shdinR[2] < -40-buffer_angle or shdinR[2] > 40+buffer_angle:
                    if verbose == True: print(idx, shdinR[2], "right shdin 2 out of range")
                    flag_pose = True


                if reset_angles == True:
                    shdoutL = np.array(libKinematics.dir_cos_angles_from_matrix(libKinematics.matrix_from_dir_cos_angles(capsule_angles[45:48])))*180/np.pi
                    resting_pose_data_list[idx][0][45:48] =  list(shdoutL*np.pi/180.)
                else: shdoutL = np.array(capsule_angles[45:48])*180/np.pi
                if shdoutL[0] < -70-buffer_angle or shdoutL[0] > 70+buffer_angle:
                    if verbose == True: print(idx, shdoutL[0], "left shdout 0 out of range")
                    flag_pose = True
                if shdoutL[1] < -90-buffer_angle or shdoutL[1] > 35+buffer_angle:
                    if verbose == True: print(idx, shdoutL[1], "left shdout 1 out of range")
                    flag_pose = True
                if shdoutL[2] < -90-buffer_angle or shdoutL[2] > 60+buffer_angle:
                    if verbose == True: print(idx, shdoutL[2], "left shdout 2 out of range")
                    flag_pose = True

                if reset_angles == True:
                    shdoutR = np.array(libKinematics.dir_cos_angles_from_matrix(libKinematics.matrix_from_dir_cos_angles(capsule_angles[48:51])))*180/np.pi
                    resting_pose_data_list[idx][0][48:51] =  list(shdoutR*np.pi/180.)
                else: shdoutR = np.array(capsule_angles[48:51])*180/np.pi
                if shdoutR[0] < -70-buffer_angle or shdoutR[0] > 70+buffer_angle:
                    if verbose == True: print(idx, shdoutR[0], "right shdout 0 out of range")
                    flag_pose = True
                if shdoutR[1] < -35-buffer_angle or shdoutR[1] > 90+buffer_angle:
                    if verbose == True: print(idx, shdoutR[1], "right shdout 1 out of range")
                    flag_pose = True
                if shdoutR[2] < -60-buffer_angle or shdoutR[2] > 90+buffer_angle:
                    if verbose == True: print(idx, shdoutR[2], "right shdout 2 out of range")
                    flag_pose = True

                if reset_angles == True:
                    elbowL = np.array(libKinematics.dir_cos_angles_from_matrix(libKinematics.matrix_from_dir_cos_angles(capsule_angles[51:54])))*180/np.pi
                    resting_pose_data_list[idx][0][51:54] =  list(elbowL*np.pi/180.)
                else: elbowL = np.array(libKinematics.dir_cos_angles_from_matrix(libKinematics.matrix_from_dir_cos_angles(capsule_angles[51:54])))*180/np.pi
                if elbowL[1] < -170 or elbowL[1] > 1.+buffer_angle:
                    if verbose == True: print(idx, elbowL[1], "left elbow 1 out of range")
                    flag_pose = True

                if reset_angles == True:
                    elbowR = np.array(libKinematics.dir_cos_angles_from_matrix(libKinematics.matrix_from_dir_cos_angles(capsule_angles[54:57])))*180/np.pi
                    resting_pose_data_list[idx][0][54:57] = list(elbowR*np.pi/180.)
                else: elbowR = np.array(capsule_angles[54:57])*180/np.pi
                if elbowR[1] < -1-buffer_angle or elbowR[1] > 170.:
                    if verbose == True: print(idx, elbowR[1], "right elbow 1 out of range")
                    flag_pose = True



                #resting_pose_data_list[idx][0] = resting_pose_data_list[idx][0]
                #if np.sum(np.array(resting_pose_data_list[idx][0]) - np.array(capsule_angles)) > 0.01:
                #    #print(idx, capsule_angles, 'capsule angles before filter')
                #    #print(resting_pose_data_list[idx][0], 'capsule angles after filter')
                #    print(idx, np.array(resting_pose_data_list[idx][0]) - np.array(capsule_angles))


                #print(idx)
                if flag_pose == True:
                    flagged_poses_in_file += 1
                    total_flagged_poses += 1
                else:
                    valid_poses_in_file += 1
                    total_valid_poses += 1
                    resting_pose_data_list_filtered.append(resting_pose_data_list[idx])

            print(set_criteria, "    flagged poses in file:", flagged_poses_in_file, "   valid poses in file:", valid_poses_in_file, len(resting_pose_data_list_filtered))
            np.save("/home/henry/data/02_resting_poses/slp_filtered/resting_pose_"+filename+"_"+str(len(resting_pose_data_list_filtered))+"_filtered.npy", np.array(resting_pose_data_list_filtered))

        print("total flagged poses:", total_flagged_poses, "   total valid poses:", total_valid_poses)



    def map_slp_to_rand_angles(self, original_pose, alter_angles = True):
        angle_type = "angle_axis"
        #angle_type = "euler"

        dircos_limit = {}
        dircos_limit['hip0_L'] = -140.*np.pi/180.
        dircos_limit['hip0_U'] = 10.*np.pi/180.
        dircos_limit['hip1_L'] = -60.*np.pi/180.
        dircos_limit['hip1_U'] = 120.*np.pi/180.
        dircos_limit['hip2_L'] = -60.*np.pi/180.
        dircos_limit['hip2_U'] = 60.*np.pi/180.
        dircos_limit['knee_L'] = -1.*np.pi/180.
        dircos_limit['knee_U'] = 170.*np.pi/180.

        dircos_limit['shd00_L'] = -40.*np.pi/180.
        dircos_limit['shd00_U'] = 40.*np.pi/180.
        dircos_limit['shd01_L'] = -50.*np.pi/180.
        dircos_limit['shd01_U'] = 20.*np.pi/180.
        dircos_limit['shd02_L'] = -40.*np.pi/180.
        dircos_limit['shd02_U'] = 40.*np.pi/180.

        dircos_limit['shd10_L'] = -70.*np.pi/180.
        dircos_limit['shd10_U'] = 70.*np.pi/180.
        dircos_limit['shd11_L'] = -90.*np.pi/180.
        dircos_limit['shd11_U'] = 35.*np.pi/180.
        dircos_limit['shd12_L'] = -90.*np.pi/180.
        dircos_limit['shd12_U'] = 60.*np.pi/180.

        dircos_limit['elbow_L'] = -170.*np.pi/180.
        dircos_limit['elbow_U'] = 1.*np.pi/180.

        print('alter angs',alter_angles)

        while True:
            self.reset_pose = False

            R_root = libKinematics.matrix_from_dir_cos_angles(original_pose[0:3])

            flip_root_euler = np.pi
            flip_root_R = libKinematics.eulerAnglesToRotationMatrix([flip_root_euler, 0.0, 0.0])

            root_rot_rand_R = libKinematics.eulerAnglesToRotationMatrix([0.0,  random.normalvariate(0.0, np.pi/12),  random.normalvariate(0.0, np.pi/12)]) #randomize the root rotation
            #root_rot_rand_R = libKinematics.eulerAnglesToRotationMatrix([0.0,  0.0, 0.0]) #randomize the root rotation

            dir_cos_root = libKinematics.dir_cos_angles_from_matrix(np.matmul(root_rot_rand_R, np.matmul(R_root, flip_root_R)))


            #R_root2 = libKinematics.matrix_from_dir_cos_angles([original_pose[0]-4*np.pi, original_pose[1], original_pose[2]])
            #dir_cos_root2 = libKinematics.dir_cos_angles_from_matrix(R_root2)
            #print('eulers2', libKinematics.rotationMatrixToEulerAngles(R_root2))


            self.m.pose[0] = dir_cos_root[0]
            self.m.pose[1] = -dir_cos_root[1]
            self.m.pose[2] = dir_cos_root[2]

            #print(original_pose[0:3])
            #print(dir_cos_root)
            #print(dir_cos_root2)




            self.m.pose[3] = generator.get_noisy_angle_hard_limit(original_pose[3], dircos_limit['hip0_L'], dircos_limit['hip0_U'])
            self.m.pose[4] = generator.get_noisy_angle_hard_limit(original_pose[4], dircos_limit['hip1_L'], dircos_limit['hip1_U'])
            self.m.pose[5] = generator.get_noisy_angle_hard_limit(original_pose[5], dircos_limit['hip2_L'], dircos_limit['hip2_U'])
            self.m.pose[12] = generator.get_noisy_angle_hard_limit(original_pose[12], dircos_limit['knee_L'], dircos_limit['knee_U'])
            self.m.pose[13] = original_pose[13]
            self.m.pose[14] = original_pose[14]



            self.m.pose[6] = generator.get_noisy_angle_hard_limit(original_pose[6], dircos_limit['hip0_L'], dircos_limit['hip0_U'])
            self.m.pose[7] = generator.get_noisy_angle_hard_limit(original_pose[7], -dircos_limit['hip1_U'], -dircos_limit['hip1_L'])
            self.m.pose[8] = generator.get_noisy_angle_hard_limit(original_pose[8], -dircos_limit['hip2_U'], -dircos_limit['hip2_L'])
            self.m.pose[15] = generator.get_noisy_angle_hard_limit(original_pose[15], dircos_limit['knee_L'], dircos_limit['knee_U'])
            self.m.pose[16] = original_pose[16]
            self.m.pose[17] = original_pose[17]



            self.m.pose[9] = original_pose[9] #stomach
            self.m.pose[10] = original_pose[10] #stomach
            self.m.pose[11] = original_pose[11] #stomach


            self.m.pose[18] = original_pose[18]#chest
            self.m.pose[19] = original_pose[19]#chest
            self.m.pose[20] = original_pose[20]#chest
            self.m.pose[21] = original_pose[21]#l ankle
            self.m.pose[22] = original_pose[22]#l ankle
            self.m.pose[23] = original_pose[23]#l ankle
            self.m.pose[24] = original_pose[24]#r ankle
            self.m.pose[25] = original_pose[25]#r ankle
            self.m.pose[26] = original_pose[26]#r ankle
            self.m.pose[27] = original_pose[27]#sternum
            self.m.pose[28] = original_pose[28]#sternum
            self.m.pose[29] = original_pose[29]#stermum
            self.m.pose[30] = original_pose[30]#l foot
            self.m.pose[31] = original_pose[31]#l foot
            self.m.pose[32] = original_pose[32]#l foot
            self.m.pose[33] = original_pose[33]#r foot
            self.m.pose[34] = original_pose[34]#r foot
            self.m.pose[35] = original_pose[35]#r foot
            self.m.pose[36] = original_pose[36]#neck
            self.m.pose[37] = original_pose[37]#neck
            self.m.pose[38] = original_pose[38]#neck

            self.m.pose[45] = original_pose[45]#head
            self.m.pose[46] = original_pose[46]#head
            self.m.pose[47] = original_pose[47]#head


            self.m.pose[39] = generator.get_noisy_angle_hard_limit(original_pose[39], dircos_limit['shd00_L'], dircos_limit['shd00_U'])
            self.m.pose[40] = generator.get_noisy_angle_hard_limit(original_pose[40], dircos_limit['shd01_L'], dircos_limit['shd01_U'])
            self.m.pose[41] = generator.get_noisy_angle_hard_limit(original_pose[41], dircos_limit['shd02_L'], dircos_limit['shd02_U'])
            self.m.pose[42] = generator.get_noisy_angle_hard_limit(original_pose[42], dircos_limit['shd00_L'], dircos_limit['shd00_U'])
            self.m.pose[43] = generator.get_noisy_angle_hard_limit(original_pose[43], -dircos_limit['shd01_U'], -dircos_limit['shd01_L'])
            self.m.pose[44] = generator.get_noisy_angle_hard_limit(original_pose[44], -dircos_limit['shd02_U'], -dircos_limit['shd02_L'])
            self.m.pose[54] = original_pose[54]
            self.m.pose[55] = generator.get_noisy_angle_hard_limit(original_pose[55], dircos_limit['elbow_L'], dircos_limit['elbow_U'])
            self.m.pose[56] = original_pose[56]


            self.m.pose[48] = generator.get_noisy_angle_hard_limit(original_pose[48], dircos_limit['shd10_L'], dircos_limit['shd10_U'])
            self.m.pose[49] = generator.get_noisy_angle_hard_limit(original_pose[49], dircos_limit['shd11_L'], dircos_limit['shd11_U'])
            self.m.pose[50] = generator.get_noisy_angle_hard_limit(original_pose[50], dircos_limit['shd12_L'], dircos_limit['shd12_U'])
            self.m.pose[51] = generator.get_noisy_angle_hard_limit(original_pose[51], dircos_limit['shd10_L'], dircos_limit['shd10_U'])
            self.m.pose[52] = generator.get_noisy_angle_hard_limit(original_pose[52], -dircos_limit['shd11_U'], -dircos_limit['shd11_L'])
            self.m.pose[53] = generator.get_noisy_angle_hard_limit(original_pose[53], -dircos_limit['shd12_U'], -dircos_limit['shd12_L'])
            self.m.pose[57] = original_pose[57]
            self.m.pose[58] = generator.get_noisy_angle_hard_limit(original_pose[58], -dircos_limit['elbow_U'], -dircos_limit['elbow_L'])
            self.m.pose[59] = original_pose[59]

            for i in range(60, 72):
                self.m.pose[i] = original_pose[i]


            print "stuck in loop", self.reset_pose
            if self.reset_pose == True:
                pass
            else:
                break

        from capsule_body import get_capsules, joint2name, rots0
        capsules = get_capsules(self.m)
        joint2name = joint2name
        rots0 = rots0
        print len(capsules)

        return self.m, capsules, joint2name, rots0





    def generate_dataset(self, gender, some_subject, num_samp_per_slp_pose, posture):
        #NEED FOR DATASET: pose Nx72, shape Nx10
        shape_pose_vol_list = []
        #contact_check_bns = [1, 2, 4, 5, 7, 8, 14, 15, 16, 17, 18, 19]
        contact_exceptions = [[9, 14],[9, 15]]


        if posture == 'lay': pose_num_bounds = [1, 16]
        elif posture == 'lside': pose_num_bounds = [16, 31]
        elif posture == 'rside': pose_num_bounds = [31, 46]

        for pose_num in range(pose_num_bounds[0], pose_num_bounds[1]):#45 pose per participant.
            #here load some subjects joint angle data within danaLab and found by SMPLIFY

            total_num_skips = 0
            #try:
                #original_pose_data = load_pickle('/home/henry/git/smplify_public/output_'+some_subject+'/%04d.pkl' % (pose_num))
            original_pose_data = load_pickle('/home/henry/data/01_init_poses/slp_gt/uncover_'+some_subject+'/results/image_%06d/000.pkl' % (pose_num))
            #original_pose_data = load_pickle('/home/henry/data/01_init_poses/slp_1-14/uncover_'+some_subject+'/results/image_%06d/000.pkl' % (pose_num))
            #except:
            #    continue
            #print(original_pose_data)
            for item in original_pose_data: print item

            original_pose = np.array(list(original_pose_data['global_orient']) + list(original_pose_data['body_pose']))
            print("original pose", np.shape(original_pose))

            for i in range(num_samp_per_slp_pose):
                contact_check_bns = [4, 5, 7, 8, 16, 17, 18, 19]
                shape_pose_vol = [[],[],[],[],[],[],[]]
                if i - total_num_skips == 15: break

                #root_rot = np.random.uniform(-np.pi / 16, np.pi / 16)
                shift_side = np.random.uniform(-0.2, 0.2)  # in meters
                shift_ud = np.random.uniform(-0.2, 0.2)  # in meters
                shape_pose_vol[3] = None #instead of root_rot
                shape_pose_vol[4] = shift_side
                shape_pose_vol[5] = shift_ud

                generator.sample_body_shape(sampling = "UNIFORM", sigma = 0, one_side_range = 3)
                in_collision = True

                print('init m betas', self.m.betas)
                self.m.pose[:] = np.random.rand(self.m.pose.size) * 0.
                dss = dart_skel_sim.DartSkelSim(render=True, m=self.m, gender=gender, posture='lay', stiffness=None, check_only_distal = True, filepath_prefix=self.filepath_prefix, add_floor = False)
                print "dataset create type", DATASET_CREATE_TYPE
                #print self.m.pose

                volumes = dss.getCapsuleVolumes(mm_resolution = 1., dataset_num = DATASET_CREATE_TYPE)
                #volumes = 0
                #libRender.standard_render(self.m)
                print volumes
                shape_pose_vol[6] = volumes
                dss.world.reset()
                dss.world.destroy()

                num_resamp_pose_from_shape = 0
                too_many_attempts = False


                while in_collision == True:

                    #print "GOT HERE"
                    #time.sleep(2)

                    self.m.pose[:] = np.random.rand(self.m.pose.size) * 0.

                    m, capsules, joint2name, rots0 = generator.map_slp_to_rand_angles(original_pose)

                    #print "GOT HERE"
                    #time.sleep(2)

                    shape_pose_vol[0] = np.asarray(m.betas).tolist()

                    print "stepping", m.betas
                    dss = dart_skel_sim.DartSkelSim(render=True, m=m, gender=gender, posture = 'lay', stiffness=None, check_only_distal = True, filepath_prefix=self.filepath_prefix, add_floor = True)

                    #print "stepping", m.pose
                    invalid_pose = False
                    #run a step to check for collisions
                    dss.run_sim_step()

                    dss.world.check_collision()
                    #print "checked collisions"
                    #dss.run_simulation(1)
                    #print dss.world.CollisionResult()
                    #print dss.world.collision_result.contacted_bodies
                    print dss.world.collision_result.contact_sets
                    if len(dss.world.collision_result.contacted_bodies) != 0:
                        for contact_set in dss.world.collision_result.contact_sets:
                            if contact_set[0] in contact_check_bns or contact_set[1] in contact_check_bns: #consider removing spine 3 and upper legs
                                if contact_set in contact_exceptions:
                                    pass

                                else:
                                    #print "one of the limbs in contact"
                                    print contact_set
                                    print "resampling pose from the same shape, invalid pose. Pose num:", pose_num, "   sample num:", i, "   try number: ", num_resamp_pose_from_shape, "   total skips: ", total_num_skips, "   list len: " ,len(shape_pose_vol_list)

                                    num_resamp_pose_from_shape += 1

                                    #dss.run_simulation(1)

                                    #dss.run_simulation(1000)

                                    #libRender.standard_render(self.m)
                                    in_collision = True
                                    invalid_pose = True

                                    if num_resamp_pose_from_shape >= 500:
                                        too_many_attempts = True
                                        in_collision = False
                                        total_num_skips += 1
                                    elif num_resamp_pose_from_shape >= 400:
                                        contact_check_bns = [4, 5, 7, 8, 16, 17]
                                break

                        if invalid_pose == False:
                            print "resampling shape and pose, collision not important. Pose num:", pose_num, "   sample num:", i
                            #dss.run_simulation(1)
                            #libRender.standard_render(self.m)
                            in_collision = False
                    else: # no contacts anywhere.

                        print "resampling shape and pose, no collision. Pose num:", pose_num, "   sample num:", i
                        #dss.run_simulation(1)
                        in_collision = False
                        #libRender.standard_render(self.m)



                    #dss.world.skeletons[0].remove_all_collision_pairs()

                    #libRender.standard_render(self.m)
                    dss.world.reset()
                    dss.world.destroy()

                #pose_indices = [0, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 27, 36, 39, 40, 41, 42, 43, 44, 45, 48, 49, 50, 51, 52, 53, 55, 58]
                pose_angles = []
                pose_indices = []
                for index in range(72):
                    pose_indices.append(int(index))
                    pose_angles.append(float(m.pose[index]))

                shape_pose_vol[1] = pose_indices
                shape_pose_vol[2] = pose_angles

                if too_many_attempts == False:
                    shape_pose_vol_list.append(shape_pose_vol)

        print "SAVING! "
        #print shape_pose_vol_list
        #pickle.dump(shape_pose_vol_list, open("/home/henry/git/volumetric_pose_gen/valid_shape_pose_vol_list1.pkl", "wb"))
        np.save(self.filepath_prefix+"/data/01_init_poses/slp/valid_shape_pose_vol_"+some_subject+"_"+gender+"_"+posture+"_"+str(len(shape_pose_vol_list))+"_setB.npy", np.array(shape_pose_vol_list))



    def doublecheck_prechecked_list(self, gender, filename):
        prechecked_pose_list = np.load(filename).tolist()
        contact_check_bns = [4, 5, 7, 8, 16, 17, 18, 19]

        left_abd, right_abd = [], []
        for shape_pose_vol in prechecked_pose_list[0:100]:
            for idx in range(len(shape_pose_vol[0])):
                #print shape_pose_vol[0][idx]
                self.m.betas[idx] = shape_pose_vol[0][idx]


            for idx in range(len(shape_pose_vol[1])):
                #print shape_pose_vol[1][idx]
                #print self.m.pose[shape_pose_vol[1][idx]]
                #print shape_pose_vol[2][idx]

                self.m.pose[shape_pose_vol[1][idx]] = shape_pose_vol[2][idx]

            left_abd.append(np.array(self.m.pose[5]))
            right_abd.append(float(self.m.pose[8]))
            #sleep(1)

            dss = dart_skel_sim.DartSkelSim(render=True, m=self.m, gender=gender, posture='lay', stiffness=None,
                                            shiftSIDE=shape_pose_vol[4], shiftUD=shape_pose_vol[5],
                                            filepath_prefix=self.filepath_prefix, add_floor = False)
            dss.run_sim_step()
            dss.world.destroy()
        print np.mean(left_abd) #.15 .16
        print np.mean(right_abd) #-.18 -.13

                # print "one of the limbs in contact"
            #print dss.world.collision_result.contact_sets

    def render_pose(self, original_pose):
        for i in range(72):
            self.m.pose[i] = original_pose[i]



        human_mesh_vtx_all = [np.array(self.m.r)]
        human_mesh_face_all = [np.array(self.m.f)]


        #GET MESH WITH PMAT
        tm_curr = trimesh.base.Trimesh(vertices=np.array(human_mesh_vtx_all[0]), faces = np.array(human_mesh_face_all[0]))
        tm_list = [tm_curr]

        self.human_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.0, 0.0, 0.0, 0.0])  # [0.05, 0.05, 0.8, 0.0])#
        mesh_list = []
        mesh_list.append(pyrender.Mesh.from_trimesh(tm_list[0], material = self.human_mat, wireframe = True)) #this is for the main human



        #print "Viewing"
        if self.first_pass == True:

            for mesh_part in mesh_list:
                self.scene.add(mesh_part)


            lighting_intensity = 20.

            self.viewer = pyrender.Viewer(self.scene, use_raymond_lighting=True, lighting_intensity=lighting_intensity,
                                          point_size=2, run_in_thread=True, viewport_size=(1200, 1200))

            self.first_pass = False

            self.node_list = []
            for mesh_part in mesh_list:
                for node in self.scene.get_nodes(obj=mesh_part):
                    self.node_list.append(node)


        else:
            self.viewer.render_lock.acquire()

            # reset the human mesh
            for idx in range(len(mesh_list)):
                self.scene.remove_node(self.node_list[idx])
                self.scene.add(mesh_list[idx])
                for node in self.scene.get_nodes(obj=mesh_list[idx]):
                    self.node_list[idx] = node

            self.viewer.render_lock.release()
        #time.sleep(5)
        raw_input("Press Enter to continue...")



    def graph_angles(self):

        hip0_L_list = []
        hip0_R_list = []
        hip1_L_list = []
        hip1_R_list = []
        hip2_L_list = []
        hip2_R_list = []

        knee0_L_list = []
        knee0_R_list = []

        should00_L_list = []
        should00_R_list = []
        should01_L_list = []
        should01_R_list = []
        should02_L_list = []
        should02_R_list = []

        should10_L_list = []
        should10_R_list = []
        should11_L_list = []
        should11_R_list = []
        should12_L_list = []
        should12_R_list = []

        elbow1_L_list = []
        elbow1_R_list = []
        for i in [1,2,3,4,5,6,9]:
            if i == 7: continue
            some_subject = '%05d' % (i)

            #for pose_num in range(1, 46):#45 pose per participant.
            for pose_num in range(1, 46):#45 pose per participant.

                #here load some subjects joint angle data within danaLab and found by SMPLIFY

                try:
                    original_pose_data = load_pickle('/home/henry/data/01_init_poses/slp_fits_1-7/uncover_'+some_subject+'/results/image_%06d/000.pkl' % (pose_num))

                    #print(original_pose_data['body_pose'])

                    #original_pose = list(original_pose_data['body_pose'][0])
                    original_pose = list(original_pose_data['body_pose'])
                    original_pose = [0.0, 0.0, 0.0] + original_pose

                    original_pose_deg = list((np.array(original_pose)*180/np.pi).astype(np.int))

                    '''if original_pose_deg[3] < -120: continue
                    if original_pose_deg[3] > 10: continue

                    if original_pose_deg[4] < -60: continue
                    if original_pose_deg[4] > 120: continue

                    if original_pose_deg[5] < -60: continue
                    if original_pose_deg[5] > 60: continue

                    if original_pose_deg[6] < -120: continue
                    if original_pose_deg[6] > 10: continue

                    if original_pose_deg[7] < -120: continue
                    if original_pose_deg[7] > 60: continue

                    if original_pose_deg[8] < -60: continue
                    if original_pose_deg[8] > 60: continue

                    #[-120., 10.], [-60., 120.], [-60., 60.],
                    #[-120., 10.], [-120., 60.], [-60., 60.],



                    if original_pose_deg[39] < -40: continue
                    if original_pose_deg[39] > 50: continue

                    if original_pose_deg[40] < -50: continue
                    if original_pose_deg[40] > 20: continue

                    if original_pose_deg[41] < -40: continue
                    if original_pose_deg[41] > 50: continue

                    if original_pose_deg[42] < -40: continue
                    if original_pose_deg[42] > 50: continue

                    if original_pose_deg[43] < -20: continue
                    if original_pose_deg[43] > 50: continue

                    if original_pose_deg[44] < -50: continue
                    if original_pose_deg[44] > 40: continue


                    if original_pose_deg[48] < -70: continue
                    if original_pose_deg[48] > 70: continue

                    if original_pose_deg[49] < -90: continue
                    if original_pose_deg[49] > 35: continue

                    if original_pose_deg[50] < -90: continue
                    if original_pose_deg[50] > 30: continue

                    if original_pose_deg[51] > 70: continue
                    if original_pose_deg[51] < -70: continue

                    if original_pose_deg[52] < -35: continue
                    if original_pose_deg[52] > 90: continue

                    if original_pose_deg[53] < -30: continue
                    if original_pose_deg[53] > 90: continue'''


                    print(some_subject, pose_num, original_pose_deg[39:45], original_pose_deg[48:54])
                    #print(some_subject, pose_num, original_pose_deg[3:6], original_pose_deg[6:9])
                    #if original_pose_deg[40] < -60:
                    #if original_pose_deg[39] > 40:
                    if original_pose_deg[7] < -120:
                        self.render_pose(original_pose)

                    #(33, [-9, 7, -29, 25, 92, -50], [39, -4, -69, 65, 66, -21])

                    '''if original_pose[3] > 0.4:
                        print('/home/henry/git/smplify_public/output_'+some_subject+'/%04d.pkl' % (pose_num))
                        pass
                    if original_pose[6] > 0.4:
                        print('/home/henry/git/smplify_public/output_'+some_subject+'/%04d.pkl' % (pose_num))
                        pass
                    if original_pose[15] < -0.05:
                        print('/home/henry/git/smplify_public/output_'+some_subject+'/%04d.pkl' % (pose_num))
                        pass
                    elif original_pose[12] < -0.05:
                        print('/home/henry/git/smplify_public/output_'+some_subject+'/%04d.pkl' % (pose_num))
                        pass
                    elif original_pose[55] > 0.2:
                        print('/home/henry/git/smplify_public/output_'+some_subject+'/%04d.pkl' % (pose_num))
                        pass
                    elif original_pose[58] < -0.2:
                        print('/home/henry/git/smplify_public/output_'+some_subject+'/%04d.pkl' % (pose_num))
                        pass
                    elif  original_pose[55] < -3:
                        print('/home/henry/git/smplify_public/output_'+some_subject+'/%04d.pkl' % (pose_num))
                        pass
                    elif  original_pose[58] > 3:
                        print('/home/henry/git/smplify_public/output_'+some_subject+'/%04d.pkl' % (pose_num))
                        pass
                    elif original_pose[49] > 0.5:
                        print('/home/henry/git/smplify_public/output_'+some_subject+'/%04d.pkl' % (pose_num))
                        pass
                    elif original_pose[52] < -0.5:
                        print('/home/henry/git/smplify_public/output_'+some_subject+'/%04d.pkl' % (pose_num))
                        pass'''

                    if False: pass
                    else:
                        hip0_L_list.append(original_pose[3]*180./np.pi)
                        hip1_L_list.append(original_pose[4]*180./np.pi)
                        hip2_L_list.append(original_pose[5]*180./np.pi)
                        hip0_R_list.append(original_pose[6]*180./np.pi)
                        hip1_R_list.append(-original_pose[7]*180./np.pi)
                        hip2_R_list.append(-original_pose[8]*180./np.pi)
                        knee0_L_list.append(original_pose[12]*180./np.pi)
                        knee0_R_list.append(original_pose[15]*180./np.pi)

                        should00_L_list.append(original_pose[39]*180./np.pi)
                        should01_L_list.append(original_pose[40]*180./np.pi)
                        should02_L_list.append(original_pose[41]*180./np.pi)
                        should00_R_list.append(original_pose[42]*180./np.pi)
                        should01_R_list.append(-original_pose[43]*180./np.pi)
                        should02_R_list.append(-original_pose[44]*180./np.pi)

                        should10_L_list.append(original_pose[48]*180./np.pi)
                        should11_L_list.append(original_pose[49]*180./np.pi)
                        should12_L_list.append(original_pose[50]*180./np.pi)
                        should10_R_list.append(original_pose[51]*180./np.pi)
                        should11_R_list.append(-original_pose[52]*180./np.pi)
                        should12_R_list.append(-original_pose[53]*180./np.pi)

                        elbow1_L_list.append(original_pose[55]*180./np.pi)
                        elbow1_R_list.append(-original_pose[58]*180./np.pi)



                except:
                    pass
                #    print('cannot load ', 'output_'+some_subject+'/%04d.pkl' % (pose_num))

        print(len(hip0_L_list))

        import matplotlib.pyplot as plt

        plt.plot(np.ones(len(hip0_L_list)), hip0_L_list, 'k.')
        plt.plot(np.ones(len(hip0_R_list))*2, hip0_R_list, 'k.')
        print('hip0 minmax', np.min([hip0_L_list+hip0_R_list]), np.max([hip0_L_list+hip0_R_list]))
        plt.plot(np.ones(len(hip1_L_list))*3, hip1_L_list, 'k.')
        plt.plot(np.ones(len(hip1_R_list))*4, hip1_R_list, 'k.')
        print('hip1 minmax', np.min([hip1_L_list+hip1_R_list]), np.max([hip1_L_list+hip1_R_list]))
        plt.plot(np.ones(len(hip2_L_list))*5, hip2_L_list, 'k.')
        plt.plot(np.ones(len(hip2_R_list))*6, hip2_R_list, 'k.')
        print('hip2 minmax', np.min([hip2_L_list+hip2_R_list]), np.max([hip2_L_list+hip2_R_list]))

        plt.plot(np.ones(len(knee0_L_list))*7, knee0_L_list, 'b.')
        plt.plot(np.ones(len(knee0_R_list))*8, knee0_R_list, 'b.')
        print('knee minmax', np.min([knee0_L_list+knee0_R_list]), np.max([knee0_L_list+knee0_R_list]))

        plt.plot(np.ones(len(should00_L_list))*9, should00_L_list, 'r.')
        plt.plot(np.ones(len(should00_R_list))*10, should00_R_list, 'r.')
        print('should00 minmax', np.min([should00_L_list+should00_R_list]), np.max([should00_L_list+should00_R_list]))
        plt.plot(np.ones(len(should01_L_list))*11, should01_L_list, 'r.')
        plt.plot(np.ones(len(should01_R_list))*12, should01_R_list, 'r.')
        print('should01 minmax', np.min([should01_L_list+should01_R_list]), np.max([should01_L_list+should01_R_list]))
        plt.plot(np.ones(len(should02_L_list))*13, should02_L_list, 'r.')
        plt.plot(np.ones(len(should02_R_list))*14, should02_R_list, 'r.')
        print('should02 minmax', np.min([should02_L_list+should02_R_list]), np.max([should02_L_list+should02_R_list]))

        plt.plot(np.ones(len(should10_L_list))*15, should10_L_list, 'r.')
        plt.plot(np.ones(len(should10_R_list))*16, should10_R_list, 'r.')
        print('should10 minmax', np.min([should10_L_list+should10_R_list]), np.max([should10_L_list+should10_R_list]))
        plt.plot(np.ones(len(should11_L_list))*17, should11_L_list, 'r.')
        plt.plot(np.ones(len(should11_R_list))*18, should11_R_list, 'r.')
        print('should11 minmax', np.min([should11_L_list+should11_R_list]), np.max([should11_L_list+should11_R_list]))
        plt.plot(np.ones(len(should12_L_list))*19, should12_L_list, 'r.')
        plt.plot(np.ones(len(should12_R_list))*20, should12_R_list, 'r.')
        print('should12 minmax', np.min([should12_L_list+should12_R_list]), np.max([should12_L_list+should12_R_list]))

        plt.plot(np.ones(len(elbow1_L_list))*21, elbow1_L_list, 'g.')
        plt.plot(np.ones(len(elbow1_R_list))*22, elbow1_R_list, 'g.')
        print('elbow minmax', np.min([elbow1_L_list+elbow1_R_list]), np.max([elbow1_L_list+elbow1_R_list]))

        plt.grid()
        plt.show()

    def resave_individual_files(self):

        import os


        #starting_subj = 51
        for starting_subj in [71]:

            posture_list =  [['rside', 30]]#[["lside", 15]]#,[["lay", 0]]#
            #posture_list = [["lside", 15]]#,['rside', 30]]
            for posture in posture_list:

                for gender in ["f"]:#, "m"]:

                    new_valid_shape_pose_vol_list = []

                    #for i in range(51,103):

                    for i in range(starting_subj,starting_subj+10):
                        if i == 7: continue
                        some_subject = '%05d' % (i)


                        for number_files_subt in range(100):
                            try:
                                number_files = 225-number_files_subt
                                #print(self.filepath_prefix + "/data/01_init_poses/slp/"+posture[0]+"/valid_shape_pose_vol_" +some_subject+ "_"+gender +"_"+posture[0]+ "_"+str(number_files)+".npy")
                                subject_new_samples = list(np.load(self.filepath_prefix + "/data/01_init_poses/slp/"+posture[0]+"/valid_shape_pose_vol_" +some_subject+ "_"+gender +"_"+posture[0]+ "_"+str(number_files)+"_setB.npy", allow_pickle=True))
                                break
                            except:
                                if number_files_subt == 99:
                                    print("INSUFFICIENT FILES!")
                                    sys.exit()
                                pass
                        new_valid_shape_pose_vol_list = new_valid_shape_pose_vol_list + subject_new_samples

                        print(i, len(subject_new_samples), len(new_valid_shape_pose_vol_list))




                    np.save(self.filepath_prefix + "/data/01_init_poses/slp/valid_shape_pose_vol_" +str(starting_subj) + "to" +str(starting_subj+9) + "_" + posture[0] + "_" + gender + "_" +  str(len(new_valid_shape_pose_vol_list)) + "_setB.npy", new_valid_shape_pose_vol_list)




    def fix_root(self):

        import os



        for gender in ["f", "m"]:

            #for i in range(51,103):
            for i in range(52, 103):
                if i == 7: continue
                some_subject = '%05d' % (i)


                onlyfiles = next(os.walk('/home/henry/git/smplify_public/output_'+some_subject))[2]
                print(len(onlyfiles))
                number_files = str(int(len(onlyfiles)*15/2))

                new_valid_shape_pose_vol_list = []

                try:
                    subject_new_samples = np.load(self.filepath_prefix + "/data/init_poses/slp/orig_bins/valid_shape_pose_vol_" +some_subject+ "_"+gender + "_"+number_files+".npy", allow_pickle=True)
                except:
                    continue

                print(len(subject_new_samples))

                #for pose_num in range(1, 46):#45 pose per participant.
                for pose_num in range(0, 45):#45 pose per participant.

                    #here load some subjects joint angle data within danaLab and found by SMPLIFY
                    try:
                        subject_new_samples[pose_num][2][1] = -float(subject_new_samples[pose_num][2][1])

                    except:
                        #pass
                        print(i,pose_num,' is missing for subject:', some_subject)


                np.save(self.filepath_prefix + "/data/init_poses/slp/orig_bins/valid_shape_pose_vol_" +some_subject+ "_"+gender + "_"+number_files+".npy", subject_new_samples)




if __name__ == "__main__":


    '''some_subject = "00076"
    gender = "m"

    valid_shape_pos = np.load('/home/henry/data/init_poses/slp/valid_shape_pose_vol_' + some_subject + "_" + gender + '_1260.npy', allow_pickle=True)
    print(len(valid_shape_pos), "LENGTH")
    valid_shape_pos_half = []
    for i in range(len(valid_shape_pos)):
        if i%2 == 0:
            pass
        else:
            valid_shape_pos_half.append(valid_shape_pos[i])
    print(len(valid_shape_pos_half))
    np.save("/home/henry/data/init_poses/slp/valid_shape_pose_vol_" + some_subject + "_" + gender + "_" + str(
        len(valid_shape_pos_half)) + ".npy", np.array(valid_shape_pos_half))'''


    #posture = 'rside'#'lside'#'lay' #
    #some_subject = '00005'

    posture = 'lay' #'rside'#'lside'#
    some_subject = '00001'


    filepath_prefix = "/home/henry"



    phys_arr = np.load('../../data/SLP/danaLab/physiqueData.npy')
    phys_arr[:, [2, 0]] = phys_arr[:, [0, 2]]
    gender_bin = phys_arr[int(some_subject) - 1][2]
    if int(gender_bin) == 1:
        gender = "f"
    else:
        gender = "m"
    #gender = "f"

    #DATASET_CREATE_TYPE = 1
    print(gender, "GENDER")


    generator = GeneratePose(gender, filepath_prefix)

    #generator.graph_angles()

    #generator.resave_individual_files()
    #generator.check_for_invalid_poses_resave()
    #generator.fix_root()
    generator.generate_dataset(gender, some_subject = some_subject, num_samp_per_slp_pose = 30, posture = posture)
    #generator.fix_dataset(gender = "m", num_data = 3000, filepath_prefix = filepath_prefix)
    #generator.doublecheck_prechecked_list(gender, filepath_prefix+"/data/init_poses/valid_shape_pose_"+gender+"_"+str(num_data)+".npy")

