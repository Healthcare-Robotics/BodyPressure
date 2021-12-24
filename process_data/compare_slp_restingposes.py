
import numpy as np
import random
import copy
import trimesh
import pyrender


txtfile = open("../FILEPATH.txt")
FILEPATH = txtfile.read().replace("\n", "")
txtfile.close()

import sys
sys.path.insert(0, '../lib_py')
sys.path.insert(-1,FILEPATH)
sys.path.insert(-1,FILEPATH+'smpl/smpl_webuser3')
print(sys.path, 'sys path for evaluate_depthreal_slp.py')

try:
    from smpl.smpl_webuser.serialization import load_model
except:
    print("importing load model3")
    from smpl.smpl_webuser3.serialization import load_model





#volumetric pose gen libraries
#import lib_kinematics as libKinematics
import kinematics_lib_bp as kinematics_lib_br
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


#MISC
import time as time
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D




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


            if self.reset_pose == True:
                pass
            else:
                break

        from capsule_body import get_capsules, joint2name, rots0
        capsules = get_capsules(self.m)
        joint2name = joint2name
        rots0 = rots0

        return self.m, capsules, joint2name, rots0





    def generate_dataset(self, gender, posture, some_subject_int):
        #NEED FOR DATASET: pose Nx72, shape Nx10
        shape_pose_vol_list = []

        some_subject = '%03d' % (some_subject_int)



        if posture == 'lay' and some_subject_int <= 40:# and gender == 'f':

            init_pose_data = np.load('/home/henry/data/01_init_poses/slp/lay/valid_shape_pose_vol_00001_f_lay_225.npy', allow_pickle=True, encoding = 'latin1')
            #synth_pose_data = load_pickle('/home/henry/git/BodyPressure/data_BP/synth/train_slp_lay_f_1to40_8549.p')
            synth_pose_data = load_pickle('/home/henry/git/BodyPressure/data_BP/synth/train_slp_lay_m_1to40_8493.p')


            #synth_pose_data = load_pickle('/home/henry/git/BodyPressure/data_BP/synth/train_slp_lay_m_1to40_8493.p')
            synth_pose_data_ct = 28

        #for pose_idx in range(len(init_pose_data)):
        #    for data_idx in range(np.shape(init_pose_data[pose_idx])[0]):
        #        print('init pose data: ', pose_idx, np.shape(init_pose_data[pose_idx][data_idx]))


        for item in synth_pose_data: print(item, np.shape(synth_pose_data[item]))

        if posture == 'lay': pose_num_bounds = [1, 16]
        elif posture == 'lside': pose_num_bounds = [16, 31]
        elif posture == 'rside': pose_num_bounds = [31, 46]



        for pose_num in range(pose_num_bounds[0], pose_num_bounds[1]):#45 pose per participant.
            #here load some subjects joint angle data within danaLab and found by SMPLIFY

            total_num_skips = 0

            original_pose_data = load_pickle('/home/henry/git/BodyPressure/data_BP/SLP_SMPL_fits/fits/p'+some_subject+'/sd%02d.pkl' % (pose_num))
            print('/home/henry/git/BodyPressure/data_BP/SLP_SMPL_fits/fits/p'+some_subject+'/sd%02d.pkl' % (pose_num))

            for item in original_pose_data: print(item, np.shape(original_pose_data[item]))

            original_pose = np.array(list(original_pose_data['global_orient']) + list(original_pose_data['body_pose']))
            print("original pose", np.shape(original_pose))

            for ct in range(90):
                self.render_pose(synth_pose_data['joint_angles'][synth_pose_data_ct], synth_pose_data['body_shape'][synth_pose_data_ct])
                #print(synth_pose_data['joint_angles'][synth_pose_data_ct])
                #print(original_pose)
                difference =synth_pose_data['joint_angles'][synth_pose_data_ct]-original_pose
                print(synth_pose_data_ct, np.sum(np.abs(difference)), 'SLP to rest')

                init_betas = np.array(init_pose_data[ct][0])
                init_joint_angles = np.array(init_pose_data[ct][2])
                #self.render_pose(init_joint_angles, init_betas)


                difference2 =synth_pose_data['joint_angles'][synth_pose_data_ct]-init_joint_angles
                print(synth_pose_data_ct, np.sum(np.abs(difference2)), 'SLPw noise to rest')


                difference3 =original_pose-init_joint_angles
                print(synth_pose_data_ct, np.sum(np.abs(difference3)), 'SLP to SLPw noise')

                synth_pose_data_ct += 1


            #self.render_pose(original_pose, original_pose_data['betas'])
            #break




    def render_pose(self, original_pose, original_shape):
        for i in range(72):
            self.m.pose[i] = original_pose[i]

        for j in range(10):
            self.m.betas[j] = original_shape[j]



        human_mesh_vtx_all = [np.array(self.m.r)]
        human_mesh_face_all = [np.array(self.m.f)]


        #GET MESH WITH PMAT
        tm_curr = trimesh.base.Trimesh(vertices=np.array(human_mesh_vtx_all[0]), faces = np.array(human_mesh_face_all[0]))
        tm_list = [tm_curr]

        #self.human_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.05, 0.05, 0.3, 0.0])  # [0.05, 0.05, 0.8, 0.0])#
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
        input("Press Enter to continue...")




if __name__ == "__main__":



    #posture = 'rside'#'lside'#'lay' #
    #some_subject = '00005'

    posture = 'lay' #'rside'#'lside'#
    some_subject_int = 2

    filepath_prefix = "/home/henry"



    phys_arr = np.load(filepath_prefix+'/git/BodyPressure/data_BP/SLP/danaLab/physiqueData.npy')
    phys_arr[:, [2, 0]] = phys_arr[:, [0, 2]]
    gender_bin = phys_arr[some_subject_int - 1][2]
    if int(gender_bin) == 1:
        gender = "f"
    else:
        gender = "m"
    #gender = "f"

    #DATASET_CREATE_TYPE = 1
    print(gender, "GENDER")


    generator = GeneratePose(gender, filepath_prefix)

    generator.generate_dataset(gender, posture = posture, some_subject_int=some_subject_int)
    #generator.doublecheck_prechecked_list(gender, filepath_prefix+"/data/init_poses/valid_shape_pose_"+gender+"_"+str(num_data)+".npy")

