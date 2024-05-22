
try:
    import open3d as o3d
except:
    print ("COULD NOT IMPORT 03D")
import trimesh
import pyrender
import pyglet
from scipy import ndimage
from scipy.ndimage.interpolation import zoom

import numpy as np
import random
import copy
import sys

txtfile = open("/home/ganyong/Githubwork/Examples/BodyPressure/FILEPATH.txt")
FILEPATH = txtfile.read().replace("\n", "")
txtfile.close()

try:
    from smpl.smpl_webuser.serialization import load_model
except:
    from smpl.smpl_webuser3.serialization import load_model



from time import sleep

#ROS
#import rospy
#import tf
DATASET_CREATE_TYPE = 1

import cv2

import math
from random import shuffle
import torch
import torch.nn as nn


#MISC
import time as time
import matplotlib.pyplot as plt
import matplotlib.cm as cm #use cm.jet(list)

viridis = cm.get_cmap('viridis', 255)
inferno = cm.get_cmap('inferno', 255)
jet = cm.get_cmap('jet', 255)
#from mpl_toolkits.mplot3d import Axes3D

#hmr
#from hmr.src.tf_smpl.batch_smpl import SMPL

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

import os


from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvasAgg




class pyRenderMesh():
    def __init__(self, render):

        # terms = 'f', 'frustum', 'background_image', 'overdraw', 'num_channels'
        # dterms = 'vc', 'camera', 'bgcolor'
        self.first_pass = True
        self.render = render
        self.scene = pyrender.Scene()

        #self.human_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.0, 0.0, 1.0 ,0.0])
        self.human_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.05, 0.05, 0.8, 0.0], metallicFactor=0.6, roughnessFactor=0.5)#
        self.human_mat_gt = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.05, 0.05, 0.05, 0.0], metallicFactor=0.6, roughnessFactor=0.5)#

        self.human_mat_GT = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.0, 0.3, 0.0 ,0.0])
        self.human_arm_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.1, 0.1, 0.8 ,1.0])
        self.human_mat_for_study = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.3, 0.3, 0.3 ,0.5])
        self.human_bed_for_study = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.7, 0.7, 0.2 ,0.5])
        self.human_mat_D = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.1, 0.1, 0.1, 1.0], alphaMode="BLEND")

        #if render == True:
        mesh_color_mult = 0.25

        self.mesh_parts_mat_list = [
            pyrender.MetallicRoughnessMaterial(baseColorFactor=[mesh_color_mult * 166. / 255., mesh_color_mult * 206. / 255., mesh_color_mult * 227. / 255., 0.0]),
            pyrender.MetallicRoughnessMaterial(baseColorFactor=[mesh_color_mult * 31. / 255., mesh_color_mult * 120. / 255., mesh_color_mult * 180. / 255., 0.0]),
            pyrender.MetallicRoughnessMaterial(baseColorFactor=[mesh_color_mult * 251. / 255., mesh_color_mult * 154. / 255., mesh_color_mult * 153. / 255., 0.0]),
            pyrender.MetallicRoughnessMaterial(baseColorFactor=[mesh_color_mult * 227. / 255., mesh_color_mult * 26. / 255., mesh_color_mult * 28. / 255., 0.0]),
            pyrender.MetallicRoughnessMaterial(baseColorFactor=[mesh_color_mult * 178. / 255., mesh_color_mult * 223. / 255., mesh_color_mult * 138. / 255., 0.0]),
            pyrender.MetallicRoughnessMaterial(baseColorFactor=[mesh_color_mult * 51. / 255., mesh_color_mult * 160. / 255., mesh_color_mult * 44. / 255., 0.0]),
            pyrender.MetallicRoughnessMaterial(baseColorFactor=[mesh_color_mult * 253. / 255., mesh_color_mult * 191. / 255., mesh_color_mult * 111. / 255., 0.0]),
            pyrender.MetallicRoughnessMaterial(baseColorFactor=[mesh_color_mult * 255. / 255., mesh_color_mult * 127. / 255., mesh_color_mult * 0. / 255., 0.0]),
            pyrender.MetallicRoughnessMaterial(baseColorFactor=[mesh_color_mult * 202. / 255., mesh_color_mult * 178. / 255., mesh_color_mult * 214. / 255., 0.0]),
            pyrender.MetallicRoughnessMaterial(baseColorFactor=[mesh_color_mult * 106. / 255., mesh_color_mult * 61. / 255., mesh_color_mult * 154. / 255., 0.0])]

        self.artag_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.3, 1.0, 0.3, 0.5])
        self.artag_mat_other = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.1, 0.1, 0.1, 0.0])
        #self.artag_r = np.array([[-0.055, -0.055, 0.0], [-0.055, 0.055, 0.0], [0.055, -0.055, 0.0], [0.055, 0.055, 0.0]])
        self.artag_r = np.array([[0.0, 0.0, 0.075], [0.0286*64*1.04/1.04, 0.0, 0.075], [0.0, 0.01, 0.075], [0.0286*64*1.04/1.04, 0.01, 0.075],
                                 [0.0, 0.0, 0.075], [0.0, 0.0286*27 /1.06, 0.075], [0.01, 0.0, 0.075], [0.01, 0.0286*27 /1.06, 0.075],
                                 [0.0,  0.0286*27 /1.06, 0.075], [0.0286*64*1.04/1.04, 0.0286*27 /1.06, 0.075], [0.0,  0.0286*27 /1.06+0.01, 0.075], [0.0286*64*1.04/1.04,  0.0286*27 /1.06+0.01, 0.075],
                                 [0.0286*64*1.04/1.04, 0.0, 0.075], [0.0286*64*1.04/1.04, 0.0286*27 /1.06, 0.075], [0.0286*64*1.04/1.04-0.01, 0.0, 0.075], [0.0286*64*1.04/1.04-0.01, 0.0286*27 /1.06, 0.075],
                                 ])
        #self.artag_f = np.array([[0, 1, 3], [3, 1, 0], [0, 2, 3], [3, 2, 0], [1, 3, 2]])
        self.artag_f = np.array([[0, 1, 2], [0, 2, 1], [1, 2, 3], [1, 3, 2],
                                 [4, 5, 6], [4, 6, 5], [5, 6, 7], [5, 7, 6],
                                 [8, 9, 10], [8, 10, 9], [9, 10, 11], [9, 11, 10],
                                 [12, 13, 14], [12, 14, 13], [13, 14, 15], [13, 15, 14]])
        #self.artag_facecolors_root = np.array([[0.0, 1.0, 0.0],[0.0, 1.0, 0.0],[0.0, 1.0, 0.0],[0.0, 1.0, 0.0],[0.0, 1.0, 0.0]])
        self.artag_facecolors_root =  np.array([[0.3, 0.3, 0.0],[0.3, 0.3, 0.0],[0.3, 0.3, 0.0],[0.3, 0.3, 0.0],
                                                [0.3, 0.3, 0.0],[0.3, 0.3, 0.0],[0.3, 0.3, 0.0],[0.3, 0.3, 0.0],
                                                [0.3, 0.3, 0.0],[0.3, 0.3, 0.0],[0.3, 0.3, 0.0],[0.3, 0.3, 0.0],
                                                [0.3, 0.3, 0.0],[0.3, 0.3, 0.0],[0.3, 0.3, 0.0],[0.3, 0.3, 0.0],
                                                ])
        self.artag_facecolors_root_gt =  np.array([[0.1, 0.1, 0.1],[0.1, 0.1, 0.1],[0.1, 0.1, 0.1],[0.1, 0.1, 0.1],
                                                    [0.1, 0.1, 0.1],[0.1, 0.1, 0.1],[0.1, 0.1, 0.1],[0.1, 0.1, 0.1],
                                                    [0.1, 0.1, 0.1],[0.1, 0.1, 0.1],[0.1, 0.1, 0.1],[0.1, 0.1, 0.1],
                                                    [0.1, 0.1, 0.1],[0.1, 0.1, 0.1],[0.1, 0.1, 0.1],[0.1, 0.1, 0.1],
                                                    ])*0.5
        #self.artag_facecolors = np.array([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],])
        self.artag_facecolors = np.copy(self.artag_facecolors_root)
        self.artag_facecolors_gt = np.copy(self.artag_facecolors_root_gt)


        self.pic_num = 0




    def get_human_mesh_parts(self, smpl_verts, smpl_faces, viz_type = None, segment_limbs = False):

        if segment_limbs == True:
            if viz_type == 'arm_penetration':
                segmented_dict = load_pickle('segmented_mesh_idx_faces_larm.p')
                human_mesh_vtx_parts = [smpl_verts[segmented_dict['l_arm_idx_list'], :]]
                human_mesh_face_parts = [segmented_dict['l_arm_face_list']]
            elif viz_type == 'leg_correction':
                segmented_dict = load_pickle('segmented_mesh_idx_faces_rleg.p')
                human_mesh_vtx_parts = [smpl_verts[segmented_dict['r_leg_idx_list'], :]]
                human_mesh_face_parts = [segmented_dict['r_leg_face_list']]
            else:
                segmented_dict = load_pickle('segmented_mesh_idx_faces.p')
                human_mesh_vtx_parts = [smpl_verts[segmented_dict['l_lowerleg_idx_list'], :],
                                        smpl_verts[segmented_dict['r_lowerleg_idx_list'], :],
                                        smpl_verts[segmented_dict['l_upperleg_idx_list'], :],
                                        smpl_verts[segmented_dict['r_upperleg_idx_list'], :],
                                        smpl_verts[segmented_dict['l_forearm_idx_list'], :],
                                        smpl_verts[segmented_dict['r_forearm_idx_list'], :],
                                        smpl_verts[segmented_dict['l_upperarm_idx_list'], :],
                                        smpl_verts[segmented_dict['r_upperarm_idx_list'], :],
                                        smpl_verts[segmented_dict['head_idx_list'], :],
                                        smpl_verts[segmented_dict['torso_idx_list'], :]]
                human_mesh_face_parts = [segmented_dict['l_lowerleg_face_list'],
                                         segmented_dict['r_lowerleg_face_list'],
                                         segmented_dict['l_upperleg_face_list'],
                                         segmented_dict['r_upperleg_face_list'],
                                         segmented_dict['l_forearm_face_list'],
                                         segmented_dict['r_forearm_face_list'],
                                         segmented_dict['l_upperarm_face_list'],
                                         segmented_dict['r_upperarm_face_list'],
                                         segmented_dict['head_face_list'],
                                         segmented_dict['torso_face_list']]
        else:
            human_mesh_vtx_parts = [smpl_verts]
            human_mesh_face_parts = [smpl_faces]

        return human_mesh_vtx_parts, human_mesh_face_parts




    def render_mesh_pc_bed_pyrender_everything(self, smpl_verts, smpl_faces, RESULTS_DICT,
                                    pmat = None, pmat_est = None, color_im_occl = None,
                                    color_im = None, depth_im = None, current_pose_type_ct = None,
                                    participant = None):

        markers = [[0.0, 0.0, 0.0],
                   [0.0, 1.5, 0.0],
                   [0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0]]

        #pmat[pmat>0] += 1
        #pmat_est[pmat_est>0] += 1
        #pmat *= 0.75
        #pmat[pmat>0] += 10

        smpl_verts[:, 0] += 0.0286*2 #vertical
        smpl_verts[:, 1] += 0.0143

        #print np.min(smpl_verts[:, 0])
        #print np.min(smpl_verts[:, 1])

        shift_estimate_sideways = np.max([0.8, np.max(smpl_verts[:, 1])])
        #print shift_estimate_sideways
        shift_estimate_sideways = 0.1 + shift_estimate_sideways



        shift_both_amount =  - np.min([-0.1, np.min(smpl_verts[:, 1])])

        #print np.max(smpl_verts[:, 1]), 'max smpl'

        #shift_both_amount = 0.6
        #smpl_verts[:, 2] += 0.5
        #pc[:, 2] += 0.5

        #pc[:, 0] = pc[:, 0] # - 0.17 - 0.036608
        #pc[:, 1] = pc[:, 1]# + 0.09

        #adjust the point cloud


        #segment_limbs = True

        #if pmat is not None:
         #   if np.sum(pmat) < 5000:
         #       smpl_verts = smpl_verts * 0.001


        smpl_verts_quad = np.concatenate((smpl_verts, np.ones((smpl_verts.shape[0], 1))), axis = 1)
        smpl_verts_quad = np.swapaxes(smpl_verts_quad, 0, 1)

        #print smpl_verts_quad.shape

        transform_A = np.identity(4)
        transform_A[1, 3] = shift_both_amount
        smpl_verts_A = np.swapaxes(np.matmul(transform_A, smpl_verts_quad), 0, 1)[:, 0:3]

        transform_B = np.identity(4)
        transform_B[1, 3] = shift_estimate_sideways + shift_both_amount#4.0 #move things over

        transform_C = np.copy(transform_B)
        middle_horiz_filler = 100
        transform_C[0, 3] -= (64*0.0286 + (64*0.0286)*middle_horiz_filler/880)#2.0 #move things over
        #smpl_verts_C = np.swapaxes(np.matmul(transform_C, smpl_verts_quad), 0, 1)[:, 0:3]


        if True:
            smpl_verts_A *= (64*0.0286 / 1.92)
            smpl_verts_A[:, 0] += (1.92 - 64*0.0286)/2
            smpl_verts_A[:, 1] += (0.84 - 27*0.0286)/2


        #from matplotlib import cm
        human_mesh_vtx_all, human_mesh_face_all = self.get_human_mesh_parts(smpl_verts_A, smpl_faces, segment_limbs=False)

        #GET MESH WITH PMAT
        tm_curr = trimesh.base.Trimesh(vertices=np.array(human_mesh_vtx_all[0]), faces = np.array(human_mesh_face_all[0]))
        tm_list = [tm_curr]
        original_mesh = [tm_curr]


        mesh_list = []
        mesh_list.append(pyrender.Mesh.from_trimesh(tm_list[0], material = self.human_mat, smooth=True))#wireframe = False)) #this is for the main human


        top_idx = 0
        bot_idx = 128
        perc_total = 1.0

        fig = plt.figure()
        if self.render == True:

            #print m.r
            #print artag_r
            #create mini meshes for AR tags
            artag_meshes = []
            if markers is not None:
                for marker in markers:
                    if markers[2] is None:
                        artag_meshes.append(None)
                    elif marker is None:
                        artag_meshes.append(None)
                    else:
                        #print marker - markers[2]
                        if marker is markers[2]:
                            print ("is markers 2", marker)
                            #artag_tm = trimesh.base.Trimesh(vertices=self.artag_r, faces=self.artag_f, face_colors = self.artag_facecolors_root)
                            #artag_meshes.append(pyrender.Mesh.from_trimesh(artag_tm, smooth = False))
                        else:
                            artag_tm = trimesh.base.Trimesh(vertices=self.artag_r + [0.0, shift_both_amount, 0.0], faces=self.artag_f, face_colors = self.artag_facecolors)
                            artag_meshes.append(pyrender.Mesh.from_trimesh(artag_tm, smooth = False))




            #print "Viewing"
            if self.first_pass == True:

                for mesh_part in mesh_list:
                    self.scene.add(mesh_part)

                for artag_mesh in artag_meshes:
                    if artag_mesh is not None:
                        self.scene.add(artag_mesh)


                self.first_pass = False

                self.node_list = []
                for mesh_part in mesh_list:
                    for node in self.scene.get_nodes(obj=mesh_part):
                        self.node_list.append(node)



                self.artag_nodes = []
                for artag_mesh in artag_meshes:
                    if artag_mesh is not None:
                        for node in self.scene.get_nodes(obj=artag_mesh):
                            self.artag_nodes.append(node)


                camera_pose = np.eye(4)
                # camera_pose[0,0] = -1.0
                # camera_pose[1,1] = -1.0

                camera_pose[0, 0] = np.cos(np.pi/2)
                camera_pose[0, 1] = np.sin(np.pi/2)
                camera_pose[1, 0] = -np.sin(np.pi/2)
                camera_pose[1, 1] = np.cos(np.pi/2)
                rot_udpim = np.eye(4)

                rot_y = 180*np.pi/180.
                rot_udpim[1,1] = np.cos(rot_y)
                rot_udpim[2,2] = np.cos(rot_y)
                rot_udpim[1,2] = np.sin(rot_y)
                rot_udpim[2,1] = -np.sin(rot_y)
                camera_pose = np.matmul(rot_udpim,  camera_pose)

                camera_pose[0, 3] = 0#(middle_horiz_filler/880)*64*0.0286 #64*0.0286/2  # -1.0
                camera_pose[1, 3] = 1.2
                camera_pose[2, 3] = -1.0


                magnify =(1+middle_horiz_filler/880)*(64*.0286)/perc_total

                camera = pyrender.OrthographicCamera(xmag=magnify, ymag = magnify)

                self.scene.add(camera, pose=camera_pose)


                light = pyrender.SpotLight(color=np.ones(3), intensity=250.0, innerConeAngle=np.pi / 10.0,
                                           outerConeAngle=np.pi / 2.0)
                light_pose = np.copy(camera_pose)
                # light_pose[1, 3] = 2.0
                light_pose[0, 3] = 0.8
                light_pose[1, 3] = -0.5
                light_pose[2, 3] = -2.5

                light_pose2 = np.copy(camera_pose)
                light_pose2[0, 3] = 2.5
                light_pose2[1, 3] = 1.0
                light_pose2[2, 3] = -5.0

                light_pose3 = np.copy(camera_pose)
                light_pose3[0, 3] = 1.0
                light_pose3[1, 3] = 5.0
                light_pose3[2, 3] = -4.0

                self.scene.add(light, pose=light_pose)
                self.scene.add(light, pose=light_pose2)
                self.scene.add(light, pose=light_pose3)




            else:
                #self.viewer.render_lock.acquire()

                #reset the human mesh
                for idx in range(len(mesh_list)):
                    self.scene.remove_node(self.node_list[idx])
                    self.scene.add(mesh_list[idx])
                    for node in self.scene.get_nodes(obj=mesh_list[idx]):
                        self.node_list[idx] = node

                #reset the artag meshes
                for artag_node in self.artag_nodes:
                    self.scene.remove_node(artag_node)
                for artag_mesh in artag_meshes:
                    if artag_mesh is not None:
                        self.scene.add(artag_mesh)
                self.artag_nodes = []
                for artag_mesh in artag_meshes:
                    if artag_mesh is not None:
                        for node in self.scene.get_nodes(obj=artag_mesh):
                            self.artag_nodes.append(node)




        r = pyrender.OffscreenRenderer(600, 880+middle_horiz_filler)
        # r.render(self.scene)
        color_render, depth = r.render(self.scene)
        # plt.subplot(1, 2, 1)
        plt.axis('off')


        depth_im -= 1700

        #print(depth_im)
        depth_im = depth_im.astype(float)/ 500.
        depth_im = inferno(np.clip(depth_im, a_min = 0, a_max = 1))[:, :, 0:3]*255.
        depth_im = depth_im.astype(np.uint8)

        if color_im is not None:
            zeros_append = np.zeros((color_render.shape[0], color_im.shape[1], 3)).astype(np.uint8) + 255
            im_to_show = np.concatenate((zeros_append, color_render), axis = 1)
        else:
            zeros_append = np.zeros((color_render.shape[0], 191, 3)).astype(np.uint8) + 255
            im_to_show = np.concatenate((zeros_append, color_render), axis = 1)


        im_to_show[int(middle_horiz_filler/2):int(middle_horiz_filler/2)+440, 3:np.shape(depth_im)[1]+3, :] = depth_im


        #int(shift_both_amount/0.0286)


        if color_im_occl is not None:
            im_to_show[880+middle_horiz_filler-440:880+middle_horiz_filler, 0:int(np.shape(color_im_occl)[1]), :] = color_im_occl

        if color_im is not None:
        #int(shift_both_amount/0.0286)
            im_to_show[880+middle_horiz_filler-440:880+middle_horiz_filler, int(8*shift_both_amount/0.0286)+np.shape(color_im)[1]:int(8*shift_both_amount/0.0286)+int(np.shape(color_im)[1]*2), :] = color_im





        side_pmat_shift_px =  int(7.5*(shift_estimate_sideways+shift_both_amount)/0.0286)+179


        if pmat_est is not None:
            pmat_est = pmat_est.reshape(64, 27)
            pmat_est = zoom(pmat_est, (3.435*2, 3.435*2/1.04), order=0)
            pmat_est_colors = (np.clip((jet(pmat_est/30)[:, :, 0:3] + 0.1), a_min = 0, a_max = 1)*255).astype(np.uint8)
            im_to_show[int(middle_horiz_filler/2):int(middle_horiz_filler/2)+440, side_pmat_shift_px:side_pmat_shift_px+178, :] = pmat_est_colors[:, :, 0:3]


        if pmat is not None:
            pmat = pmat.reshape(64, 27)
            pmat = zoom(pmat, (3.435*2, 3.435*2/1.04), order=0)
            pmat_colors = (np.clip((jet(pmat/30)[:, :, 0:3] + 0.1), a_min = 0, a_max = 1 )*255).astype(np.uint8)
            im_to_show[880+middle_horiz_filler-440:880+middle_horiz_filler, side_pmat_shift_px:side_pmat_shift_px+178, :] = pmat_colors[:, :, 0:3]







        plt.imshow(im_to_show)
        fig.set_size_inches(10., 10.)
        fig.tight_layout()
        save_name = participant+'_'+current_pose_type_ct

        print ("saving!")
        fig.savefig(FILEPATH+'data_BP/results/results_qualitative_DMR_slp_108160ct_128b_x1pm_rgangs_slpb_dpns_rt_v2v_40e_update/'+save_name+'.png', dpi=300)
        #fig.savefig('/home/henry/git/depth-pose-pressure/data/results/'+save_name+'.png', dpi=300)

        self.pic_num += 1


        return RESULTS_DICT


