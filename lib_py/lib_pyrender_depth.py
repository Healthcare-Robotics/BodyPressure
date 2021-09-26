
try:
    import open3d as o3d
except:
    print("COULD NOT IMPORT 03D")
import trimesh
import pyrender
import pyglet
from scipy import ndimage

import numpy as np
import random
import copy

try:
    from smpl.smpl_webuser.serialization import load_model
except:
    from smpl.smpl_webuser3.serialization import load_model


#volumetric pose gen libraries
#import dart_skel_sim
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
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm

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
import lib_pyrender_functions as LPF



class pyRenderMesh():
    def __init__(self, render):

        # terms = 'f', 'frustum', 'background_image', 'overdraw', 'num_channels'
        # dterms = 'vc', 'camera', 'bgcolor'
        self.first_pass = True
        self.render = render
        if True: #render == True:
            self.scene = pyrender.Scene()

            #self.human_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.0, 0.0, 1.0 ,0.0])
            self.human_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.0, 0.0, 0.0 ,0.0])#[0.05, 0.05, 0.8, 0.0])#
            self.human_mat_GT = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.0, 0.3, 0.0 ,0.0])
            self.human_arm_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.1, 0.1, 0.8 ,1.0])
            self.human_mat_for_study = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.3, 0.3, 0.3 ,0.5])
            self.human_bed_for_study = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.7, 0.7, 0.2 ,0.5])
            self.human_mat_D = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.1, 0.1, 0.1, 1.0], alphaMode="BLEND")

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

            self.x_bump = -0.0143*2
            self.y_bump = -(0.0286*64*1.04 - 0.0286*64)/2 - 0.0143*2




        self.pic_num = 0
        self.tr_ck_arr = np.zeros((13776, 20))


    def get_transforms(self):

        transform_dict = {}

        transform_dict['smpl_pmat_est'] = np.identity(4)

        transform_dict['smpl_pmat_gt'] = np.identity(4)
        #transform_dict['smpl_pmat_gt'][1, 3] = 1.0  # move things over
        transform_dict['smpl_pmat_gt'][0, 3] = -2.0  # move things over

        transform_dict['B'] = np.identity(4)
        transform_dict['B'][1, 3] = 1.0  # move things over

        transform_dict['C'] = np.identity(4)
        transform_dict['C'][1, 3] = 2.0  # move things over

        transform_dict['D'] = np.identity(4)
        transform_dict['D'][1, 3] = 3.0  # move things over

        transform_dict['pressure_proj_est'] = np.identity(4)
        #transform_dict['pressure_proj_est'][0, 3] = -2.0  # move things over
        #transform_dict['pressure_proj_est'][1, 3] = 0.0  # move things over
        transform_dict['pressure_proj_est'][1, 3] = 1.0  # move things over

        transform_dict['pressure_proj_gt'] = np.identity(4)
        transform_dict['pressure_proj_gt'][0, 3] = -2.0  # move things over
        transform_dict['pressure_proj_gt'][1, 3] = 1.0  # move things over

        return transform_dict




    def convert_joints_to_mesh(self, joint_3D_positions, color, transform):
        if joint_3D_positions is not None:
            joint_3D_positions = joint_3D_positions.numpy()
            try:
                joint_3D_positions = joint_3D_positions.reshape(int(joint_3D_positions.shape[0] / 3), 3)
            except:
                pass
            joint_3D_positions = np.concatenate((joint_3D_positions[:, 1:2], joint_3D_positions[:, 0:1], -joint_3D_positions[:, 2:3]), axis=1)
            joint_3D_pts_quad = np.swapaxes(np.concatenate((joint_3D_positions, np.ones((joint_3D_positions.shape[0], 1))), axis=1), 0, 1)
            joint_3D_pts_tf = np.swapaxes(np.matmul(transform, joint_3D_pts_quad), 0, 1)[:, 0:3]
            sm = trimesh.creation.uv_sphere(radius=0.04)
            sm.visual.vertex_colors = color
            tfs = np.tile(np.eye(4), (len(joint_3D_pts_tf), 1, 1))
            tfs[:, :3, 3] = joint_3D_pts_tf
            joint_3D_mesh = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        else:
            joint_3D_mesh = None
        return joint_3D_mesh



    def convert_pimgarr_to_mesh(self, pimgarr, transform):
        if pimgarr is not None and self.render == True:
            pmat_verts, pmat_faces, pmat_facecolors = LPF.get_3D_pmat_markers(pimgarr, self.x_bump, self.y_bump, 0.0)
            pmat_verts = np.array(pmat_verts)
            pmat_verts = np.concatenate((np.swapaxes(pmat_verts, 0, 1), np.ones((1, pmat_verts.shape[0]))), axis=0)
            pmat_verts = np.swapaxes(np.matmul(transform, pmat_verts), 0, 1)[:, 0:3]
            pmat_tm = trimesh.base.Trimesh(vertices=pmat_verts, faces=pmat_faces, face_colors=pmat_facecolors)
            pmat_mesh = pyrender.Mesh.from_trimesh(pmat_tm, smooth=False)
        else:
            pmat_tm = None
            pmat_mesh = None
        return pmat_tm, pmat_mesh



    def convert_smplarr_to_mesh(self, smpl_verts, smpl_faces, material, transform):
        if smpl_verts is not None:
            smpl_verts_quad = np.concatenate((smpl_verts, np.ones((smpl_verts.shape[0], 1))), axis = 1)
            smpl_verts_quad = np.swapaxes(smpl_verts_quad, 0, 1)
            smpl_verts = np.swapaxes(np.matmul(transform, smpl_verts_quad), 0, 1)[:, 0:3] #gt over pressure mat

            human_mesh_vtx_all, human_mesh_face_all = LPF.get_human_mesh_parts(smpl_verts, smpl_faces, segment_limbs=False)
            tm_curr = trimesh.base.Trimesh(vertices=np.array(human_mesh_vtx_all[0]), faces = np.array(human_mesh_face_all[0]), process = False)
            smpl_trimesh_list = [tm_curr]
            smpl_mesh_list = []
            smpl_mesh_list.append(pyrender.Mesh.from_trimesh(smpl_trimesh_list[0], material=material, wireframe=True))
        else:
            smpl_trimesh_list = []
            smpl_mesh_list = []

        return smpl_trimesh_list, smpl_mesh_list




    def get_smpl_pressure_proj(self, smpl_trimesh_list_est, pmat_est, transform_prev, transform):

        if len(smpl_trimesh_list_est) > 0:

            # get the nearest point from ALL verts to some pc point, regardless of the normal - for coloring
            # we need this as a hack because the face indexing only refers to the original set of verts
            vertex_normals_pimg = np.array(smpl_trimesh_list_est[0].vertex_normals)
            vertices_pimg = np.array(smpl_trimesh_list_est[0].vertices)
            faces_pimg = np.array(smpl_trimesh_list_est[0].faces)

            vertices_pimg[:, 0] = vertices_pimg[:, 0] + transform[0, 3] - transform_prev[0, 3]
            vertices_pimg[:, 1] = vertices_pimg[:, 1] + transform[1, 3] - transform_prev[1, 3]


            norm_area_avg_all_est = LPF.get_triangle_area_vert_weight(vertices_pimg, faces_pimg, None)

            vertex_pressure_init_list_EST = []
            for all_vert_idx in range(vertices_pimg.shape[0]):
                #convert the vertex to 64x27 coords. pick color based on indexing 2d pressure map
                color_idx_y = int(64 - (vertices_pimg[all_vert_idx, 0] - transform[0, 3] - self.y_bump)/ (0.0286*1.04) )# - 2.0) + 0.035
                color_idx_x = int( (vertices_pimg[all_vert_idx, 1] - transform[1, 3] - self.x_bump)/ (0.0286) )#+ 0.5)# + 35.5 + 0.012

                try:
                    if vertex_normals_pimg[all_vert_idx, 2] > 0:
                        cancel_pressure, self.tr_ck_arr = LPF.check_vertex(all_vert_idx, self.tr_ck_arr, vertices_pimg, faces_pimg)
                        if cancel_pressure == 1:
                            vertex_pressure_init_list_EST.append(0.0)
                        else:
                            vertex_pressure_init_list_EST.append(pmat_est[color_idx_y, color_idx_x])
                    else:
                        vertex_pressure_init_list_EST.append(0.0)
                except:
                    vertex_pressure_init_list_EST.append(0.0)


            vertex_pressure_norm_est = vertex_pressure_init_list_EST*norm_area_avg_all_est

            verts_color_error = np.array(vertex_pressure_init_list_EST) /23.
            verts_color_jet = np.clip(cm.jet(verts_color_error)[:, 0:3], a_min = 0.0, a_max = 1.0)

            verts_color_jet_top = np.concatenate((verts_color_jet, np.ones((verts_color_jet.shape[0], 1))*0.9), axis = 1)
            verts_color_jet_bot = np.concatenate((verts_color_jet*0.3, np.ones((verts_color_jet.shape[0], 1))*0.9), axis = 1)

            all_verts = np.array(vertices_pimg)
            faces_red = np.array(faces_pimg)
            faces_underside = np.concatenate((faces_red[:, 0:1],
                                              faces_red[:, 2:3],
                                              faces_red[:, 1:2]), axis = 1) + 6890

            human_vtx_both_sides = np.concatenate((all_verts, all_verts+0.0001), axis = 0)
            human_mesh_faces_both_sides = np.concatenate((faces_red, faces_underside), axis = 0)
            verts_color_jet_both_sides = np.concatenate((verts_color_jet_top, verts_color_jet_bot), axis = 0)

            tm_curr = trimesh.base.Trimesh(vertices=human_vtx_both_sides,
                                           faces=human_mesh_faces_both_sides,
                                           vertex_colors = verts_color_jet_both_sides)
            smpl_trimesh_list_pimg =[tm_curr]

            smpl_mesh_list_pimg = []
            smpl_mesh_list_pimg.append(pyrender.Mesh.from_trimesh(smpl_trimesh_list_pimg[0], smooth=False))
            smpl_mesh_list_pimg.append(pyrender.Mesh.from_trimesh(smpl_trimesh_list_pimg[0], material=self.human_mat, wireframe=True))
        else:
            smpl_trimesh_list_pimg = []
            smpl_mesh_list_pimg = []
            vertex_pressure_norm_est = 0


        return smpl_trimesh_list_pimg, smpl_mesh_list_pimg, vertex_pressure_norm_est








    def render_mesh_pc_bed_pyrender_everything(self, smpl_verts, smpl_faces, camera_point, RESULTS_DICT,
                                    smpl_verts_gt = None, pc = None, pmat = None, pmat_est = None,
                                    targets=None, scores = None):


        transform_dict = self.get_transforms()


        score_mesh = self.convert_joints_to_mesh(scores, color=[0.15, 0.15, 0.0], transform=transform_dict['smpl_pmat_est'])
        target_mesh = self.convert_joints_to_mesh(targets, color=[0.0, 0.15, 0.0], transform=transform_dict['smpl_pmat_gt'])



        #process the point cloud and transform it so it overlays the ground truth SMPL mesh
        pc_red, pc_red_norm = LPF.downspl_pc_get_normals(pc, camera_point)
        pc_red_quad = np.swapaxes(np.concatenate((pc_red, np.ones((pc_red.shape[0], 1))), axis = 1), 0, 1)
        pc_red_F = np.swapaxes(np.matmul(transform_dict['smpl_pmat_gt'], pc_red_quad), 0, 1)[:, 0:3]
        pc_mesh = pyrender.Mesh.from_points(pc_red_F, colors=pc_red_F*0.0)



        #get the pmat flat mesh
        _, pmat_mesh_est = self.convert_pimgarr_to_mesh(pmat_est, transform_dict['smpl_pmat_est'])
        _, pmat_mesh_GT = self.convert_pimgarr_to_mesh(pmat, transform_dict['smpl_pmat_gt'])



        #get the SMPL mesh and trimesh
        smpl_trimesh_list_est, smpl_mesh_list_est = self.convert_smplarr_to_mesh(smpl_verts, smpl_faces, material=self.human_mat, transform=transform_dict['smpl_pmat_est'])
        smpl_trimesh_list_GT, smpl_mesh_list_GT = self.convert_smplarr_to_mesh(smpl_verts_gt, smpl_faces, material=self.human_mat_GT, transform=transform_dict['smpl_pmat_gt'])



        #smpl_verts_C = np.swapaxes(np.matmul(transform_dict['C'], smpl_verts_quad), 0, 1)[:, 0:3]
        #smpl_verts_D = np.swapaxes(np.matmul(transform_dict['D'], smpl_verts_quad), 0, 1)[:, 0:3]



        #get the projections
        smpl_trimesh_list_pimg, smpl_mesh_list_pimg, vertex_pressure_norm_est = self.get_smpl_pressure_proj(smpl_trimesh_list_est, pmat_est,
                                                                                                            transform_prev=transform_dict['smpl_pmat_est'],
                                                                                                            transform=transform_dict['pressure_proj_est'])
        RESULTS_DICT['vertex_pressure_list_EST'].append(vertex_pressure_norm_est)




        smpl_trimesh_list_pimg_GT, smpl_mesh_list_pimg_GT, vertex_pressure_norm_GT = self.get_smpl_pressure_proj(smpl_trimesh_list_GT, pmat,
                                                                                                            transform_prev=transform_dict['smpl_pmat_gt'],
                                                                                                            transform=transform_dict['pressure_proj_gt'])
        RESULTS_DICT['vertex_pressure_list_GT'].append(vertex_pressure_norm_GT)








        RESULTS_DICT['vertex_pressure_list_abs_err'].append(list(np.abs(np.array(RESULTS_DICT['vertex_pressure_list_EST'][-1]) - np.array(RESULTS_DICT['vertex_pressure_list_GT'][-1]) )))
        RESULTS_DICT['vertex_pressure_list_sq_err'].append(list(np.square(np.array(RESULTS_DICT['vertex_pressure_list_EST'][-1]) - np.array(RESULTS_DICT['vertex_pressure_list_GT'][-1]) )))

        print("v2vP error, mmHg squared", np.mean( RESULTS_DICT['vertex_pressure_list_sq_err'][-1]))
        print("v2vP error, kPa squared", 133.32 * 133.32 * (1 / 1000000) * np.mean( RESULTS_DICT['vertex_pressure_list_sq_err'][-1]))









        if self.render == True:

            if self.first_pass == True:
                for mesh_part in smpl_mesh_list_est:
                    self.scene.add(mesh_part)

                for mesh_part_pimg in smpl_mesh_list_pimg:
                    self.scene.add(mesh_part_pimg)


                if smpl_verts_gt is not None:
                    for mesh_part_GT in smpl_mesh_list_GT:
                        self.scene.add(mesh_part_GT)

                    for mesh_part_pimg_GT in smpl_mesh_list_pimg_GT:
                        self.scene.add(mesh_part_pimg_GT)


                if pc_mesh is not None:
                    self.scene.add(pc_mesh)

                if pmat_mesh_est is not None:
                    self.scene.add(pmat_mesh_est)

                if pmat_mesh_GT is not None:
                    self.scene.add(pmat_mesh_GT)


                if target_mesh is not None:
                    self.scene.add(target_mesh)

                if score_mesh is not None:
                    self.scene.add(score_mesh)

                lighting_intensity = 20.

                self.viewer = pyrender.Viewer(self.scene, use_raymond_lighting=True, lighting_intensity=lighting_intensity,
                                              point_size=2, run_in_thread=True, viewport_size=(1200, 1200))

                self.first_pass = False

                self.node_list = []
                for mesh_part in smpl_mesh_list_est:
                    for node in self.scene.get_nodes(obj=mesh_part):
                        self.node_list.append(node)

                self.node_list_GT = []
                for mesh_part_GT in smpl_mesh_list_GT:
                    for node in self.scene.get_nodes(obj=mesh_part_GT):
                        self.node_list_GT.append(node)

                self.node_list_pimg_GT = []
                for mesh_part_pimg_GT in smpl_mesh_list_pimg_GT:
                    for node in self.scene.get_nodes(obj=mesh_part_pimg_GT):
                        self.node_list_pimg_GT.append(node)

                self.node_list_pimg = []
                for mesh_part_pimg in smpl_mesh_list_pimg:
                    for node in self.scene.get_nodes(obj=mesh_part_pimg):
                        self.node_list_pimg.append(node)

                if pc_mesh is not None:
                    for node in self.scene.get_nodes(obj=pc_mesh):
                        self.pc_node = node


                if pmat_mesh_est is not None:
                    for node in self.scene.get_nodes(obj=pmat_mesh_est):
                        self.pmat_node = node
                if pmat_mesh_GT is not None:
                    for node in self.scene.get_nodes(obj=pmat_mesh_GT):
                        self.pmat_node2 = node
                if target_mesh is not None:
                    for node in self.scene.get_nodes(obj=target_mesh):
                        self.target_node = node
                if score_mesh is not None:
                    for node in self.scene.get_nodes(obj=score_mesh):
                        self.score_node = node

            else:
                self.viewer.render_lock.acquire()

                #reset the inferred human mesh
                for idx in range(len(smpl_mesh_list_est)):
                    self.scene.remove_node(self.node_list[idx])
                    self.scene.add(smpl_mesh_list_est[idx])
                    for node in self.scene.get_nodes(obj=smpl_mesh_list_est[idx]):
                        self.node_list[idx] = node

                #reset the ground truth human mesh
                for idx in range(len(smpl_mesh_list_GT)):
                    self.scene.remove_node(self.node_list_GT[idx])
                    self.scene.add(smpl_mesh_list_GT[idx])
                    for node in self.scene.get_nodes(obj=smpl_mesh_list_GT[idx]):
                        self.node_list_GT[idx] = node

                #reset the inferred human mesh with pressure projection
                for idx in range(len(smpl_mesh_list_pimg)):
                    self.scene.remove_node(self.node_list_pimg[idx])
                    self.scene.add(smpl_mesh_list_pimg[idx])
                    for node in self.scene.get_nodes(obj=smpl_mesh_list_pimg[idx]):
                        self.node_list_pimg[idx] = node


                #reset the ground truth human mesh with ground truth pressure projection
                for idx in range(len(smpl_mesh_list_pimg_GT)):
                    self.scene.remove_node(self.node_list_pimg_GT[idx])
                    self.scene.add(smpl_mesh_list_pimg_GT[idx])
                    for node in self.scene.get_nodes(obj=smpl_mesh_list_pimg_GT[idx]):
                        self.node_list_pimg_GT[idx] = node


                #reset the point cloud mesh for mesherr
                if pc_mesh is not None:
                    self.scene.remove_node(self.pc_node)
                    self.scene.add(pc_mesh)
                    for node in self.scene.get_nodes(obj=pc_mesh):
                        self.pc_node = node



                #reset the pmat mesh
                if pmat_mesh_est is not None:
                    self.scene.remove_node(self.pmat_node)
                    self.scene.add(pmat_mesh_est)
                    for node in self.scene.get_nodes(obj=pmat_mesh_est):
                        self.pmat_node = node

                #reset the pmat mesh
                if pmat_mesh_GT is not None:
                    self.scene.remove_node(self.pmat_node2)
                    self.scene.add(pmat_mesh_GT)
                    for node in self.scene.get_nodes(obj=pmat_mesh_GT):
                        self.pmat_node2 = node

                #reset the pmat mesh
                if target_mesh is not None:
                    self.scene.remove_node(self.target_node)
                    self.scene.add(target_mesh)
                    for node in self.scene.get_nodes(obj=target_mesh):
                        self.target_node = node

                #reset the pmat mesh
                if score_mesh is not None:
                    self.scene.remove_node(self.score_node)
                    self.scene.add(score_mesh)
                    for node in self.scene.get_nodes(obj=score_mesh):
                        self.score_node = node

                self.viewer.render_lock.release()


        return RESULTS_DICT
