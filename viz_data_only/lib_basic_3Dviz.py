
try:
    import open3d as o3d
except:
    print("COULD NOT IMPORT 03D")
import trimesh
import pyrender
from scipy import ndimage

import numpy as np
import random
import copy


from time import sleep

import math
from random import shuffle
import torch
import torch.nn as nn


#MISC
import time as time
import matplotlib.pyplot as plt
import matplotlib.cm as cm #use cm.jet(list)

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




class pyRenderMesh():
    def __init__(self, render):

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

            self.artag_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.3, 1.0, 0.3, 0.5])
            self.artag_mat_other = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.1, 0.1, 0.1, 0.0])
            #self.artag_r = np.array([[-0.055, -0.055, 0.0], [-0.055, 0.055, 0.0], [0.055, -0.055, 0.0], [0.055, 0.055, 0.0]])
            self.artag_r = np.array([[0.0 + self.y_bump, 0.0 + self.x_bump, 0.075], [0.0286*64*1.04 + self.y_bump, 0.0 + self.x_bump, 0.075], [0.0 + self.y_bump, 0.01 + self.x_bump, 0.075], [0.0286*64*1.04 + self.y_bump, 0.01 + self.x_bump, 0.075],
                                     [0.0 + self.y_bump, 0.0 + self.x_bump, 0.075], [0.0 + self.y_bump, 0.0286*27 + self.x_bump, 0.075], [0.01 + self.y_bump, 0.0 + self.x_bump, 0.075], [0.01 + self.y_bump, 0.0286*27 + self.x_bump, 0.075],
                                     [0.0 + self.y_bump,  0.0286*27 + self.x_bump, 0.075], [0.0286*64*1.04 + self.y_bump, 0.0286*27 + self.x_bump, 0.075], [0.0 + self.y_bump,  0.0286*27+0.01 + self.x_bump, 0.075], [0.0286*64*1.04 + self.y_bump,  0.0286*27+0.01 + self.x_bump, 0.075],
                                     [0.0286*64*1.04 + self.y_bump, 0.0 + self.x_bump, 0.075], [0.0286*64*1.04 + self.y_bump, 0.0286*27 + self.x_bump, 0.075], [0.0286*64*1.04-0.01 + self.y_bump, 0.0 + self.x_bump, 0.075], [0.0286*64*1.04-0.01 + self.y_bump, 0.0286*27 + self.x_bump, 0.075],
                                     ])
            self.artag_f = np.array([[0, 1, 2], [0, 2, 1], [1, 2, 3], [1, 3, 2],
                                     [4, 5, 6], [4, 6, 5], [5, 6, 7], [5, 7, 6],
                                     [8, 9, 10], [8, 10, 9], [9, 10, 11], [9, 11, 10],
                                     [12, 13, 14], [12, 14, 13], [13, 14, 15], [13, 15, 14]])
            self.artag_facecolors_root =  np.array([[0.3, 0.3, 0.0],[0.3, 0.3, 0.0],[0.3, 0.3, 0.0],[0.3, 0.3, 0.0],
                                                    [0.3, 0.3, 0.0],[0.3, 0.3, 0.0],[0.3, 0.3, 0.0],[0.3, 0.3, 0.0],
                                                    [0.3, 0.3, 0.0],[0.3, 0.3, 0.0],[0.3, 0.3, 0.0],[0.3, 0.3, 0.0],
                                                    [0.3, 0.3, 0.0],[0.3, 0.3, 0.0],[0.3, 0.3, 0.0],[0.3, 0.3, 0.0],
                                                    ])*0.0
            self.artag_facecolors = np.copy(self.artag_facecolors_root)


        self.pic_num = 0
        self.tr_ck_arr = np.zeros((13776, 20))


    def get_3D_pmat_markers(self, pmat, angle = 60.0):

        pmat_reshaped = pmat.reshape(64, 27)
        pmat_colors = cm.jet(pmat_reshaped/30)
        pmat_colors[:, :, 3] = 0.7 #translucency

        pmat_xyz = np.zeros((65, 28, 3))
        pmat_faces = []
        pmat_facecolors = []

        for j in range(65):
            for i in range(28):

                pmat_xyz[j, i, 1] = i * 0.0286 + self.x_bump# /1.06# * 1.02 #1.0926 - 0.02
                pmat_xyz[j, i, 0] = ((64 - j) * 0.0286) * 1.04 + self.y_bump#/1.04#1.1406 + 0.05 #only adjusts pmat NOT the SMPL person
                pmat_xyz[j, i, 2] = 0.075#0.12 + 0.075

                if j < 64 and i < 27:
                    coord1 = j * 28 + i
                    coord2 = j * 28 + i + 1
                    coord3 = (j + 1) * 28 + i
                    coord4 = (j + 1) * 28 + i + 1

                    pmat_faces.append([coord1, coord2, coord3]) #bottom surface
                    pmat_faces.append([coord1, coord3, coord2]) #top surface
                    pmat_faces.append([coord4, coord3, coord2]) #bottom surface
                    pmat_faces.append([coord2, coord3, coord4]) #top surface
                    pmat_facecolors.append(pmat_colors[j, i, :])
                    pmat_facecolors.append(pmat_colors[j, i, :])
                    pmat_facecolors.append(pmat_colors[j, i, :])
                    pmat_facecolors.append(pmat_colors[j, i, :])

        pmat_verts = list((pmat_xyz).reshape(1820, 3))
        return pmat_verts, pmat_faces, pmat_facecolors




    def downspl_pc_get_normals(self, pc, camera_point):

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        #print("Downsample the point cloud with a voxel of 0.01")
        downpcd = o3d.geometry.PointCloud.voxel_down_sample(pcd, voxel_size=0.01)

        o3d.geometry.PointCloud.estimate_normals(
            downpcd,
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05,
                                                              max_nn=30))

        o3d.geometry.PointCloud.orient_normals_towards_camera_location(downpcd, camera_location=np.array(camera_point))

        points = np.array(downpcd.points)
        normals = np.array(downpcd.normals)

        return points, normals



    def check_if_under_everything1(self,all_vert_idx, vertices_pimg, faces_pimg):

        coinciding_verts = self.tr_ck_arr[:, 13].nonzero()

        p1_verts = vertices_pimg[faces_pimg[coinciding_verts,0], 0:3][0]
        p2_verts = vertices_pimg[faces_pimg[coinciding_verts,1], 0:3][0]
        p3_verts = vertices_pimg[faces_pimg[coinciding_verts,2], 0:3][0]

        pl_ck_arr = np.concatenate((p1_verts, p2_verts, p3_verts), axis = 1)
        pl_ck_arr = np.concatenate((np.zeros((pl_ck_arr.shape[0],3)), pl_ck_arr, np.zeros((pl_ck_arr.shape[0],12))), axis = 1)
        pl_ck_arr[:, 0] = vertices_pimg[all_vert_idx][0]
        pl_ck_arr[:, 1] = vertices_pimg[all_vert_idx][1]
        pl_ck_arr[:, 2] = vertices_pimg[all_vert_idx][2]

        plane_check_list = []
        for k in range(np.shape(pl_ck_arr)[0]):

            n = np.cross((pl_ck_arr[k, 6:9] - pl_ck_arr[k, 3:6]), (pl_ck_arr[k, 9:12] - pl_ck_arr[k, 3:6]) )
            if n[2] > 0:
                n = np.cross((pl_ck_arr[k, 9:12] - pl_ck_arr[k, 3:6]),(pl_ck_arr[k, 6:9] - pl_ck_arr[k, 3:6]) )
            is_on_side = np.dot((pl_ck_arr[k, 0:3] - pl_ck_arr[k, 3:6]), n  )
            plane_check_list.append(is_on_side)

        plane_check_list = np.array(plane_check_list)
        plane_check_list[plane_check_list >= 0] = 1
        plane_check_list[plane_check_list < 0] = 0

        is_under_everything = np.all(plane_check_list == 0)

        return is_under_everything




    def check_vertex(self,all_vert_idx, vertices_pimg, faces_pimg):

        #here we check if its within the 2D triangle
        self.tr_ck_arr[:, 0] = vertices_pimg[all_vert_idx][0]
        self.tr_ck_arr[:, 1] = vertices_pimg[all_vert_idx][1]
        self.tr_ck_arr[:, 2:4] = vertices_pimg[faces_pimg[:,0], 0:2]
        self.tr_ck_arr[:, 4:6] = vertices_pimg[faces_pimg[:,1], 0:2]
        self.tr_ck_arr[:, 6:8] = vertices_pimg[faces_pimg[:,2], 0:2]
        
        self.tr_ck_arr[:, 8] = (self.tr_ck_arr[:, 5]-self.tr_ck_arr[:, 3])*(self.tr_ck_arr[:, 0]-self.tr_ck_arr[:, 2]) + (-self.tr_ck_arr[:, 4]+self.tr_ck_arr[:, 2])*(self.tr_ck_arr[:, 1]-self.tr_ck_arr[:, 3])
        self.tr_ck_arr[:, 9] = (self.tr_ck_arr[:, 7]-self.tr_ck_arr[:, 5])*(self.tr_ck_arr[:, 0]-self.tr_ck_arr[:, 4]) + (-self.tr_ck_arr[:, 6]+self.tr_ck_arr[:, 4])*(self.tr_ck_arr[:, 1]-self.tr_ck_arr[:, 5])
        self.tr_ck_arr[:, 10] = (self.tr_ck_arr[:, 3]-self.tr_ck_arr[:, 7])*(self.tr_ck_arr[:, 0]-self.tr_ck_arr[:, 6]) + (-self.tr_ck_arr[:, 2]+self.tr_ck_arr[:, 6])*(self.tr_ck_arr[:, 1]-self.tr_ck_arr[:, 7])
        self.tr_ck_arr[self.tr_ck_arr[:, 8] >= 0, 8] = 1
        self.tr_ck_arr[self.tr_ck_arr[:, 9] >= 0, 9] = 1
        self.tr_ck_arr[self.tr_ck_arr[:, 10] >= 0, 10] = 1
        self.tr_ck_arr[self.tr_ck_arr[:, 8] < 0, 8] = 0
        self.tr_ck_arr[self.tr_ck_arr[:, 9] < 0, 9] = 0
        self.tr_ck_arr[self.tr_ck_arr[:, 10] < 0, 10] = 0
        self.tr_ck_arr[:, 11] = self.tr_ck_arr[:, 8]*self.tr_ck_arr[:, 9]*self.tr_ck_arr[:, 10]
        self.tr_ck_arr[:, 12] = self.tr_ck_arr[:, 8]+self.tr_ck_arr[:, 9]+self.tr_ck_arr[:, 10]
        self.tr_ck_arr[self.tr_ck_arr[:, 12] > 0, 12] = 1
        self.tr_ck_arr[:, 12] = 1-self.tr_ck_arr[:, 12]
        self.tr_ck_arr[:, 13] = self.tr_ck_arr[:, 12] + self.tr_ck_arr[:, 11]

        num_passing_faces_before_overlap3 = int(np.sum(self.tr_ck_arr[:, 13]))
        if num_passing_faces_before_overlap3 >= 2:
            is_under_everything = self.check_if_under_everything1(all_vert_idx, vertices_pimg, faces_pimg)
            if is_under_everything == 1:
                cancel_point = 0
            elif is_under_everything == 0:
                cancel_point = 1
        else:
            cancel_point = 0

        return cancel_point


    def render_mesh_pc_bed_pyrender_everything(self, smpl_verts, smpl_faces, camera_point, bedangle,
                                    smpl_verts_gt = None, pc = None, pmat = None, pmat_est = None, smpl_render_points = False, markers = None,
                                    dropout_variance=None, targets=None, scores = None):


        #pmat_est[pmat_est > 0] += 4.
        #pmat[pmat > 0] += 4.


        if smpl_verts_gt is not None:
            smpl_verts_quad_GT = np.concatenate((smpl_verts_gt, np.ones((smpl_verts_gt.shape[0], 1))), axis = 1)
            smpl_verts_quad_GT = np.swapaxes(smpl_verts_quad_GT, 0, 1)


        transform_A = np.identity(4)

        transform_B = np.identity(4)
        transform_B[1, 3] = 1.0 #move things over
        if smpl_verts_gt is not None:
            smpl_verts_B_GT = np.swapaxes(np.matmul(transform_B, smpl_verts_quad_GT), 0, 1)[:, 0:3] #gt over pressure mat

        transform_E = np.identity(4)
        transform_E[1, 3] = -1.0 #move things over

        transform_F = np.identity(4)
        transform_F[1, 3] = 1.0 #move things over



        if targets is not None:
            targets = targets.numpy()
            targets = targets.reshape(int(targets.shape[0]/3), 3)
            targets = np.concatenate((targets[:, 1:2], targets[:, 0:1], -targets[:, 2:3]), axis = 1)
            target_pts_quad = np.swapaxes(np.concatenate((targets, np.ones((targets.shape[0], 1))), axis = 1), 0, 1)
            target_pts_B = np.swapaxes(np.matmul(transform_F, target_pts_quad), 0, 1)[:, 0:3]
            sm = trimesh.creation.uv_sphere(radius=0.05)
            sm.visual.vertex_colors = [0.15, 0.0, 0.0]
            tfs = np.tile(np.eye(4), (len(target_pts_B), 1, 1))
            tfs[:, :3, 3] = target_pts_B
            target_trimesh = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        else:
            target_trimesh = None


        if scores is not None:
            scores = scores.numpy()
            scores = np.concatenate((scores[:, 1:2], scores[:, 0:1], -scores[:, 2:3]), axis = 1)
            score_pts_quad = np.swapaxes(np.concatenate((scores, np.ones((scores.shape[0], 1))), axis = 1), 0, 1)
            score_pts_B = np.swapaxes(np.matmul(transform_A, score_pts_quad), 0, 1)[:, 0:3]
            sm = trimesh.creation.uv_sphere(radius=0.05)
            sm.visual.vertex_colors = [0.15, 0.15, 0.0]
            tfs = np.tile(np.eye(4), (len(score_pts_B), 1, 1))
            tfs[:, :3, 3] = score_pts_B
            score_trimesh = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        else:
            score_trimesh = None


        from matplotlib import cm

        if pc is not None:
            pc_red, pc_red_norm = self.downspl_pc_get_normals(pc, camera_point)
            pc_red_quad = np.swapaxes(np.concatenate((pc_red, np.ones((pc_red.shape[0], 1))), axis = 1), 0, 1)
            pc_red_F = np.swapaxes(np.matmul(transform_F, pc_red_quad), 0, 1)[:, 0:3]



        #GET MESH WITH PMAT
        tm_curr = trimesh.base.Trimesh(vertices=np.array(smpl_verts), faces = np.array(smpl_faces), process = False)
        tm_list = [tm_curr]

        if smpl_verts_gt is not None:
            #GET GT LIMBS WITH PMAT
            tm_curr_GT = trimesh.base.Trimesh(vertices=np.array(smpl_verts_B_GT), faces = np.array(smpl_faces), process = False)
            tm_list_GT = [tm_curr_GT]



        #GET PIMG COLORED TRIMESH
        if pmat is not None and self.render == True:
            pmat_verts, pmat_faces, pmat_facecolors = self.get_3D_pmat_markers(pmat_est, bedangle)
            pmat_tm = trimesh.base.Trimesh(vertices=pmat_verts, faces=pmat_faces, face_colors=pmat_facecolors)
            pmat_mesh = pyrender.Mesh.from_trimesh(pmat_tm, smooth=False)

        else:
            pmat_mesh = None


        vertex_normals_pimg = np.array(tm_list[0].vertex_normals)
        vertices_pimg = np.array(tm_list[0].vertices)
        faces_pimg = np.array(tm_list[0].faces)

        vertices_pimg[:, 0] = vertices_pimg[:, 0] + transform_E[0, 3] #-1.0
        vertices_pimg[:, 1] = vertices_pimg[:, 1] + transform_E[1, 3] #-1.0


        vertex_pressure_init_list_EST = []
        for all_vert_idx in range(vertices_pimg.shape[0]):
            #convert the vertex to 64x27 coords. pick color based on indexing 2d pressure map
            color_idx_y = int(64 - (vertices_pimg[all_vert_idx, 0] - transform_E[0, 3] - self.y_bump)/ (0.0286*1.04) )# - 2.0) + 0.035
            color_idx_x = int( (vertices_pimg[all_vert_idx, 1] - transform_E[1, 3] - self.x_bump)/ (0.0286) )#+ 0.5)# + 35.5 + 0.012
            try:
                if vertex_normals_pimg[all_vert_idx, 2] > 0:
                    cancel_pressure = self.check_vertex(all_vert_idx, vertices_pimg, faces_pimg)
                    if cancel_pressure == 1:
                        vertex_pressure_init_list_EST.append(0.0)
                    else:
                        vertex_pressure_init_list_EST.append(pmat_est[color_idx_y, color_idx_x])
                else:
                    vertex_pressure_init_list_EST.append(0.0)
            except:
                vertex_pressure_init_list_EST.append(0.0)


        verts_color_error = np.array(vertex_pressure_init_list_EST) /30.
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
        tm_list_pimg =[tm_curr]



        mesh_list = []
        mesh_list.append(pyrender.Mesh.from_trimesh(tm_list[0], material = self.human_mat, wireframe = True)) #this is for the main human

        if smpl_verts_gt is not None:
            mesh_list_GT = []
            mesh_list_GT.append(pyrender.Mesh.from_trimesh(tm_list_GT[0], material=self.human_mat_GT, wireframe=True))


        mesh_list_pimg = []
        mesh_list_pimg.append(pyrender.Mesh.from_trimesh(tm_list_pimg[0], smooth=False))
        mesh_list_pimg.append(pyrender.Mesh.from_trimesh(tm_list_pimg[0], material = self.human_mat, wireframe = True))

        if pc is not None:
            pc_greysc_color2 = 0.0 * (pc_red_F[:, 2:3] - np.max(pc_red_F[:, 2])) / (np.min(pc_red_F[:, 2]) - np.max(pc_red_F[:, 2]))
            pc_mesh_mesherr2 = pyrender.Mesh.from_points(pc_red_F, colors=np.concatenate((pc_greysc_color2, pc_greysc_color2, pc_greysc_color2), axis=1))
        else:
            pc_mesh_mesherr2 = None


        if self.render == True:
            if smpl_render_points == True:
                verts_idx_red = np.unique(human_mesh_face_all_red[0])

                verts_red = smpl_verts[verts_idx_red, :]
                smpl_pc_mesh = pyrender.Mesh.from_points(verts_red, colors = [5.0, 0.0, 0.0])
            else: smpl_pc_mesh = None



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
                            artag_tm = trimesh.base.Trimesh(vertices=self.artag_r+marker-markers[2], faces=self.artag_f, face_colors = self.artag_facecolors_root)
                            artag_meshes.append(pyrender.Mesh.from_trimesh(artag_tm, smooth = False))
                        else:
                            artag_tm = trimesh.base.Trimesh(vertices=self.artag_r+marker-markers[2], faces=self.artag_f, face_colors = self.artag_facecolors)
                            artag_meshes.append(pyrender.Mesh.from_trimesh(artag_tm, smooth = False))



            #print "Viewing"
            if self.first_pass == True:
                for mesh_part in mesh_list:
                    print(mesh_part, 'adding meshpart')
                    self.scene.add(mesh_part)

                if smpl_verts_gt is not None:
                    for mesh_part_GT in mesh_list_GT:
                        self.scene.add(mesh_part_GT)

                for mesh_part_pimg in mesh_list_pimg:
                    self.scene.add(mesh_part_pimg)


                if pc_mesh_mesherr2 is not None:
                    self.scene.add(pc_mesh_mesherr2)

                if pmat_mesh is not None:
                    self.scene.add(pmat_mesh)

                if smpl_pc_mesh is not None:
                    self.scene.add(smpl_pc_mesh)

                for artag_mesh in artag_meshes:
                    if artag_mesh is not None:
                        self.scene.add(artag_mesh)

                if target_trimesh is not None:
                    self.scene.add(target_trimesh)

                if score_trimesh is not None:
                    self.scene.add(score_trimesh)


                lighting_intensity = 20.

                self.viewer = pyrender.Viewer(self.scene, use_raymond_lighting=True, lighting_intensity=lighting_intensity,
                                              point_size=2, run_in_thread=True, viewport_size=(1200, 1200))

                self.first_pass = False

                self.node_list = []
                for mesh_part in mesh_list:
                    for node in self.scene.get_nodes(obj=mesh_part):
                        self.node_list.append(node)

                self.node_list_GT = []
                for mesh_part_GT in mesh_list_GT:
                    for node in self.scene.get_nodes(obj=mesh_part_GT):
                        self.node_list_GT.append(node)

                self.node_list_pimg = []
                for mesh_part_pimg in mesh_list_pimg:
                    for node in self.scene.get_nodes(obj=mesh_part_pimg):
                        self.node_list_pimg.append(node)


                if pc_mesh_mesherr2 is not None:
                    for node in self.scene.get_nodes(obj=pc_mesh_mesherr2):
                        self.point_cloud_node_mesherr2 = node

                if smpl_pc_mesh is not None:
                    for node in self.scene.get_nodes(obj=smpl_pc_mesh):
                        self.smpl_pc_mesh_node = node

                self.artag_nodes = []
                for artag_mesh in artag_meshes:
                    if artag_mesh is not None:
                        for node in self.scene.get_nodes(obj=artag_mesh):
                            self.artag_nodes.append(node)
                if pmat_mesh is not None:
                    for node in self.scene.get_nodes(obj=pmat_mesh):
                        self.pmat_node = node
                if target_trimesh is not None:
                    for node in self.scene.get_nodes(obj=target_trimesh):
                        self.target_node = node
                if score_trimesh is not None:
                    for node in self.scene.get_nodes(obj=score_trimesh):
                        self.score_node = node


            else:
                self.viewer.render_lock.acquire()


                #reset the human mesh
                for idx in range(len(mesh_list)):
                    self.scene.remove_node(self.node_list[idx])
                    self.scene.add(mesh_list[idx])
                    for node in self.scene.get_nodes(obj=mesh_list[idx]):
                        self.node_list[idx] = node

                #reset the human mesh
                for idx in range(len(mesh_list_GT)):
                    self.scene.remove_node(self.node_list_GT[idx])
                    self.scene.add(mesh_list_GT[idx])
                    for node in self.scene.get_nodes(obj=mesh_list_GT[idx]):
                        self.node_list_GT[idx] = node


                #reset the pc error human rendering
                for idx in range(len(mesh_list_pimg)):
                    self.scene.remove_node(self.node_list_pimg[idx])
                    self.scene.add(mesh_list_pimg[idx])
                    for node in self.scene.get_nodes(obj=mesh_list_pimg[idx]):
                        self.node_list_pimg[idx] = node




                #reset the point cloud
                if pc_mesh_mesherr2 is not None:
                    self.scene.remove_node(self.point_cloud_node_mesherr2)
                    self.scene.add(pc_mesh_mesherr2)
                    for node in self.scene.get_nodes(obj=pc_mesh_mesherr2):
                        self.point_cloud_node_mesherr2 = node

                #reset the vert pc mesh
                if smpl_pc_mesh is not None:
                    self.scene.remove_node(self.smpl_pc_mesh_node)
                    self.scene.add(smpl_pc_mesh)
                    for node in self.scene.get_nodes(obj=smpl_pc_mesh):
                        self.smpl_pc_mesh_node = node


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


                #reset the pmat mesh
                if pmat_mesh is not None:
                    self.scene.remove_node(self.pmat_node)
                    self.scene.add(pmat_mesh)
                    for node in self.scene.get_nodes(obj=pmat_mesh):
                        self.pmat_node = node

                #reset the pmat mesh
                if target_trimesh is not None:
                    self.scene.remove_node(self.target_node)
                    self.scene.add(target_trimesh)
                    for node in self.scene.get_nodes(obj=target_trimesh):
                        self.target_node = node

                #reset the pmat mesh
                if score_trimesh is not None:
                    self.scene.remove_node(self.score_node)
                    self.scene.add(score_trimesh)
                    for node in self.scene.get_nodes(obj=score_trimesh):
                        self.score_node = node


                self.viewer.render_lock.release()


