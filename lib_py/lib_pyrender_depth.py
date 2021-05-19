
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

            self.artag_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.3, 1.0, 0.3, 0.5])
            self.artag_mat_other = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.1, 0.1, 0.1, 0.0])
            #self.artag_r = np.array([[-0.055, -0.055, 0.0], [-0.055, 0.055, 0.0], [0.055, -0.055, 0.0], [0.055, 0.055, 0.0]])
            self.artag_r = np.array([[0.0 + self.y_bump, 0.0 + self.x_bump, 0.075], [0.0286*64*1.04 + self.y_bump, 0.0 + self.x_bump, 0.075], [0.0 + self.y_bump, 0.01 + self.x_bump, 0.075], [0.0286*64*1.04 + self.y_bump, 0.01 + self.x_bump, 0.075],
                                     [0.0 + self.y_bump, 0.0 + self.x_bump, 0.075], [0.0 + self.y_bump, 0.0286*27 + self.x_bump, 0.075], [0.01 + self.y_bump, 0.0 + self.x_bump, 0.075], [0.01 + self.y_bump, 0.0286*27 + self.x_bump, 0.075],
                                     [0.0 + self.y_bump,  0.0286*27 + self.x_bump, 0.075], [0.0286*64*1.04 + self.y_bump, 0.0286*27 + self.x_bump, 0.075], [0.0 + self.y_bump,  0.0286*27+0.01 + self.x_bump, 0.075], [0.0286*64*1.04 + self.y_bump,  0.0286*27+0.01 + self.x_bump, 0.075],
                                     [0.0286*64*1.04 + self.y_bump, 0.0 + self.x_bump, 0.075], [0.0286*64*1.04 + self.y_bump, 0.0286*27 + self.x_bump, 0.075], [0.0286*64*1.04-0.01 + self.y_bump, 0.0 + self.x_bump, 0.075], [0.0286*64*1.04-0.01 + self.y_bump, 0.0286*27 + self.x_bump, 0.075],
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
                                                    ])*0.0
            #self.artag_facecolors = np.array([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],])
            self.artag_facecolors = np.copy(self.artag_facecolors_root)


        self.pic_num = 0
        self.tr_ck_arr = np.zeros((13776, 20))


    def get_3D_pmat_markers(self, pmat, angle = 60.0):

        pmat_reshaped = pmat.reshape(64, 27)

        #pmat_colors = cm.jet(pmat_reshaped*100)
        pmat_colors = cm.jet(pmat_reshaped/23)
        #print pmat_colors.shape
        pmat_colors[:, :, 3] = 0.7 #translucency
        #pmat_colors[:, :, 3] = 0.2#0.7 #translucency
        #pmat_colors[:, :, 0] = 0.6
        #pmat_colors[:, :, 1] = 0.6
        #pmat_colors[:, :, 2] = 0.0


        pmat_xyz = np.zeros((65, 28, 3))
        pmat_faces = []
        pmat_facecolors = []

        for j in range(65):
            for i in range(28):

                pmat_xyz[j, i, 1] = i * 0.0286 + self.x_bump# /1.06# * 1.02 #1.0926 - 0.02
                pmat_xyz[j, i, 0] = ((64 - j) * 0.0286) * 1.04 + self.y_bump#/1.04#1.1406 + 0.05 #only adjusts pmat NOT the SMPL person
                pmat_xyz[j, i, 2] = 0.075#0.12 + 0.075
                #if j > 23:
                #    pmat_xyz[j, i, 0] = ((64 - j) * 0.0286 - 0.0286 * 3 * np.sin(np.deg2rad(angle)))*1.04 + 0.15#1.1406 + 0.05
                #    pmat_xyz[j, i, 2] = 0.12 + 0.075
                #    # print marker.pose.position.x, 'x'
                #else:

                #    pmat_xyz[j, i, 0] = ((41) * 0.0286 + (23 - j) * 0.0286 * np.cos(np.deg2rad(angle)) \
                #                        - (0.0286 * 3 * np.sin(np.deg2rad(angle))) * 0.85)*1.04 + 0.15#1.1406 + 0.05
                #    pmat_xyz[j, i, 2] = -((23 - j) * 0.0286 * np.sin(np.deg2rad(angle))) * 0.85 + 0.12 + 0.075
                    # print j, marker.pose.position.z, marker.pose.position.y, 'head'

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

        #print np.min(pmat_faces), np.max(pmat_faces), 'minmax'


        pmat_verts = list((pmat_xyz).reshape(1820, 3))

        #print "len faces: ", len(pmat_faces)
        #print "len verts: ", len(pmat_verts)
        #print len(pmat_faces), len(pmat_facecolors)

        return pmat_verts, pmat_faces, pmat_facecolors



    def reduce_by_cam_dir(self, vertices, faces, camera_point, transform):

        vertices = np.array(vertices)
        faces = np.array(faces)

        #print np.min(vertices[:, 0]), np.max(vertices[:, 0])
        #print np.min(vertices[:, 1]), np.max(vertices[:, 1])
        #print np.min(vertices[:, 2]), np.max(vertices[:, 2])
        #for i in range(vertices.shape[0]):
        #    print vertices[i]

        #print transform

        #kill everything thats hanging off the side of the bed
        vertices[vertices[:, 0] < 0 + transform[0], 2] = 0
        vertices[vertices[:, 0] > (0.0286 * 64  + transform[0])*1.04, 2] = 0
        vertices[vertices[:, 1] < 0 + transform[1], 2] = 0
        vertices[vertices[:, 1] > 0.0286 * 27 + transform[1], 2] = 0

        tri_norm = np.cross(vertices[faces[:, 1], :] - vertices[faces[:, 0], :],
                            vertices[faces[:, 2], :] - vertices[faces[:, 0], :]) #find normal of every mesh triangle


        tri_norm = tri_norm/np.linalg.norm(tri_norm, axis = 1)[:, None] #convert normal to a unit vector

        tri_norm[tri_norm[:, 2] == -1, 2] = 1

        tri_to_cam = camera_point - vertices[faces[:, 0], :] ## triangle to camera vector
        tri_to_cam = tri_to_cam/np.linalg.norm(tri_to_cam, axis = 1)[:, None]

        angle_list = tri_norm[:, 0]*tri_to_cam[:, 0] + tri_norm[:, 1]*tri_to_cam[:, 1] + tri_norm[:, 2]*tri_to_cam[:, 2]
        angle_list = np.arccos(angle_list) * 180 / np.pi



        angle_list = np.array(angle_list)

        #print np.shape(angle_list), 'angle list shape'

        faces = np.array(faces)
        faces_red = faces[angle_list < 90, :]

        return list(faces_red)


    def get_triangle_area_vert_weight(self, verts, faces, verts_idx_red = None):

        #first we need all the triangle areas
        tri_verts = verts[faces, :]
        a = np.linalg.norm(tri_verts[:,0]-tri_verts[:,1], axis = 1)
        b = np.linalg.norm(tri_verts[:,1]-tri_verts[:,2], axis = 1)
        c = np.linalg.norm(tri_verts[:,2]-tri_verts[:,0], axis = 1)
        s = (a+b+c)/2
        A = np.sqrt(s*(s-a)*(s-b)*(s-c))

        #print np.shape(verts), np.shape(faces), np.shape(A), np.mean(A), 'area'

        A = np.swapaxes(np.stack((A, A, A)), 0, 1) #repeat the area for each vert in the triangle
        A = A.flatten()
        faces = np.array(faces).flatten()
        i = np.argsort(faces) #sort the faces and the areas by the face idx
        faces_sorted = faces[i]
        A_sorted = A[i]
        last_face = 0
        area_minilist = []
        area_avg_list = []
        face_sort_list = [] #take the average area for all the trianges surrounding each vert
        for vtx_connect_idx in range(np.shape(faces_sorted)[0]):
            if faces_sorted[vtx_connect_idx] == last_face and vtx_connect_idx != np.shape(faces_sorted)[0]-1:
                area_minilist.append(A_sorted[vtx_connect_idx])
            elif faces_sorted[vtx_connect_idx] > last_face or vtx_connect_idx == np.shape(faces_sorted)[0]-1:
                if len(area_minilist) != 0:
                    area_avg_list.append(np.mean(area_minilist))
                else:
                    area_avg_list.append(0)
                face_sort_list.append(last_face)
                area_minilist = []
                last_face += 1
                if faces_sorted[vtx_connect_idx] == last_face:
                    area_minilist.append(A_sorted[vtx_connect_idx])
                elif faces_sorted[vtx_connect_idx] > last_face:
                    num_tack_on = np.copy(faces_sorted[vtx_connect_idx] - last_face)
                    for i in range(num_tack_on):
                        area_avg_list.append(0)
                        face_sort_list.append(last_face)
                        last_face += 1
                        if faces_sorted[vtx_connect_idx] == last_face:
                            area_minilist.append(A_sorted[vtx_connect_idx])

        #print np.mean(area_avg_list), 'area avg'

        area_avg = np.array(area_avg_list)
        area_avg_red = area_avg[area_avg > 0] #find out how many of the areas correspond to verts facing the camera

        #print np.mean(area_avg_red), 'area avg'
        #print np.sum(area_avg_red), np.sum(area_avg)

        norm_area_avg = area_avg/np.sum(area_avg_red)
        norm_area_avg = norm_area_avg*np.shape(area_avg_red) #multiply by the REDUCED num of verts
        #print norm_area_avg[0:3], np.min(norm_area_avg), np.max(norm_area_avg), np.mean(norm_area_avg), np.sum(norm_area_avg)
        #print norm_area_avg.shape, np.shape(verts_idx_red)

        #print np.shape(verts_idx_red), np.min(verts_idx_red), np.max(verts_idx_red)
        #print np.shape(norm_area_avg), np.min(norm_area_avg), np.max(norm_area_avg)

        if verts_idx_red is not None:
            try:
                norm_area_avg = norm_area_avg[verts_idx_red]
            except:
                norm_area_avg = norm_area_avg[verts_idx_red[:-1]]

        #print norm_area_avg[0:3], np.min(norm_area_avg), np.max(norm_area_avg), np.mean(norm_area_avg), np.sum(norm_area_avg)
        return norm_area_avg


    def get_triangle_norm_to_vert(self, verts, faces, verts_idx_red):

        tri_norm = np.cross(verts[np.array(faces)[:, 1], :] - verts[np.array(faces)[:, 0], :],
                            verts[np.array(faces)[:, 2], :] - verts[np.array(faces)[:, 0], :])

        tri_norm = tri_norm/np.linalg.norm(tri_norm, axis = 1)[:, None] #but this is for every TRIANGLE. need it per vert
        tri_norm = np.stack((tri_norm, tri_norm, tri_norm))
        tri_norm = np.swapaxes(tri_norm, 0, 1)

        tri_norm = tri_norm.reshape(tri_norm.shape[0]*tri_norm.shape[1], tri_norm.shape[2])

        faces = np.array(faces).flatten()

        i = np.argsort(faces) #sort the faces and the areas by the face idx
        faces_sorted = faces[i]

        tri_norm_sorted = tri_norm[i]

        last_face = 0
        face_sort_list = [] #take the average area for all the trianges surrounding each vert
        vertnorm_minilist = []
        vertnorm_avg_list = []

        for vtx_connect_idx in range(np.shape(faces_sorted)[0]):
            if faces_sorted[vtx_connect_idx] == last_face and vtx_connect_idx != np.shape(faces_sorted)[0]-1:
                vertnorm_minilist.append(tri_norm_sorted[vtx_connect_idx])
            elif faces_sorted[vtx_connect_idx] > last_face or vtx_connect_idx == np.shape(faces_sorted)[0]-1:
                if len(vertnorm_minilist) != 0:
                    mean_vertnorm = np.mean(vertnorm_minilist, axis = 0)
                    mean_vertnorm = mean_vertnorm/np.linalg.norm(mean_vertnorm)
                    vertnorm_avg_list.append(mean_vertnorm)
                else:
                    vertnorm_avg_list.append(np.array([0.0, 0.0, 0.0]))
                face_sort_list.append(last_face)
                vertnorm_minilist = []
                last_face += 1
                if faces_sorted[vtx_connect_idx] == last_face:
                    vertnorm_minilist.append(tri_norm_sorted[vtx_connect_idx])
                elif faces_sorted[vtx_connect_idx] > last_face:
                    num_tack_on = np.copy(faces_sorted[vtx_connect_idx] - last_face)
                    for i in range(num_tack_on):
                        vertnorm_avg_list.append([0.0, 0.0, 0.0])
                        face_sort_list.append(last_face)
                        last_face += 1
                        if faces_sorted[vtx_connect_idx] == last_face:
                            vertnorm_minilist.append(tri_norm_sorted[vtx_connect_idx])


        vertnorm_avg = np.array(vertnorm_avg_list)
        vertnorm_avg_red = np.swapaxes(np.stack((vertnorm_avg[vertnorm_avg[:, 0] != 0, 0],
                                                vertnorm_avg[vertnorm_avg[:, 1] != 0, 1],
                                                vertnorm_avg[vertnorm_avg[:, 2] != 0, 2])), 0, 1)
        return vertnorm_avg_red


    def downspl_pc_get_normals(self, pc, camera_point):

        #for i in range(3):
        #    print np.min(pc[:, i]), np.max(pc[:, i])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        #print("Downsample the point cloud with a voxel of 0.01")
        downpcd = o3d.geometry.PointCloud.voxel_down_sample(pcd, voxel_size=0.01)


        #o3d.visualization.draw_geometries([downpcd])

        #print("Recompute the normal of the downsampled point cloud")
        o3d.geometry.PointCloud.estimate_normals(
            downpcd,
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05,
                                                              max_nn=30))

        o3d.geometry.PointCloud.orient_normals_towards_camera_location(downpcd, camera_location=np.array(camera_point))

        #o3d.visualization.draw_geometries([downpcd])

        points = np.array(downpcd.points)
        normals = np.array(downpcd.normals)

        #for i in range(3):
        #    print np.min(points[:, i]), np.max(points[:, i])

        return points, normals


    def pc_get_normals(self, pc, camera_point):

        #for i in range(3):
        #    print np.min(pc[:, i]), np.max(pc[:, i])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)


        #o3d.visualization.draw_geometries([downpcd])

        #print("Recompute the normal of the downsampled point cloud")
        o3d.geometry.PointCloud.estimate_normals(
            pcd,
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05,
                                                              max_nn=30))

        o3d.geometry.PointCloud.orient_normals_towards_camera_location(pcd, camera_location=np.array(camera_point))

        #o3d.visualization.draw_geometries([downpcd])

        points = np.array(pcd.points)
        normals = np.array(pcd.normals)

        #for i in range(3):
        #    print np.min(points[:, i]), np.max(points[:, i])

        return points, normals





    def plot_mesh_norms(self, verts, verts_norm):

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(verts)
        pcd.normals = o3d.utility.Vector3dVector(verts_norm)

        o3d.visualization.draw_geometries([pcd])


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
                #print "got here"
                segmented_dict = load_pickle('../lib_py/segmented_mesh_idx_faces.p')
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


    def compare_pc_to_voxelmesh(self, smpl_verts, smpl_faces, gt_points, pmat, RESULTS_DICT, synth = False):

        #gt_points[:, 2] -= 1.0
        #cut off things that aren't overlaying the bed

        #smpl_verts[smpl_verts[:, 0] < 0, 0] = 0
        #smpl_verts[smpl_verts[:, 0] > (0.0286 * 64)*1.04, 2] = (0.0286 * 64)*1.04
        #smpl_verts[smpl_verts[:, 1] < 0, 1] = 0
        #smpl_verts[smpl_verts[:, 1] > 0.0286 * 27, 1] = 0.0286 * 27

        #VOXELIZE TRIMESH


        pmat = ndimage.zoom(pmat, 2, order=0)

        resolution = 1.1*0.0127 #meters. about a half inch.


        #for i in range(2):
        #    print np.min(smpl_verts[:, i]), np.max(smpl_verts[:, i]), np.max(smpl_verts[:, i]) - np.min(smpl_verts[:, i]), "voxel smpl min max range"
        #    print np.min(smpl_verts[:, i])/resolution, np.max(smpl_verts[:, i])/resolution, (np.max(smpl_verts[:, i]) - np.min(smpl_verts[:, i]))/resolution, "voxel smpl min max range"

        smpl_verts[:, 2] *= (1.1*2.54) #this is for 5 mm orthographic resolution. spatial is still 1/2 inch
        gt_points[:, 2] *= (1.1*2.54)


        tm_curr = trimesh.base.Trimesh(vertices=smpl_verts, faces = smpl_faces)

        v = tm_curr.voxelized(pitch = resolution)
        voxelgrid = np.copy(v.matrix)
        voxelgrid = np.flip(voxelgrid, axis = 2)
        voxelgrid = np.flip(voxelgrid, axis = 0)

        #print voxelgrid.shape
        #print "max index x: ",

        if synth == True:
            tm_curr_gt = trimesh.base.Trimesh(vertices=gt_points, faces=smpl_faces)

            v_gt = tm_curr_gt.voxelized(pitch=resolution)
            voxelgrid_gt = np.copy(v_gt.matrix)
            voxelgrid_gt = np.flip(voxelgrid_gt, axis=2)
            voxelgrid_gt = np.flip(voxelgrid_gt, axis=0)

        #print np.shape(voxelgrid), np.max(smpl_verts[:, 0]) - np.min(smpl_verts[:, 0]), \
        #                            np.max(smpl_verts[:, 1]) - np.min(smpl_verts[:, 1]),\
        #                            np.max(smpl_verts[:, 2]) - np.min(smpl_verts[:, 2])

        pc_smpl_minmax = np.array([[np.min(gt_points[:, 0]), np.min(gt_points[:, 1]), -np.max(gt_points[:, 2])],
                                   [np.max(gt_points[:, 0]), np.max(gt_points[:, 1]), -np.min(gt_points[:, 2])],
                                   [np.min(smpl_verts[:, 0]), np.min(smpl_verts[:, 1]), -np.max(smpl_verts[:, 2])],
                                   [np.max(smpl_verts[:, 0]), np.max(smpl_verts[:, 1]), -np.min(smpl_verts[:, 2])],
                                   [0, 0, 0],
                                   [128*resolution, 54*resolution, 0]])


        pc_smpl_minmax /= resolution
        pc_smpl_minmax[pc_smpl_minmax < 0] -= 0.5
        pc_smpl_minmax[pc_smpl_minmax > 0] += 0.5
        pc_smpl_minmax = pc_smpl_minmax.astype(int)

        pc_smpl_minmax_new_ids = np.copy(pc_smpl_minmax)
        pc_smpl_minmax_new_ids[:, 0] -= np.min(pc_smpl_minmax_new_ids[:, 0])
        pc_smpl_minmax_new_ids[:, 1] -= np.min(pc_smpl_minmax_new_ids[:, 1])


        viz_maps = np.zeros((np.max([pc_smpl_minmax[1,0], pc_smpl_minmax[3,0], pc_smpl_minmax[5,0]])\
                                   -np.min([pc_smpl_minmax[0,0], pc_smpl_minmax[2,0], pc_smpl_minmax[4,0]])+1+0,
                                   np.max([pc_smpl_minmax[1,1], pc_smpl_minmax[3,1], pc_smpl_minmax[5,1]]) \
                                   -np.min([pc_smpl_minmax[0,1], pc_smpl_minmax[2,1], pc_smpl_minmax[4,1]])+1+0,
                                   6)).astype(int)

        if synth == False:
            pc_int_array = gt_points/resolution
            #print pc_int_array
            pc_int_array[pc_int_array < 0] -= 0.5
            pc_int_array[pc_int_array > 0] += 0.5
            pc_int_array = (pc_int_array).astype(int)
            pc_int_array[:, 2] += pc_smpl_minmax[2,2]
            #print pc_int_array

            pc_int_array = np.concatenate((np.zeros((pc_int_array.shape[0], 1)).astype(int), pc_int_array), axis = 1)
            #print pc_int_array
            y_range = np.max(pc_int_array[:, 1])  - np.min(pc_int_array[:, 1])
            x_range = np.max(pc_int_array[:, 2])  - np.min(pc_int_array[:, 2])
            x_min = np.min(pc_int_array[:, 2])
            y_min = np.min(pc_int_array[:, 1])
            filler_array = np.zeros(((y_range+1)*(x_range+1), 4)).astype(int)
            for y in range(y_range+1):
                for x in range(x_range+1):
                    idx = y*(x_range+1) + x
                    filler_array[idx, 1] = y + y_min
                    filler_array[idx, 2] = x + x_min
            #print filler_array[0:100], 'filler'

            #print pc_int_array, np.shape(pc_int_array)

            pc_int_array = np.concatenate((pc_int_array, filler_array), axis = 0)

            pc_int_array[:, 0] = pc_int_array[:, 1]*(x_range+1) + pc_int_array[:, 2]
            #print pc_int_array, np.shape(pc_int_array)

            pc_int_array = pc_int_array[pc_int_array[:, 0].argsort()]
            unique_keys, indices = np.unique(pc_int_array[:, 0], return_index=True)
            pc_int_array = pc_int_array[indices]

            pc_int_array = np.flip(-pc_int_array[:, 3].reshape(y_range+1, x_range+1), axis = 0)

        mesh_height_arr = np.zeros((voxelgrid.shape[0], voxelgrid.shape[1])).astype(int)


        #gt min: pc_smpl_minmax_new_ids[0, 2]
        #mesh min: pc_smpl_minmax_new_ids[2, 2]
        # #if the ground truth is lower then we need to add some to the mesh
        if pc_smpl_minmax_new_ids[0, 2] < pc_smpl_minmax_new_ids[2, 2]:
            add_mesh = pc_smpl_minmax_new_ids[2,2] - pc_smpl_minmax_new_ids[0,2]
        else:
            add_mesh = 0

        #if the mesh is lower we need to add some to the ground truth
        if pc_smpl_minmax_new_ids[2, 2] < pc_smpl_minmax_new_ids[0, 2]:
            add_gt = pc_smpl_minmax_new_ids[0,2] - pc_smpl_minmax_new_ids[2,2]
        else:
            add_gt = 0
        print ("adding to mesh", add_mesh)
        print ("adding to gt", add_gt)

        for i in range(voxelgrid.shape[2]):
            #print mesh_height_arr.shape, voxelgrid[:, :, i].shape
            mesh_height_arr[voxelgrid[:, :, i] == True] = i
        mesh_height_arr[mesh_height_arr != 0] += add_mesh

        if synth == True:
            mesh_height_arr_gt = np.zeros((voxelgrid_gt.shape[0], voxelgrid_gt.shape[1])).astype(int)
            for i in range(voxelgrid_gt.shape[2]):
                # print mesh_height_arr.shape, voxelgrid[:, :, i].shape
                mesh_height_arr_gt[voxelgrid_gt[:, :, i] == True] = i
            mesh_height_arr_gt[mesh_height_arr_gt != 0] += add_gt

        total_L = viz_maps.shape[0]



        #print np.min(mesh_height_arr), np.max(mesh_height_arr)
        #print np.min(mesh_height_arr_gt), np.max(mesh_height_arr_gt)
        #print np.min(pc_int_array), np.max(pc_int_array)


        if synth == False:
            viz_maps[total_L - pc_smpl_minmax_new_ids[1, 0] - 1:total_L - pc_smpl_minmax_new_ids[0, 0],
                     pc_smpl_minmax_new_ids[0, 1]:pc_smpl_minmax_new_ids[1, 1] + 1, 0] = pc_int_array
        else:
            viz_maps[total_L - pc_smpl_minmax_new_ids[1, 0] - 1:total_L - pc_smpl_minmax_new_ids[0, 0],
                     pc_smpl_minmax_new_ids[0, 1]:pc_smpl_minmax_new_ids[1, 1] + 1, 0] = mesh_height_arr_gt



        viz_maps[total_L - pc_smpl_minmax_new_ids[3,0]-1:total_L - pc_smpl_minmax_new_ids[2,0], pc_smpl_minmax_new_ids[2,1]:pc_smpl_minmax_new_ids[3,1]+1, 1] = mesh_height_arr
        viz_maps[total_L - pc_smpl_minmax_new_ids[5,0]:total_L - pc_smpl_minmax_new_ids[4,0], pc_smpl_minmax_new_ids[4,1]:pc_smpl_minmax_new_ids[5,1], 2] = pmat

        viz_maps[viz_maps < 0] = 0

        viz_maps = viz_maps.astype(float)

        if synth == False:
            # fix holes
            ys = viz_maps.shape[0]+2 #66
            xs = viz_maps.shape[1]+2 #29

            abc = np.zeros((ys, xs, 4))
            abc[1:ys-1, 1:xs-1, 0] = np.copy(viz_maps[:, :, 0])
            abc[1:ys-1, 1:xs-1, 0] = abc[0:ys-2, 0:xs-2, 0] + abc[1:ys-1, 0:xs-2, 0] + abc[2:ys, 0:xs-2, 0] + \
                                 abc[0:ys-2, 1:xs-1, 0] + abc[2:ys, 1:xs-1, 0] + \
                                 abc[0:ys-2, 2:xs, 0] + abc[1:ys-1, 2:xs, 0] + abc[2:ys, 2:xs, 0]
            abc[:, :, 0] /= 8


            abc[1:ys-1, 1:xs-1, 1] = np.copy(viz_maps[:, :, 0]) #this makes sure that you only fill if there's 5/8 adjacent filled points
            abc[1:ys - 1, 1:xs - 1, 1][abc[1:ys-1, 1:xs-1, 1] < 0] = 0
            abc[1:ys - 1, 1:xs - 1, 1][abc[1:ys-1, 1:xs-1, 1] > 0] = 1

            abc[1:ys-1, 1:xs-1, 1] = abc[0:ys-2, 0:xs-2, 1] + abc[1:ys-1, 0:xs-2, 1] + abc[2:ys, 0:xs-2, 1] + \
                                 abc[0:ys-2, 1:xs-1, 1] + abc[2:ys, 1:xs-1, 1] + \
                                 abc[0:ys-2, 2:xs, 1] + abc[1:ys-1, 2:xs, 1] + abc[2:ys, 2:xs, 1]

            abc[1:ys - 1, 1:xs - 1, 1][abc[1:ys-1, 1:xs-1, 1] < 5] = 0
            abc[1:ys - 1, 1:xs - 1, 1][abc[1:ys-1, 1:xs-1, 1] >= 5] = 1
            abc[:, :, 0] = abc[:, :, 0]*abc[:, :, 1]


            abc = abc[1:ys-1, 1:xs-1, :]



            abc[:, :, 1] = np.copy(viz_maps[:, :, 0])
            abc[:, :, 1][abc[:, :, 1] > 0] = -1
            abc[:, :, 1][abc[:, :, 1] == 0] = 1
            abc[:, :, 1][abc[:, :, 1] < 0] = 0
            abc[:, :, 2] = abc[:, :, 0] * abc[:, :, 1]
            abc[:, :, 3] = np.copy(abc[:, :, 2])
            abc[:, :, 3][abc[:, :, 3] != 0] = 1.
            abc[:, :, 3] = 1 - abc[:, :, 3]

            viz_maps[:, :, 5] = viz_maps[:, :, 0] * abc[:, :, 3] #now fill in the original depth image
            viz_maps[:, :, 5] += abc[:, :, 2]
        else:
            viz_maps[:, :, 5] = np.copy(viz_maps[:, :, 0])

        viz_maps = np.flip(viz_maps, axis = 0)

        #print viz_maps.shape
        #print viz_maps.shape
        #print pc_smpl_minmax
        side_cutoff_L = -np.min(np.array([pc_smpl_minmax[0, 1], pc_smpl_minmax[2, 1]-1, pc_smpl_minmax[4,1], 0]))
        ud_cutoff_L = -np.min(np.array([pc_smpl_minmax[0, 0], pc_smpl_minmax[2, 0], pc_smpl_minmax[4,0], 0]))

        #print side_cutoff_L, ud_cutoff_L
        viz_maps = viz_maps[ud_cutoff_L:ud_cutoff_L+int((0.0286 * 64)*1.04/resolution + 0.5), side_cutoff_L:side_cutoff_L+int((0.0286 * 27)/resolution + 0.5), :]
        #print viz_maps.shape
        #print int((0.0286 * 64)*1.04/resolution + 0.5)
        #print int((0.0286 * 27)/resolution + 0.5)


        viz_maps = np.flip(viz_maps, axis = 0)

        #print viz_maps.shape

        #get the precision and recall map
        viz_maps[:, :, 3] = np.copy(viz_maps[:, :, 5]) #point cloud
        viz_maps[:, :, 3][viz_maps[:, :, 3] > 0] = 1
        recall_denom = np.sum(viz_maps[:, :, 3])

        viz_maps[:, :, 4] = np.copy(viz_maps[:, :, 1]) #mesh
        viz_maps[:, :, 4][viz_maps[:, :, 4] > 0] = 2
        precision_denom = np.sum(viz_maps[:, :, 4])/2
        viz_maps[:, :, 3] += viz_maps[:, :, 4]


        #now make a depth comparison over everything that overlaps
        viz_maps[:, :, 4] = np.copy(viz_maps[:, :, 3])
        viz_maps[:, :, 4][viz_maps[:, :, 4] < 3] = 0
        viz_maps[:, :, 4] = np.clip(viz_maps[:, :, 4], 0, 1)
        overlapping = np.copy(viz_maps[:, :, 4])

        overlapping_numer = np.sum(overlapping)
        viz_maps[:, :, 4] = np.abs(viz_maps[:, :, 5] - viz_maps[:, :, 1])*overlapping




        precision =  overlapping_numer/precision_denom
        recall = overlapping_numer/recall_denom
        average_err_m_overlap = 0.005*np.sum(viz_maps[:, :, 4])/np.sum(overlapping)
        average_err_m = 0.005*np.sum(np.abs(viz_maps[:, :, 5] - viz_maps[:, :, 1]))/(np.count_nonzero(viz_maps[:, :, 3]))

        print ("Precision is:", precision)
        print ("Recall is:", recall)
        print ("Average error from overlapping:", average_err_m_overlap)
        print ("Average error:", average_err_m)
        RESULTS_DICT['precision'].append(precision)
        RESULTS_DICT['recall'].append(recall)
        RESULTS_DICT['overlap_d_err'].append(average_err_m_overlap)
        RESULTS_DICT['all_d_err'].append(average_err_m)

        if self.render == True:

            if synth == True:
                num_plots = 6
            else:
                num_plots = 7

            fig = plt.figure(figsize = (3*num_plots, 5))
            mngr = plt.get_current_fig_manager()
            ax1 = fig.add_subplot(1, num_plots, 1)
            #ax1.set_xlim([-10.0*p_map_mult, 37.0*p_map_mult])
            #ax1.set_ylim([74.0*p_map_mult, -10.0*p_map_mult])
            #ax1.set_facecolor('cyan')
            ax1.set_title("Pressure Image")
            ax1.imshow(viz_maps[:, :, 2], interpolation='nearest', cmap=plt.cm.jet, origin='upper', vmin=0, vmax=100)

            ax1 = fig.add_subplot(1, num_plots, 2)

            ax1.set_title("Estimated Mesh - \n Orthographic")
            ax1.imshow(viz_maps[:, :, 1], interpolation='nearest', cmap=plt.cm.jet, origin='upper', vmin=0, vmax=100)
            ax1 = fig.add_subplot(1, num_plots, 3)
            #ax1.set_xlim([-10.0*p_map_mult, 37.0*p_map_mult])
            #ax1.set_ylim([74.0*p_map_mult, -10.0*p_map_mult])
            #ax1.set_facecolor('cyan')
            ax1.set_title("Ground Truth - \n Orthographic")
            ax1.imshow(viz_maps[:, :, 0], interpolation='nearest', cmap=plt.cm.jet, origin='upper', vmin=0, vmax=100)

            if synth == False:
                ax1 = fig.add_subplot(1, num_plots, 4)
                #ax1.set_xlim([-10.0*p_map_mult, 37.0*p_map_mult])
                #ax1.set_ylim([74.0*p_map_mult, -10.0*p_map_mult])
                #ax1.set_facecolor('cyan')
                ax1.set_title("Fixed Point Cloud - \n Orthographic")
                ax1.imshow(viz_maps[:, :, 5], interpolation='nearest', cmap=plt.cm.jet, origin='upper', vmin=0, vmax=100)

            ax1 = fig.add_subplot(1, num_plots, num_plots-2)
            ax1.set_title("Precision and Recall")
            ax1.imshow(viz_maps[:, :, 3], interpolation='nearest', cmap=plt.cm.jet, origin='upper', vmin=0, vmax=3)

            ax1 = fig.add_subplot(1, num_plots, num_plots-1)
            ax1.set_title("Depth Error for \n Overlapping GT and est")
            ax1.imshow(viz_maps[:, :, 4], interpolation='nearest', cmap=plt.cm.jet, origin='upper', vmin=0, vmax=50)

            ax1 = fig.add_subplot(1, num_plots, num_plots)
            ax1.set_title("Depth Error for \n all GT and est")
            ax1.imshow(np.abs(viz_maps[:, :, 5] - viz_maps[:, :, 1]), interpolation='nearest', cmap=plt.cm.jet, origin='upper', vmin=0, vmax=100)


            plt.show()

        return RESULTS_DICT

    def check_if_in_triangle(self, p, p1, p2, p3):

        x_orig = p[0]
        y_orig = p[1]
        x1 = p1[0]
        y1 = p1[1]
        x2 = p2[0]
        y2 = p2[1]
        x3 = p3[0]
        y3 = p3[1]

        dot1 = (y2 - y1)*(x_orig - x1) + (-x2 + x1)*(y_orig - y1)

        dot2 = (y3 - y2)*(x_orig - x2) + (-x3 + x2)*(y_orig - y2)

        dot3 = (y1 - y3)*(x_orig - x3) + (-x1 + x3)*(y_orig - y3)

        if dot1 >= 0 and dot2 >= 0 and dot3 >= 0:
            if np.sum(np.abs(p1-p)) == 0 or np.sum(np.abs(p2-p)) == 0 or np.sum(np.abs(p3-p)) == 0:
                return 0
            else:
                return 1
        else:
            return 0



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


    def render_mesh_pc_bed_pyrender_everything(self, smpl_verts, smpl_faces, camera_point, bedangle, RESULTS_DICT,
                                    smpl_verts_gt = None, pc = None, pmat = None, pmat_est = None, smpl_render_points = False, markers = None,
                                    dropout_variance=None, targets=None, scores = None):
        #pmat[pmat>0] += 0.001
        #pmat_est[pmat_est>0] += 0.001

        #pmat_est[pmat_est > 0] += 4.
        #pmat[pmat > 0] += 4.

        print(smpl_verts.shape, smpl_faces.shape)#, smpl_verts_gt.shape)


        smpl_verts_quad = np.concatenate((smpl_verts, np.ones((smpl_verts.shape[0], 1))), axis = 1)
        smpl_verts_quad = np.swapaxes(smpl_verts_quad, 0, 1)

        if smpl_verts_gt is not None:
            smpl_verts_quad_GT = np.concatenate((smpl_verts_gt, np.ones((smpl_verts_gt.shape[0], 1))), axis = 1)
            smpl_verts_quad_GT = np.swapaxes(smpl_verts_quad_GT, 0, 1)


        transform_A = np.identity(4)

        transform_B = np.identity(4)
        transform_B[1, 3] = 1.0 #move things over
        if smpl_verts_gt is not None:
            smpl_verts_B_GT = np.swapaxes(np.matmul(transform_B, smpl_verts_quad_GT), 0, 1)[:, 0:3] #gt over pressure mat
        #print smpl_verts_quad.shape

        transform_C = np.identity(4)
        transform_C[1, 3] = 2.0 #move things over
        smpl_verts_C = np.swapaxes(np.matmul(transform_C, smpl_verts_quad), 0, 1)[:, 0:3]

        transform_D = np.identity(4)
        transform_D[1, 3] = 3.0 #move things over
        smpl_verts_D = np.swapaxes(np.matmul(transform_D, smpl_verts_quad), 0, 1)[:, 0:3]

        transform_E = np.identity(4)
        transform_E[0, 3] = -2.0 #move things over
        transform_E[1, 3] = 0.0 #move things over

        transform_F = np.identity(4)
        transform_F[1, 3] = 1.0 #move things over

        transform_G = np.identity(4)
        transform_G[0, 3] = -2.0 #move things over
        transform_G[1, 3] = 1.0 #move things over

        if targets is not None:
            targets = targets.numpy()
            targets = targets.reshape(int(targets.shape[0]/3), 3)
            #targets[:, 0:2] -= 0.286
            targets = np.concatenate((targets[:, 1:2], targets[:, 0:1], -targets[:, 2:3]), axis = 1)
            target_pts_quad = np.swapaxes(np.concatenate((targets, np.ones((targets.shape[0], 1))), axis = 1), 0, 1)
            target_pts_B = np.swapaxes(np.matmul(transform_F, target_pts_quad), 0, 1)[:, 0:3]
            sm = trimesh.creation.uv_sphere(radius=0.05)
            #sm.visual.vertex_colors = [0.0, 0.15, 0.0]
            sm.visual.vertex_colors = [0.15, 0.0, 0.0]
            tfs = np.tile(np.eye(4), (len(target_pts_B), 1, 1))
            tfs[:, :3, 3] = target_pts_B
            target_trimesh = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        else:
            target_trimesh = None


        if scores is not None:
            scores = scores.numpy()
            #scores = scores.reshape(int(scores.shape[0]/3), 3)
            #scores[:, 0:2] -= 0.286
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

        #downsample the point cloud and get the normals

        #pc_red, pc_red_norm = self.pc_get_normals(pc, camera_point)
        pc_red, pc_red_norm = self.downspl_pc_get_normals(pc, camera_point)

        #RESULTS_DICT['pc_red'] = pc_red*1.
        #RESULTS_DICT['pc_red_norm'] = pc_red_norm*1.


        print(np.shape(pc_red), np.min(pc_red[:, 0]),np.max(pc_red[:, 0]),np.min(pc_red[:, 1]),np.max(pc_red[:, 1]),np.min(pc_red[:, 2]),np.max(pc_red[:, 2]), np.shape(pc_red_norm))




        pc_red_quad = np.swapaxes(np.concatenate((pc_red, np.ones((pc_red.shape[0], 1))), axis = 1), 0, 1)
        pc_red_B = np.swapaxes(np.matmul(transform_B, pc_red_quad), 0, 1)[:, 0:3]
        camera_point_B = np.matmul(transform_B, np.array([camera_point[0], camera_point[1], camera_point[2], 1.0]))[0:3]

        pc_red_quad = np.swapaxes(np.concatenate((pc_red, np.ones((pc_red.shape[0], 1))), axis = 1), 0, 1)
        pc_red_C = np.swapaxes(np.matmul(transform_C, pc_red_quad), 0, 1)[:, 0:3]
        pc_red_norm_tri = np.swapaxes(pc_red_norm, 0, 1)
        pc_red_norm_C = np.swapaxes(np.matmul(transform_C[0:3, 0:3], pc_red_norm_tri), 0, 1)[:, 0:3]
        camera_point_C = np.matmul(transform_C, np.array([camera_point[0], camera_point[1], camera_point[2], 1.0]))[0:3]

        pc_red_quad = np.swapaxes(np.concatenate((pc_red, np.ones((pc_red.shape[0], 1))), axis = 1), 0, 1)
        pc_red_D = np.swapaxes(np.matmul(transform_D, pc_red_quad), 0, 1)[:, 0:3]
        pc_red_norm_tri = np.swapaxes(pc_red_norm, 0, 1)
        pc_red_norm_D = np.swapaxes(np.matmul(transform_D[0:3, 0:3], pc_red_norm_tri), 0, 1)[:, 0:3]
        camera_point_D = np.matmul(transform_D, np.array([camera_point[0], camera_point[1], camera_point[2], 1.0]))[0:3]

        pc_red_quad = np.swapaxes(np.concatenate((pc_red, np.ones((pc_red.shape[0], 1))), axis = 1), 0, 1)
        pc_red_F = np.swapaxes(np.matmul(transform_F, pc_red_quad), 0, 1)[:, 0:3]
        pc_red_norm_tri = np.swapaxes(pc_red_norm, 0, 1)
        pc_red_norm_F = np.swapaxes(np.matmul(transform_F[0:3, 0:3], pc_red_norm_tri), 0, 1)[:, 0:3]
        camera_point_F = np.matmul(transform_F, np.array([camera_point[0], camera_point[1], camera_point[2], 1.0]))[0:3]



        human_mesh_vtx_all, human_mesh_face_all = self.get_human_mesh_parts(smpl_verts, smpl_faces, segment_limbs=False)
        if smpl_verts_gt is not None:
            human_mesh_vtx_all_GT, human_mesh_face_all_GT = self.get_human_mesh_parts(smpl_verts_B_GT, smpl_faces, segment_limbs=False)
        human_mesh_vtx_mesherr, human_mesh_face_mesherr = self.get_human_mesh_parts(smpl_verts_C, smpl_faces, segment_limbs=False)
        human_mesh_vtx_pcerr, human_mesh_face_pcerr = self.get_human_mesh_parts(smpl_verts_D, smpl_faces, segment_limbs=False)



        human_mesh_face_mesherr_red = []
        #only use the vertices that are facing the camera
        for part_idx in range(len(human_mesh_vtx_mesherr)):
            human_mesh_face_mesherr_red.append(self.reduce_by_cam_dir(human_mesh_vtx_mesherr[part_idx], human_mesh_face_mesherr[part_idx], camera_point_C, transform_C[0:3, 3]))


        human_mesh_face_pcerr_red = []
        #only use the vertices that are facing the camera
        for part_idx in range(len(human_mesh_vtx_mesherr)):
            human_mesh_face_pcerr_red.append(self.reduce_by_cam_dir(human_mesh_vtx_pcerr[part_idx], human_mesh_face_pcerr[part_idx], camera_point_D, transform_D[0:3, 3]))


        #GET MESH WITH PMAT
        tm_curr = trimesh.base.Trimesh(vertices=np.array(human_mesh_vtx_all[0]), faces = np.array(human_mesh_face_all[0]), process = False)
        tm_list = [tm_curr]

        if smpl_verts_gt is not None:
            #GET GT LIMBS WITH PMAT
            tm_curr_GT = trimesh.base.Trimesh(vertices=np.array(human_mesh_vtx_all_GT[0]), faces = np.array(human_mesh_face_all_GT[0]), process = False)
            tm_list_GT = [tm_curr_GT]


        #GET MESHERROR
        #overall
        verts_idx_red = np.unique(human_mesh_face_mesherr_red[0])
        verts_red = human_mesh_vtx_mesherr[0][verts_idx_red, :]


        # get the nearest point from each vert to some pc point, regardless of the normal - overall
        vert_to_nearest_point_error_list = []
        for vert_idx in range(verts_red.shape[0]):
            curr_vtx = verts_red[vert_idx, :]
            mesherr_dist = pc_red_C - curr_vtx
            #print np.shape(mesherr_dist)
            mesherr_eucl = np.linalg.norm(mesherr_dist, axis=1)
            #print np.shape(mesherr_eucl)
            curr_error = np.min(mesherr_eucl)
            vert_to_nearest_point_error_list.append(curr_error)



        # normalize by the average area of triangles around each point. the verts are much less spatially distributed well
        # than the points in the point cloud.
        # -- overall --
        norm_area_avg = self.get_triangle_area_vert_weight(human_mesh_vtx_mesherr[0], human_mesh_face_mesherr_red[0], verts_idx_red)
        #norm_area_avg_color = self.get_triangle_area_vert_weight(human_mesh_vtx_mesherr[0], human_mesh_face_mesherr_red[0], verts_idx_red)

        #print np.shape(norm_area_avg), np.shape(vert_to_nearest_point_error_list)
        vert_to_nearest_point_error_list = vert_to_nearest_point_error_list[0:np.shape(norm_area_avg)[0]]
        norm_vert_to_nearest_point_error = np.array(vert_to_nearest_point_error_list) * norm_area_avg
        v_to_gt_err = np.mean(norm_vert_to_nearest_point_error)
        print ("average vert to nearest pc point error:", v_to_gt_err)
        RESULTS_DICT['v_to_gt_err'].append(v_to_gt_err)



        # get the nearest point from ALL verts to some pc point, regardless of the normal - for coloring
        # we need this as a hack because the face indexing only refers to the original set of verts
        all_vert_to_nearest_point_error_list = []
        for all_vert_idx in range(human_mesh_vtx_mesherr[0].shape[0]):
            curr_vtx = human_mesh_vtx_mesherr[0][all_vert_idx, :]
            all_dist = pc_red_C - curr_vtx
            all_eucl = np.linalg.norm(all_dist, axis=1)
            curr_error = np.min(all_eucl)
            all_vert_to_nearest_point_error_list.append(curr_error)


        verts_color_error = np.array(all_vert_to_nearest_point_error_list) / np.max(vert_to_nearest_point_error_list)
        verts_color_jet = cm.jet(verts_color_error)[:, 0:3]# * 5.

        verts_color_jet_top = np.concatenate((verts_color_jet, np.ones((verts_color_jet.shape[0], 1))*0.9), axis = 1)
        verts_color_jet_bot = np.concatenate((verts_color_jet*0.3, np.ones((verts_color_jet.shape[0], 1))*0.9), axis = 1)


        all_verts = np.array(human_mesh_vtx_mesherr[0])
        faces_red = np.array(human_mesh_face_mesherr_red[0])
        faces_underside = np.concatenate((faces_red[:, 0:1],
                                          faces_red[:, 2:3],
                                          faces_red[:, 1:2]), axis = 1) + 6890

        human_vtx_both_sides = np.concatenate((all_verts, all_verts+0.0001), axis = 0)
        human_mesh_faces_both_sides = np.concatenate((faces_red, faces_underside), axis = 0)
        verts_color_jet_both_sides = np.concatenate((verts_color_jet_top, verts_color_jet_bot), axis = 0)

        tm_curr = trimesh.base.Trimesh(vertices=human_vtx_both_sides,
                                       faces=human_mesh_faces_both_sides,
                                       vertex_colors = verts_color_jet_both_sides)
        tm_list_mesherr =[tm_curr]


        #GET PIMG COLORED TRIMESH

        if pmat is not None and self.render == True:
            pmat_verts, pmat_faces, pmat_facecolors = self.get_3D_pmat_markers(pmat_est, bedangle)
            pmat_tm = trimesh.base.Trimesh(vertices=pmat_verts, faces=pmat_faces, face_colors=pmat_facecolors)
            pmat_mesh = pyrender.Mesh.from_trimesh(pmat_tm, smooth=False)

            pmat_verts2, _, pmat2_facecolors = self.get_3D_pmat_markers(pmat, bedangle)
            pmat_verts2 = np.array(pmat_verts2)
            pmat_verts2 = np.concatenate((np.swapaxes(pmat_verts2, 0, 1), np.ones((1, pmat_verts2.shape[0]))), axis=0)
            pmat_verts2 = np.swapaxes(np.matmul(transform_F, pmat_verts2), 0, 1)[:, 0:3]
            pmat_tm2 = trimesh.base.Trimesh(vertices=pmat_verts2, faces=pmat_faces, face_colors=pmat2_facecolors)
            pmat_mesh2 = pyrender.Mesh.from_trimesh(pmat_tm2, smooth=False)
        else:
            pmat_mesh = None
            pmat_mesh2 = None


        # get the nearest point from ALL verts to some pc point, regardless of the normal - for coloring
        # we need this as a hack because the face indexing only refers to the original set of verts
        #human_mesh_vtx_pimg = np.array(human_mesh_vtx_pimg)

        vertex_normals_pimg = np.array(tm_list[0].vertex_normals)
        vertices_pimg = np.array(tm_list[0].vertices)
        faces_pimg = np.array(tm_list[0].faces)

        vertices_pimg[:, 0] = vertices_pimg[:, 0] + transform_E[0, 3] #-1.0
        vertices_pimg[:, 1] = vertices_pimg[:, 1] + transform_E[1, 3] #-1.0


        norm_area_avg_all_est = self.get_triangle_area_vert_weight(vertices_pimg, faces_pimg, None)
        #print(np.shape(norm_area_avg_all_est), 'SUM NORM', np.sum(norm_area_avg_all_est))
        #print(norm_area_avg_all_est[0:5])

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


        RESULTS_DICT['vertex_pressure_list_EST'].append(vertex_pressure_init_list_EST*norm_area_avg_all_est)


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
        tm_list_pimg =[tm_curr]




        if smpl_verts_gt is not None:
            vertex_normals_pimg_GT = np.array(tm_list_GT[0].vertex_normals)
            vertices_pimg_GT = np.array(tm_list_GT[0].vertices)
            faces_pimg_GT = np.array(tm_list_GT[0].faces)

            vertices_pimg_GT[:, 0] = vertices_pimg_GT[:, 0] + transform_G[0, 3] #-1.0
            vertices_pimg_GT[:, 1] = vertices_pimg_GT[:, 1] + transform_G[1, 3] - 1.0

            norm_area_avg_all_gt = self.get_triangle_area_vert_weight(vertices_pimg_GT, faces_pimg_GT, None)
            #print(np.shape(norm_area_avg_all_gt), 'SUM NORM GT', np.sum(norm_area_avg_all_gt))
            #print(norm_area_avg_all_gt[0:5])


            vertex_pressure_init_list_GT = []
            for all_vert_idx in range(vertices_pimg_GT.shape[0]):
                #convert the vertex to 64x27 coords. pick color based on indexing 2d pressure map
                color_idx_y = int(64 - (vertices_pimg_GT[all_vert_idx, 0] - transform_G[0, 3] - self.y_bump)/ (0.0286*1.04) )# - 2.0) + 0.035
                color_idx_x = int( (vertices_pimg_GT[all_vert_idx, 1] - transform_G[1, 3] - self.x_bump)/ (0.0286) )#+ 0.5)# + 35.5 + 0.012

                try:
                    if vertex_normals_pimg_GT[all_vert_idx, 2] > 0:
                        cancel_pressure = self.check_vertex(all_vert_idx, vertices_pimg_GT, faces_pimg_GT)
                        if cancel_pressure == 1:
                            vertex_pressure_init_list_GT.append(0.0)
                        else:
                            vertex_pressure_init_list_GT.append(pmat[color_idx_y, color_idx_x])
                    else:
                        vertex_pressure_init_list_GT.append(0.0)
                except:
                    vertex_pressure_init_list_GT.append(0.0)

            #print(np.shape(vertex_pressure_init_list_GT))
            RESULTS_DICT['vertex_pressure_list_GT'].append(vertex_pressure_init_list_GT*norm_area_avg_all_gt)


            verts_color_error_GT = np.array(vertex_pressure_init_list_GT) /23.
            verts_color_jet_GT = np.clip(cm.jet(verts_color_error_GT)[:, 0:3], a_min = 0.0, a_max = 1.0)

            verts_color_jet_top_GT = np.concatenate((verts_color_jet_GT, np.ones((verts_color_jet_GT.shape[0], 1))*0.9), axis = 1)
            verts_color_jet_bot_GT = np.concatenate((verts_color_jet_GT*0.3, np.ones((verts_color_jet_GT.shape[0], 1))*0.9), axis = 1)


            all_verts_GT = np.array(vertices_pimg_GT)
            faces_red_GT = np.array(faces_pimg_GT)
            faces_underside_GT = np.concatenate((faces_red_GT[:, 0:1],
                                              faces_red_GT[:, 2:3],
                                              faces_red_GT[:, 1:2]), axis = 1) + 6890

            human_vtx_both_sides_GT = np.concatenate((all_verts_GT, all_verts_GT+0.0001), axis = 0)
            human_mesh_faces_both_sides_GT = np.concatenate((faces_red_GT, faces_underside_GT), axis = 0)
            verts_color_jet_both_sides_GT = np.concatenate((verts_color_jet_top_GT, verts_color_jet_bot_GT), axis = 0)

            tm_curr_GT = trimesh.base.Trimesh(vertices=human_vtx_both_sides_GT,
                                           faces=human_mesh_faces_both_sides_GT,
                                           vertex_colors = verts_color_jet_both_sides_GT)
            tm_list_pimg_GT =[tm_curr_GT]




        #shuffle(RESULTS_DICT['vertex_pressure_list_EST'][-1])

        both_0_ct = 0
        only_1_0_ct = 0
        both_pos_ct = 0

        for i in range(6890):
           # print(i, RESULTS_DICT['vertex_pressure_list_EST'][-1][i], RESULTS_DICT['vertex_pressure_list_GT'][-1][i], 'est, gt')
            if RESULTS_DICT['vertex_pressure_list_EST'][-1][i] == 0 and RESULTS_DICT['vertex_pressure_list_GT'][-1][i] == 0:
                both_0_ct += 1
            elif RESULTS_DICT['vertex_pressure_list_EST'][-1][i] == 0 and RESULTS_DICT['vertex_pressure_list_GT'][-1][i] != 0:
                only_1_0_ct += 1
            elif RESULTS_DICT['vertex_pressure_list_EST'][-1][i] != 0 and RESULTS_DICT['vertex_pressure_list_GT'][-1][i] == 0:
                only_1_0_ct += 1
            elif RESULTS_DICT['vertex_pressure_list_EST'][-1][i] != 0 and RESULTS_DICT['vertex_pressure_list_GT'][-1][i] != 0:
                both_pos_ct += 1


        print(both_0_ct, both_pos_ct, only_1_0_ct)
        #sys.exit()


        RESULTS_DICT['vertex_pressure_list_abs_err'].append(list(np.abs(np.array(RESULTS_DICT['vertex_pressure_list_EST'][-1]) - np.array(RESULTS_DICT['vertex_pressure_list_GT'][-1]) )))
        RESULTS_DICT['vertex_pressure_list_sq_err'].append(list(np.square(np.array(RESULTS_DICT['vertex_pressure_list_EST'][-1]) - np.array(RESULTS_DICT['vertex_pressure_list_GT'][-1]) )))

        print(np.mean(RESULTS_DICT['vertex_pressure_list_abs_err']), np.mean(RESULTS_DICT['vertex_pressure_list_sq_err']))





        #GET PCERROR
        all_verts = np.array(human_mesh_vtx_pcerr[0])
        faces_red = np.array(human_mesh_face_pcerr_red[0])
        faces_underside = np.concatenate((faces_red[:, 0:1],
                                          faces_red[:, 2:3],
                                          faces_red[:, 1:2]), axis = 1) + 6890


        verts_greysc_color = 1.0 * (all_verts[:, 2:3] - np.max(all_verts[:, 2])) / (np.min(all_verts[:, 2]) - np.max(all_verts[:, 2]))
        #print np.min(verts_greysc_color), np.max(verts_greysc_color), np.shape(verts_greysc_color)

        verts_greysc_color = np.concatenate((verts_greysc_color, verts_greysc_color, verts_greysc_color), axis=1)
        #print np.shape(verts_greysc_color)

        verts_color_grey_top = np.concatenate((verts_greysc_color, np.ones((verts_greysc_color.shape[0], 1))*0.7), axis = 1)
        verts_color_grey_bot = np.concatenate((verts_greysc_color*0.3, np.ones((verts_greysc_color.shape[0], 1))*0.7), axis = 1)

        human_vtx_both_sides = np.concatenate((all_verts, all_verts+0.0001), axis = 0)
        human_mesh_faces_both_sides = np.concatenate((faces_red, faces_underside), axis = 0)
        verts_color_jet_both_sides = np.concatenate((verts_color_grey_top, verts_color_grey_bot), axis = 0)

        tm_curr = trimesh.base.Trimesh(vertices=human_vtx_both_sides,
                                       faces=human_mesh_faces_both_sides,
                                       vertex_colors = verts_color_jet_both_sides)
        tm_list_pcerr = [tm_curr]




        mesh_list = []
        mesh_list.append(pyrender.Mesh.from_trimesh(tm_list[0], material = self.human_mat, wireframe = True)) #this is for the main human

        if smpl_verts_gt is not None:
            mesh_list_GT = []
            mesh_list_GT.append(pyrender.Mesh.from_trimesh(tm_list_GT[0], material=self.human_mat_GT, wireframe=True))

            mesh_list_pimg_GT = []
            mesh_list_pimg_GT.append(pyrender.Mesh.from_trimesh(tm_list_pimg_GT[0], smooth=False))
            mesh_list_pimg_GT.append(pyrender.Mesh.from_trimesh(tm_list_pimg_GT[0], material=self.human_mat, wireframe=True))

        mesh_list_mesherr = []
        mesh_list_mesherr.append(pyrender.Mesh.from_trimesh(tm_list_mesherr[0], smooth=False))

        mesh_list_pimg = []
        mesh_list_pimg.append(pyrender.Mesh.from_trimesh(tm_list_pimg[0], smooth=False))
        mesh_list_pimg.append(pyrender.Mesh.from_trimesh(tm_list_pimg[0], material = self.human_mat, wireframe = True))

        mesh_list_pcerr = []
        mesh_list_pcerr.append(pyrender.Mesh.from_trimesh(tm_list_pcerr[0], material = self.human_mat_D, smooth=False))


        #smpl_tm = trimesh.base.Trimesh(vertices=smpl_verts, faces=smpl_faces)
        #smpl_mesh = pyrender.Mesh.from_trimesh(smpl_tm, material=self.human_mat, wireframe = True)

        pc_greysc_color = 0.3 * (pc_red_C[:, 2:3] - np.max(pc_red_C[:, 2])) / (np.min(pc_red_C[:, 2]) - np.max(pc_red_C[:, 2]))
        pc_mesh_mesherr = pyrender.Mesh.from_points(pc_red_C, colors=np.concatenate((pc_greysc_color, pc_greysc_color, pc_greysc_color), axis=1))

        pc_greysc_color2 = 0.0 * (pc_red_F[:, 2:3] - np.max(pc_red_F[:, 2])) / (np.min(pc_red_F[:, 2]) - np.max(pc_red_F[:, 2]))
        pc_mesh_mesherr2 = pyrender.Mesh.from_points(pc_red_F, colors=np.concatenate((pc_greysc_color2, pc_greysc_color2, pc_greysc_color2), axis=1))


        faces_red = human_mesh_face_pcerr_red[0]
        verts_idx_red = np.unique(faces_red)
        verts_red = human_mesh_vtx_pcerr[0][verts_idx_red, :]

        # get the nearest point from each pc point to some vert, regardless of the normal
        pc_to_nearest_vert_error_list = []
        for point_idx in range(pc_red_D.shape[0]):
            curr_point = pc_red_D[point_idx, :]
            all_dist = verts_red - curr_point
            all_eucl = np.linalg.norm(all_dist, axis=1)
            curr_error = np.min(all_eucl)
            pc_to_nearest_vert_error_list.append(curr_error)
            # break
        gt_to_v_err = np.mean(pc_to_nearest_vert_error_list)
        print ("average pc point to nearest vert error:", gt_to_v_err)
        RESULTS_DICT['gt_to_v_err'].append(gt_to_v_err)


        if self.render == True:
            pc_color_error = np.array(pc_to_nearest_vert_error_list) / np.max(pc_to_nearest_vert_error_list)
            pc_color_jet = cm.jet(pc_color_error)[:, 0:3]

            pc_mesh_pcerr = pyrender.Mesh.from_points(pc_red_D, colors = pc_color_jet)






            if smpl_render_points == True:
                verts_idx_red = np.unique(human_mesh_face_all_red[0])

                verts_red = smpl_verts[verts_idx_red, :]
                smpl_pc_mesh = pyrender.Mesh.from_points(verts_red, colors = [5.0, 0.0, 0.0])
            else: smpl_pc_mesh = None


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
                            artag_tm = trimesh.base.Trimesh(vertices=self.artag_r+marker-markers[2], faces=self.artag_f, face_colors = self.artag_facecolors_root)
                            artag_meshes.append(pyrender.Mesh.from_trimesh(artag_tm, smooth = False))
                        else:
                            artag_tm = trimesh.base.Trimesh(vertices=self.artag_r+marker-markers[2], faces=self.artag_f, face_colors = self.artag_facecolors)
                            artag_meshes.append(pyrender.Mesh.from_trimesh(artag_tm, smooth = False))



            show_all = False
            #print "Viewing"
            if self.first_pass == True:
                if show_all == True:
                    for mesh_part in mesh_list:
                        print(mesh_part, 'adding meshpart')
                        self.scene.add(mesh_part)

                    if smpl_verts_gt is not None:
                        for mesh_part_GT in mesh_list_GT:
                            self.scene.add(mesh_part_GT)

                        for mesh_part_pimg_GT in mesh_list_pimg_GT:
                            self.scene.add(mesh_part_pimg_GT)



                for mesh_part_pimg in mesh_list_pimg:
                    self.scene.add(mesh_part_pimg)


                if show_all == True:

                    if pc_mesh_mesherr2 is not None:
                        self.scene.add(pc_mesh_mesherr2)

                    if pmat_mesh is not None:
                        self.scene.add(pmat_mesh)

                    if pmat_mesh2 is not None:
                        self.scene.add(pmat_mesh2)

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

                if show_all == True:
                    self.node_list = []
                    for mesh_part in mesh_list:
                        for node in self.scene.get_nodes(obj=mesh_part):
                            self.node_list.append(node)

                    self.node_list_GT = []
                    for mesh_part_GT in mesh_list_GT:
                        for node in self.scene.get_nodes(obj=mesh_part_GT):
                            self.node_list_GT.append(node)


                    self.node_list_pimg_GT = []
                    for mesh_part_pimg_GT in mesh_list_pimg_GT:
                        for node in self.scene.get_nodes(obj=mesh_part_pimg_GT):
                            self.node_list_pimg_GT.append(node)

                self.node_list_pimg = []
                for mesh_part_pimg in mesh_list_pimg:
                    for node in self.scene.get_nodes(obj=mesh_part_pimg):
                        self.node_list_pimg.append(node)


                if show_all == True:

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
                    if pmat_mesh2 is not None:
                        for node in self.scene.get_nodes(obj=pmat_mesh2):
                            self.pmat_node2 = node
                    if target_trimesh is not None:
                        for node in self.scene.get_nodes(obj=target_trimesh):
                            self.target_node = node
                    if score_trimesh is not None:
                        for node in self.scene.get_nodes(obj=score_trimesh):
                            self.score_node = node


            else:
                self.viewer.render_lock.acquire()


                if show_all == True:
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
                    for idx in range(len(mesh_list_pimg_GT)):
                        self.scene.remove_node(self.node_list_pimg_GT[idx])
                        self.scene.add(mesh_list_pimg_GT[idx])
                        for node in self.scene.get_nodes(obj=mesh_list_pimg_GT[idx]):
                            self.node_list_pimg_GT[idx] = node

                #reset the pc error human rendering
                for idx in range(len(mesh_list_pimg)):
                    self.scene.remove_node(self.node_list_pimg[idx])
                    self.scene.add(mesh_list_pimg[idx])
                    for node in self.scene.get_nodes(obj=mesh_list_pimg[idx]):
                        self.node_list_pimg[idx] = node



                if show_all == True:

                    #reset the point cloud mesh for mesherr
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
                    if pmat_mesh2 is not None:
                        self.scene.remove_node(self.pmat_node2)
                        self.scene.add(pmat_mesh2)
                        for node in self.scene.get_nodes(obj=pmat_mesh2):
                            self.pmat_node2 = node

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



                #print self.scene.get_nodes()
                self.viewer.render_lock.release()
            #time.sleep(100)

        #RESULTS_DICT = self.compare_pc_to_voxelmesh(smpl_verts, smpl_faces, pc, pmat, RESULTS_DICT, synth=False)'''
        return RESULTS_DICT
