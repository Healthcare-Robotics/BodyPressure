
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



def get_3D_pmat_markers(pmat, x_bump, y_bump, angle = 60.0):

    pmat_reshaped = pmat.reshape(64, 27)

    #pmat_colors = cm.jet(pmat_reshaped*100)
    pmat_colors = cm.jet(pmat_reshaped/23)
    #print pmat_colors.shape
    pmat_colors[:, :, 3] = 0.7 #translucency

    pmat_xyz = np.zeros((65, 28, 3))
    pmat_faces = []
    pmat_facecolors = []

    for j in range(65):
        for i in range(28):

            pmat_xyz[j, i, 1] = i * 0.0286 + x_bump# /1.06# * 1.02 #1.0926 - 0.02
            pmat_xyz[j, i, 0] = ((64 - j) * 0.0286) * 1.04 + y_bump#/1.04#1.1406 + 0.05 #only adjusts pmat NOT the SMPL person
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

    #print np.min(pmat_faces), np.max(pmat_faces), 'minmax'


    pmat_verts = list((pmat_xyz).reshape(1820, 3))

    return pmat_verts, pmat_faces, pmat_facecolors



def reduce_by_cam_dir(vertices, faces, camera_point, transform):

    vertices = np.array(vertices)
    faces = np.array(faces)

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


def get_triangle_area_vert_weight(verts, faces, verts_idx_red = None):

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

    if verts_idx_red is not None:
        try:
            norm_area_avg = norm_area_avg[verts_idx_red]
        except:
            norm_area_avg = norm_area_avg[verts_idx_red[:-1]]

    #print norm_area_avg[0:3], np.min(norm_area_avg), np.max(norm_area_avg), np.mean(norm_area_avg), np.sum(norm_area_avg)

    #print(np.mean(norm_area_avg), norm_area_avg)
    return norm_area_avg


def get_triangle_norm_to_vert(verts, faces, verts_idx_red):

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


def downspl_pc_get_normals(pc, camera_point):

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


    points = np.array(downpcd.points)
    normals = np.array(downpcd.normals)
    return points, normals


def pc_get_normals(pc, camera_point):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)


    o3d.geometry.PointCloud.estimate_normals(pcd, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

    o3d.geometry.PointCloud.orient_normals_towards_camera_location(pcd, camera_location=np.array(camera_point))

    #o3d.visualization.draw_geometries([downpcd])

    points = np.array(pcd.points)
    normals = np.array(pcd.normals)

    return points, normals





def plot_mesh_norms(verts, verts_norm):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    pcd.normals = o3d.utility.Vector3dVector(verts_norm)

    o3d.visualization.draw_geometries([pcd])


def get_human_mesh_parts(smpl_verts, smpl_faces, viz_type = None, segment_limbs = False, segment_type='joints'):

    if segment_limbs == True:
        if segment_type == 'joints':
            #print "got here"
            segmented_dict = load_pickle('../lib_py/segmented_mesh_idx_faces_joints.p')
            human_mesh_vtx_parts = [np.array(smpl_verts[segmented_dict['l_lowerleg_idx_list']]),
                                    np.array(smpl_verts[segmented_dict['r_lowerleg_idx_list']]),
                                    np.array(smpl_verts[segmented_dict['l_upperleg_idx_list']]),
                                    np.array(smpl_verts[segmented_dict['r_upperleg_idx_list']]),
                                    np.array(smpl_verts[segmented_dict['l_forearm_idx_list']]),
                                    np.array(smpl_verts[segmented_dict['r_forearm_idx_list']]),
                                    np.array(smpl_verts[segmented_dict['l_upperarm_idx_list']]),
                                    np.array(smpl_verts[segmented_dict['r_upperarm_idx_list']]),
                                    np.array(smpl_verts[segmented_dict['head_idx_list']]),
                                    np.array(smpl_verts[segmented_dict['torso_idx_list']])]
            human_mesh_face_parts = [np.array(segmented_dict['l_lowerleg_face_list']),
                                     np.array(segmented_dict['r_lowerleg_face_list']),
                                     np.array(segmented_dict['l_upperleg_face_list']),
                                     np.array(segmented_dict['r_upperleg_face_list']),
                                     np.array(segmented_dict['l_forearm_face_list']),
                                     np.array(segmented_dict['r_forearm_face_list']),
                                     np.array(segmented_dict['l_upperarm_face_list']),
                                     np.array(segmented_dict['r_upperarm_face_list']),
                                     np.array(segmented_dict['head_face_list']),
                                     np.array(segmented_dict['torso_face_list'])]

        elif segment_type == 'pressure':
            #print "got here"
            segmented_dict = load_pickle('../lib_py/segmented_mesh_idx_faces_pressure_reduced.p')
            #for item in segmented_dict: print(item)
            human_mesh_vtx_parts = [np.array(smpl_verts[segmented_dict['l_toes_idx_list']]),
                                    np.array(smpl_verts[segmented_dict['r_toes_idx_list']]),
                                    np.array(smpl_verts[segmented_dict['l_heel_idx_list']]),
                                    np.array(smpl_verts[segmented_dict['r_heel_idx_list']]),
                                    np.array(smpl_verts[segmented_dict['l_elbow_idx_list']]),
                                    np.array(smpl_verts[segmented_dict['r_elbow_idx_list']]),
                                    np.array(smpl_verts[segmented_dict['l_shoulder_idx_list']]),
                                    np.array(smpl_verts[segmented_dict['r_shoulder_idx_list']]),
                                    np.array(smpl_verts[segmented_dict['spine_idx_list']]),
                                    np.array(smpl_verts[segmented_dict['head_idx_list']]),
                                    np.array(smpl_verts[segmented_dict['l_hip_idx_list']]),
                                    np.array(smpl_verts[segmented_dict['r_hip_idx_list']]),
                                    np.array(smpl_verts[segmented_dict['sac_idx_list']]),
                                    np.array(smpl_verts[segmented_dict['isc_idx_list']])]
            human_mesh_face_parts = [np.array(segmented_dict['l_toes_face_list']),
                                     np.array(segmented_dict['r_toes_face_list']),
                                     np.array(segmented_dict['l_heel_face_list']),
                                     np.array(segmented_dict['r_heel_face_list']),
                                     np.array(segmented_dict['l_elbow_face_list']),
                                     np.array(segmented_dict['r_elbow_face_list']),
                                     np.array(segmented_dict['l_shoulder_face_list']),
                                     np.array(segmented_dict['r_shoulder_face_list']),
                                     np.array(segmented_dict['spine_face_list']),
                                     np.array(segmented_dict['head_face_list']),
                                     np.array(segmented_dict['l_hip_face_list']),
                                     np.array(segmented_dict['r_hip_face_list']),
                                     np.array(segmented_dict['sac_face_list']),
                                     np.array(segmented_dict['isc_face_list'])]

        '''elif segment_type == 'pressure':
            #print "got here"
            segmented_dict = load_pickle('../lib_py/segmented_mesh_idx_faces_pressure.p')
            human_mesh_vtx_parts = [np.array(smpl_verts[segmented_dict['l_foot_idx_list']]),
                                    np.array(smpl_verts[segmented_dict['r_foot_idx_list']]),
                                    np.array(smpl_verts[segmented_dict['l_leg_idx_list']]),
                                    np.array(smpl_verts[segmented_dict['r_leg_idx_list']]),
                                    np.array(smpl_verts[segmented_dict['l_arm_idx_list']]),
                                    np.array(smpl_verts[segmented_dict['r_arm_idx_list']]),
                                    np.array(smpl_verts[segmented_dict['l_shoulder_idx_list']]),
                                    np.array(smpl_verts[segmented_dict['r_shoulder_idx_list']]),
                                    np.array(smpl_verts[segmented_dict['spine_idx_list']]),
                                    np.array(smpl_verts[segmented_dict['head_idx_list']]),
                                    np.array(smpl_verts[segmented_dict['l_hip_idx_list']]),
                                    np.array(smpl_verts[segmented_dict['r_hip_idx_list']]),
                                    np.array(smpl_verts[segmented_dict['sac_isc_idx_list']])]
            human_mesh_face_parts = [np.array(segmented_dict['l_foot_face_list']),
                                     np.array(segmented_dict['r_foot_face_list']),
                                     np.array(segmented_dict['l_leg_face_list']),
                                     np.array(segmented_dict['r_leg_face_list']),
                                     np.array(segmented_dict['l_arm_face_list']),
                                     np.array(segmented_dict['r_arm_face_list']),
                                     np.array(segmented_dict['l_shoulder_face_list']),
                                     np.array(segmented_dict['r_shoulder_face_list']),
                                     np.array(segmented_dict['spine_face_list']),
                                     np.array(segmented_dict['head_face_list']),
                                     np.array(segmented_dict['l_hip_face_list']),
                                     np.array(segmented_dict['r_hip_face_list']),
                                     np.array(segmented_dict['sac_isc_face_list'])]'''

    else:
        human_mesh_vtx_parts = [smpl_verts]
        human_mesh_face_parts = [smpl_faces]

    return human_mesh_vtx_parts, human_mesh_face_parts



def check_if_in_triangle(p, p1, p2, p3):

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



def check_if_under_everything1(all_vert_idx, tr_ck_arr, vertices_pimg, faces_pimg):

    coinciding_verts = tr_ck_arr[:, 13].nonzero()

    p1_verts = vertices_pimg[faces_pimg[coinciding_verts, 0], 0:3][0]
    p2_verts = vertices_pimg[faces_pimg[coinciding_verts, 1], 0:3][0]
    p3_verts = vertices_pimg[faces_pimg[coinciding_verts, 2], 0:3][0]

    pl_ck_arr = np.concatenate((p1_verts, p2_verts, p3_verts), axis=1)
    pl_ck_arr = np.concatenate((np.zeros((pl_ck_arr.shape[0], 3)), pl_ck_arr, np.zeros((pl_ck_arr.shape[0], 12))),
                               axis=1)
    pl_ck_arr[:, 0] = vertices_pimg[all_vert_idx][0]
    pl_ck_arr[:, 1] = vertices_pimg[all_vert_idx][1]
    pl_ck_arr[:, 2] = vertices_pimg[all_vert_idx][2]

    plane_check_list = []
    for k in range(np.shape(pl_ck_arr)[0]):

        n = np.cross((pl_ck_arr[k, 6:9] - pl_ck_arr[k, 3:6]), (pl_ck_arr[k, 9:12] - pl_ck_arr[k, 3:6]))
        if n[2] > 0:
            n = np.cross((pl_ck_arr[k, 9:12] - pl_ck_arr[k, 3:6]), (pl_ck_arr[k, 6:9] - pl_ck_arr[k, 3:6]))
        is_on_side = np.dot((pl_ck_arr[k, 0:3] - pl_ck_arr[k, 3:6]), n)
        plane_check_list.append(is_on_side)

    plane_check_list = np.array(plane_check_list)
    plane_check_list[plane_check_list >= 0] = 1
    plane_check_list[plane_check_list < 0] = 0

    is_under_everything = np.all(plane_check_list == 0)

    return is_under_everything



def check_vertex(all_vert_idx, tr_ck_arr, vertices_pimg, faces_pimg):

    # here we check if its within the 2D triangle
    tr_ck_arr[:, 0] = vertices_pimg[all_vert_idx][0]
    tr_ck_arr[:, 1] = vertices_pimg[all_vert_idx][1]
    tr_ck_arr[:, 2:4] = vertices_pimg[faces_pimg[:, 0], 0:2]
    tr_ck_arr[:, 4:6] = vertices_pimg[faces_pimg[:, 1], 0:2]
    tr_ck_arr[:, 6:8] = vertices_pimg[faces_pimg[:, 2], 0:2]

    tr_ck_arr[:, 8] = (tr_ck_arr[:, 5] - tr_ck_arr[:, 3]) * (tr_ck_arr[:, 0] - tr_ck_arr[:, 2]) + (
                -tr_ck_arr[:, 4] + tr_ck_arr[:, 2]) * (tr_ck_arr[:, 1] - tr_ck_arr[:, 3])
    tr_ck_arr[:, 9] = (tr_ck_arr[:, 7] - tr_ck_arr[:, 5]) * (tr_ck_arr[:, 0] - tr_ck_arr[:, 4]) + (
                -tr_ck_arr[:, 6] + tr_ck_arr[:, 4]) * (tr_ck_arr[:, 1] - tr_ck_arr[:, 5])
    tr_ck_arr[:, 10] = (tr_ck_arr[:, 3] - tr_ck_arr[:, 7]) * (tr_ck_arr[:, 0] - tr_ck_arr[:, 6]) + (
                -tr_ck_arr[:, 2] + tr_ck_arr[:, 6]) * (tr_ck_arr[:, 1] - tr_ck_arr[:, 7])
    tr_ck_arr[tr_ck_arr[:, 8] >= 0, 8] = 1
    tr_ck_arr[tr_ck_arr[:, 9] >= 0, 9] = 1
    tr_ck_arr[tr_ck_arr[:, 10] >= 0, 10] = 1
    tr_ck_arr[tr_ck_arr[:, 8] < 0, 8] = 0
    tr_ck_arr[tr_ck_arr[:, 9] < 0, 9] = 0
    tr_ck_arr[tr_ck_arr[:, 10] < 0, 10] = 0
    tr_ck_arr[:, 11] = tr_ck_arr[:, 8] * tr_ck_arr[:, 9] * tr_ck_arr[:, 10]
    tr_ck_arr[:, 12] = tr_ck_arr[:, 8] + tr_ck_arr[:, 9] + tr_ck_arr[:, 10]
    tr_ck_arr[tr_ck_arr[:, 12] > 0, 12] = 1
    tr_ck_arr[:, 12] = 1 - tr_ck_arr[:, 12]
    tr_ck_arr[:, 13] = tr_ck_arr[:, 12] + tr_ck_arr[:, 11]

    num_passing_faces_before_overlap3 = int(np.sum(tr_ck_arr[:, 13]))
    if num_passing_faces_before_overlap3 >= 2:
        is_under_everything = check_if_under_everything1(all_vert_idx, tr_ck_arr, vertices_pimg, faces_pimg)
        if is_under_everything == 1:
            cancel_point = 0
        elif is_under_everything == 0:
            cancel_point = 1
    else:
        cancel_point = 0

    return cancel_point, tr_ck_arr
