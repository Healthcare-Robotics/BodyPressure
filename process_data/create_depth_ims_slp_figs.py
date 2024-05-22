try:
    import open3d as o3d
except:
    print("CANNOT IMPORT 03D. POINT CLOUD PROCESSING WON'T WORK")

import trimesh
import pyrender
import pyglet
from scipy import ndimage

import numpy as np
import random
import copy
from time import sleep
import matplotlib.gridspec as gridspec


import math
from random import shuffle
import pickle as pickle

txtfile = open("../FILEPATH.txt")
FILEPATH = txtfile.read().replace("\n", "")
txtfile.close()


import sys
sys.path.insert(0, '../lib_py')
sys.path.insert(-1,FILEPATH+'smpl/smpl_webuser3')
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

#MISC
import time as time
import matplotlib.pyplot as plt
import matplotlib.cm as cm #use cm.jet(list)
from tensorprep_lib_br import TensorPrepLib
from preprocessing_lib_br import PreprocessingLib
from visualization_lib_br import VisualizationLib
from slp_prep_lib_br import SLPPrepLib

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

import os



try:
    from smpl.smpl_webuser.serialization import load_model
except:
    from smpl.smpl_webuser3.serialization import load_model




class pyRenderMesh():
    def __init__(self, render):

        # terms = 'f', 'frustum', 'background_image', 'overdraw', 'num_channels'
        # dterms = 'vc', 'camera', 'bgcolor'
        self.first_pass = True
        self.render = render
        if True:# render == True:
            self.scene = pyrender.Scene(bg_color=[0.5,0.5,0.5])
            self.mattress_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.4, 0.4, 0.1, 0.0], metallicFactor=0.1, roughnessFactor=0.1)  #
            self.blanket_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.5, 0.02, 0.02, 0.0], metallicFactor=0.1, roughnessFactor=0.4)  #
            self.blanket_mat_underside = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.2, 0.01, 0.01, 0.0], metallicFactor=0.1, roughnessFactor=0.4)  #
            self.human_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.05, 0.05, 0.8, 0.0], metallicFactor=0.6, roughnessFactor=0.5)  #
            self.pmat_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.6, 0.6, 0.6, 0.0], metallicFactor=0.5, roughnessFactor=0.9)  #


            mesh_color_mult = 0.25


        self.pic_num = 0


    def eulerAnglesToRotationMatrix(self,theta):
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(theta[0]), -math.sin(theta[0])],
                        [0, math.sin(theta[0]), math.cos(theta[0])]
                        ])

        R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                        [0, 1, 0],
                        [-math.sin(theta[1]), 0, math.cos(theta[1])]
                        ])

        R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                        [math.sin(theta[2]), math.cos(theta[2]), 0],
                        [0, 0, 1]
                        ])

        R = np.dot(R_z, np.dot(R_y, R_x))

        return R


    def get_3D_pmat_markers(self, pmat):

        pmat_reshaped = pmat.reshape(64, 27)

        pmat_colors = cm.jet(pmat_reshaped/100)
        #print pmat_colors.shape
        pmat_colors[:, :, 3] = 0.7 #pmat translucency

        pmat_xyz = np.zeros((65, 28, 3))
        pmat_faces = []
        pmat_facecolors = []

        for j in range(65):
            for i in range(28):

                pmat_xyz[j, i, 1] = i * 0.0286# /1.06# * 1.02 #1.0926 - 0.02
                pmat_xyz[j, i, 0] = ((64 - j) * 0.0286) * 1.04 #/1.04#1.1406 + 0.05 #only adjusts pmat NOT the SMPL person
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



    def get_camera_pose(self, rot_noise = [0.0, 0.0, 0.0], trans_noise = [0.0, 0.0, 0.0]):

        camera_pose = np.eye(4)
        # camera_pose[0,0] = -1.0
        # camera_pose[1,1] = -1.0

        camera_pose[0, 0] = np.cos(np.pi / 2)
        camera_pose[0, 1] = -np.sin(np.pi / 2)
        camera_pose[1, 0] = np.sin(np.pi / 2)
        camera_pose[1, 1] = np.cos(np.pi / 2)
        rot_udpim = np.eye(4)

        rot_y = 0. * np.pi / 180. #change to 45 to look down from the side
        rot_udpim[0, 0] = np.cos(rot_y)
        rot_udpim[2, 2] = np.cos(rot_y)
        rot_udpim[0, 2] = np.sin(rot_y)
        rot_udpim[2, 0] = -np.sin(rot_y)
        camera_pose = np.matmul(rot_udpim, camera_pose)

        #original original
        #camera_pose[1, 3] = 68 * 0.0286 / 2+ 0.015
        #camera_pose[0, 3] =  0.46441343 #0.5 [, 0.46441343, -CAM_BED_DIST]
        #camera_pose[2, 3] = 2.101 + 0.2032 #1.66

        #with black line at the bottom
        #camera_pose[1, 3] = 68 * 0.0286 / 2 - 0.035
        #camera_pose[0, 3] =  0.46441343
        #camera_pose[2, 3] = 2.101 + 0.2032

        #optimzied with black line at the bottom
        #camera_pose[1, 3] = 68 * 0.0286 / 2 - 0.035 - 0.010
        #camera_pose[0, 3] =  0.46441343 + 0.014
        #camera_pose[2, 3] = 2.101 + 0.2032 #1.66

        #not optimized - need to do it to markers in training
        camera_pose[1, 3] = 68 * 0.0286 / 2
        camera_pose[0, 3] =  33 * 0.0286 / 2
        camera_pose[2, 3] = 2.101 + 0.2032 #1.66



        #camera_pose[1, 3] = 68*0.0286/2
        #camera_pose[0, 3] =  33*0.0286/2
        #camera_pose[2, 3] = 2.101 + 0.2032 #1.66

        random_rotation = self.eulerAnglesToRotationMatrix(rot_noise)#5*np.pi/180])
        random_transform = np.eye(4)
        random_transform[0, 3] = trans_noise[0]
        random_transform[1, 3] = trans_noise[1]
        random_transform[2, 3] = trans_noise[2]
        random_transform[0:3, 0:3] = random_rotation
        camera_pose = np.matmul(camera_pose, random_transform)


        return camera_pose




    def render_3D_data(self, iteration, pmat=None, human_vf = None, pmat_vf = None, blanket_vf = None, blanket_vf_other = None, mattress_vf = None, depth_im_list = None, color_im_list = None, rot_noise = [0.0, 0.0, 0.0], trans_noise = [0.0, 0.0, 0.0]):
        plot_depth = True
        set_custom_camera = True#False

        #this is for the pressure mat
        if pmat is not None:
            pmat_verts, pmat_faces, pmat_facecolors = self.get_3D_pmat_markers(pmat)
            pmat_tm = trimesh.base.Trimesh(vertices=pmat_verts, faces=pmat_faces, face_colors = pmat_facecolors)
            pmat_mesh = pyrender.Mesh.from_trimesh(pmat_tm, smooth = False)



        mesh_list = []
        if human_vf is not None:
            print("human shape: ", np.shape(human_vf[0]), np.shape(human_vf[1]))
            tm_curr = trimesh.base.Trimesh(vertices=np.array(human_vf[0]), faces = np.array(human_vf[1]))
            tm_list = [tm_curr]
            mesh_list.append(pyrender.Mesh.from_trimesh(tm_list[0], material = self.human_mat, smooth=True)) #wireframe = True)) #this is for the main human


        if pmat_vf is not None:
            print("pmat shape: ", np.shape(pmat_vf[0]), np.shape(pmat_vf[1]))
            tm_curr = trimesh.base.Trimesh(vertices=np.array(pmat_vf[0]), faces = np.array(pmat_vf[1]))
            tm_list = [tm_curr]
            mesh_list.append(pyrender.Mesh.from_trimesh(tm_list[0], material = self.pmat_mat, smooth=True)) #wireframe = True)) #this is for the main human


        if blanket_vf is not None:
            print("blanket shape: ", np.shape(blanket_vf[0]), np.shape(blanket_vf[1]))
            tm_curr = trimesh.base.Trimesh(vertices=np.array(blanket_vf[0]), faces = np.array(blanket_vf[1]))
            tm_list = [tm_curr]
            mesh_list.append(pyrender.Mesh.from_trimesh(tm_list[0], material = self.blanket_mat_underside, smooth=True)) #wireframe = True)) #this is for the main human

        if blanket_vf_other is not None:
            print("blanket shape: ", np.shape(blanket_vf_other[0]), np.shape(blanket_vf_other[1]))
            tm_curr = trimesh.base.Trimesh(vertices=np.array(blanket_vf_other[0]), faces = np.array(blanket_vf_other[1]))
            tm_list = [tm_curr]
            mesh_list.append(pyrender.Mesh.from_trimesh(tm_list[0], material = self.blanket_mat, smooth=True)) #wireframe = True)) #this is for the main human


        if mattress_vf is not None:
            print("mattress shape: ", np.shape(mattress_vf[0]), np.shape(mattress_vf[1]))
            tm_curr = trimesh.base.Trimesh(vertices=np.array(mattress_vf[0]), faces = np.array(mattress_vf[1]))
            tm_list = [tm_curr]
            mesh_list.append(pyrender.Mesh.from_trimesh(tm_list[0], material = self.mattress_mat, smooth=True)) #wireframe = True)) #this is for the main human


        if plot_depth == True:
            r = pyrender.OffscreenRenderer(viewport_width=1200,
                                           viewport_height = 1200,
                                           point_size = 1.0)



        #print "Viewing"
        if self.first_pass == True:

            if pmat is not None:
                self.scene.add(pmat_mesh)

            for mesh_part in mesh_list:
                self.scene.add(mesh_part)




            '''light = pyrender.SpotLight(color=np.ones(3), intensity=250.0, innerConeAngle=np.pi / 10.0,
                                       outerConeAngle=np.pi / 2.0)
            light_pose = np.copy(camera_pose)
            light_pose[0, 3] = 0.8
            light_pose[1, 3] = -0.5
            light_pose[2, 3] = -2.5
            self.scene.add(light, pose=light_pose)'''



            lighting_intensity = 7.

            if set_custom_camera == False:
                self.viewer = pyrender.Viewer(self.scene, use_raymond_lighting=True, lighting_intensity=lighting_intensity,
                                              point_size=2, run_in_thread=True, viewport_size=(1200, 1200), show_world_axis = False)
            else:
                #magnify = (64 * .0286)
                #camera = pyrender.OrthographicCamera(xmag=magnify, ymag = magnify)
                self.camera = pyrender.PerspectiveCamera(yfov=70.6*np.pi/180., aspectRatio=70.6/60.)

                #5 * np.pi / 180, 5 * np.pi / 180, 5 * np.pi / 180
                #rot_noise = [0.0, np.pi/12, 0.0]#np.pi/12]

                #print rot_noise, trans_noise, "rot noise trans noise"
                camera_pose = self.get_camera_pose(rot_noise, trans_noise)

                ####here alter the camera pose


                self.scene.add(self.camera, pose=camera_pose)
                light = pyrender.SpotLight(color=np.ones(3), intensity=150.0, innerConeAngle=np.pi / 10.0,
                                           outerConeAngle=np.pi / 2.0)
                light_pose = np.copy(camera_pose)
                light_pose[0, 3] = 0.8
                light_pose[1, 3] = -0.5
                light_pose[2, 3] = 3.5

                self.scene.add(light, pose=light_pose)


            self.first_pass = False


            if pmat is not None:
                for node in self.scene.get_nodes(obj=pmat_mesh):
                    self.pmat_node = node

            self.node_list = []
            for mesh_part in mesh_list:
                for node in self.scene.get_nodes(obj=mesh_part):
                    self.node_list.append(node)


        else:
            if plot_depth == False and set_custom_camera == False:
                self.viewer.render_lock.acquire()

            camera_pose = self.get_camera_pose(rot_noise, trans_noise)
            for cam_node in self.scene.get_nodes(obj=self.camera):
                #print camera_pose
                self.scene.set_pose(cam_node, pose = camera_pose)
                break

            #reset the pmat mesh
            if pmat is not None:
                self.scene.remove_node(self.pmat_node)
                self.scene.add(pmat_mesh)
                for node in self.scene.get_nodes(obj=pmat_mesh):
                    self.pmat_node = node

            #reset the human mesh
            for idx in range(len(mesh_list)):
                self.scene.remove_node(self.node_list[idx])
                self.scene.add(mesh_list[idx])
                for node in self.scene.get_nodes(obj=mesh_list[idx]):
                    self.node_list[idx] = node

            if plot_depth == False and set_custom_camera == False:
                self.viewer.render_lock.release()


        if plot_depth == True:# and iteration > 5:

            #r = pyrender.OffscreenRenderer(246, 246*60/70.6)
            #r = pyrender.OffscreenRenderer(2*246, 2*246*60/70.6) #64*0.0286
            r = pyrender.OffscreenRenderer(2*234.4, 2*234.4*60/70.6) #1.92
            #r = pyrender.OffscreenRenderer(2*216, 2*216*60/70.6)
            color, depth = r.render(self.scene)
            depth = (depth*1000)

            depth = np.clip(depth, a_min = 0, a_max = 65535).astype(np.uint16)

            print(np.shape(color), np.shape(depth), 'color depth shape')


            color_r = ndimage.zoom(color[:, :, 0:1], 0.5, order=1)
            color_g = ndimage.zoom(color[:, :, 1:2], 0.5, order=1)
            color_b = ndimage.zoom(color[:, :, 2:3], 0.5, order=1)
            color = np.concatenate((color_r, color_g, color_b), axis = 2)



            print(depth.shape, 'depth shape')
            depth = ndimage.zoom(depth, 0.5, order=1)
            print(depth.shape, 'depth shape')

            depth = np.flipud(np.swapaxes(depth, 0, 1))
            #depth = depth[59:-59, 78:-79] #64 x 0.0286
            depth = depth[53:-53, 72:-73] #1.92
            #depth = depth[44:-44, 64:-65]
            print(depth.shape, 'depth shape')


            color = np.flipud(np.swapaxes(color, 0, 1))
            #color = color[59:-59, 78:-79] #64 x 0.0286
            color = color[53:-53, 72:-73] #1.92
            #color = color[44:-44, 64:-65]
            #depth = depth[:, 53:53 + 108]



            depth_im_list.append(depth)
            color_im_list.append(color)
            print(np.shape(color), np.shape(depth), np.sum(depth), np.max(depth), 'shape sum max color depth')

            #print(depth, np.min(depth), np.max(depth), 'minmax depth')


        return depth_im_list, color_im_list


 
if __name__ == '__main__':
    pRM = pyRenderMesh(True)

    import trimesh as trimesh
    #person = trimesh.load('person.obj')
    #print(np.max(person.faces))


    #for gpsn in [["f", "lay", 1944, 8549, "1to10", "train", 0, 0]]: #YES
    #for gpsn in [["f", "lay", 2210, 8549, "11to20", "train"]]:
    #for gpsn in [["f", "lay", 2201, 8549, "21to30", "train"]]:
    #for gpsn in [["f", "lay", 2194, 8549, "31to40", "train"]]:

    #for gpsn in [["f", "lside", 1857, 8136, "1to10", "train", 1200, 0]]: #YES
    #for gpsn in [["f", "lside", 2087, 8136, "11to20", "train"]]:
    #for gpsn in [["f", "lside", 2086, 8136, "21to30", "train"]]:
    #for gpsn in [["f", "lside", 2106, 8136, "31to40", "train"]]:

    #for gpsn in [["f", "rside", 1805, 7677, "1to10", "train"]]:
    #for gpsn in [["f", "rside", 2001, 7677, "11to20", "train"]]:
    #for gpsn in [["f", "rside", 1922, 7677, "21to30", "train", 157, 1805+2001]]:  #YES
    #for gpsn in [["f", "rside", 1949, 7677, "31to40", "train"]]:

    #for gpsn in [["m", "lay", 1946, 8493, "1to10", "train"]]:
    for gpsn in [["m", "lay", 2192, 8493, "11to20", "train", 1203, 1946]]: #NO
    #for gpsn in [["m", "lay", 2178, 8493, "21to30", "train", 455, 1946+2192]]: #YES
    #for gpsn in [["m", "lay", 2177, 8493, "31to40", "train"]]:

    #for gpsn in [["m", "lside", 1731, 7761, "1to10", "train"]]:
    #for gpsn in [["m", "lside", 2007, 7761, "11to20", "train", 730, 1731]]: #YES
    #for gpsn in [["m", "lside", 2002, 7761, "21to30", "train"]]:
    #for gpsn in [["m", "lside", 2021, 7761, "31to40", "train"]]:

    #for gpsn in [["m", "rside", 1704, 7377, "1to10", "train"]]:
    #for gpsn in [["m", "rside", 1927, 7377, "11to20", "train"]]:
    #for gpsn in [["m", "rside", 1844, 7377, "21to30", "train"]]:
    #for gpsn in [["m", "rside", 1902, 7377, "31to40", "train"]]:



        gender = gpsn[0]
        posture = gpsn[1]
        num_resting_poses = gpsn[2]
        num_databag_poses = gpsn[3]
        subj_nums = gpsn[4]
        dattype = gpsn[5]
        sample_idx = gpsn[6]
        sample_pimg_idx_addition = gpsn[7]

    #for highct_setnum in [["445", "601", "1", "f", "_plo"],["1782", "2601", "2", "f", "_plo"],["1769", "2601", "4", "f", "_plo"],["560", "800", "1", "m", "_plo"],["1958", "2601", "2", "m", "_plo"]]:

        color_im_list = []
        color_im_list_noblanket = []
        color_im_list_onlyhuman = []

        depth_im_list = []
        depth_im_list_noblanket = []
        depth_im_list_onlyhuman = []
        depth_im_list_noblanket_noisey = []
        depth_im_list_onlyhuman_noisey = []
        depth_im_list_onlyblanket = []




        mesh_folder = FILEPATH+"data_BP/04_resting_meshes/slp/"
        file_name = 'slp_' + subj_nums + '_' + posture + '_' + gender + '_' + str(num_resting_poses) +'_filtered_'+str(sample_idx)




        mesh_folder = mesh_folder + file_name
        mv = np.load(mesh_folder+"/mv.npy", allow_pickle = True)
        mf = np.load(mesh_folder+"/mf.npy", allow_pickle = True)
        hv = np.load(mesh_folder+"/hv.npy", allow_pickle = True)
        hf = np.load(mesh_folder+"/hf.npy", allow_pickle = True)
        bv = np.load(mesh_folder+"/bv.npy", allow_pickle = True)
        bf = np.load(mesh_folder+"/bf.npy", allow_pickle = True)
        k = np.load(mesh_folder+"/k.npy", allow_pickle = True)

        CTRL_PNL = {}
        CTRL_PNL['normalize_per_image'] = True
        CTRL_PNL['clip_sobel'] = False
        # load the synthetic pressure images
        #pmat_folder = "./data_BR/synth/general/"




        pmat_folder = FILEPATH+"data_BP/synth/"

        if gender == "f":
            dat_f_synth = TensorPrepLib().load_files_to_database([[pmat_folder + dattype + '_slp_'  + posture + '_' + gender + '_1to40_' + str(num_databag_poses)+".p"]],\
                                                                     creation_type = 'synth', reduce_data = False)

            dat_m_synth = TensorPrepLib().load_files_to_database([[]], creation_type = 'synth', reduce_data = False)
        elif gender == "m":
            dat_m_synth = TensorPrepLib().load_files_to_database([[pmat_folder + dattype + '_slp_'  + posture + '_' + gender + '_1to40_' + str(num_databag_poses)+".p"]],\
                                                                     creation_type = 'synth', reduce_data = False)

            dat_f_synth = TensorPrepLib().load_files_to_database([[]], creation_type = 'synth', reduce_data = False)

        test_x = np.zeros((9000, 5, 64, 27)).astype(np.float32)
        synth_xa = TensorPrepLib().prep_images(test_x, None, None, dat_f_synth, dat_m_synth, filter_sigma = 0.5, start_map_idx = 0)

        train_y_flat = []
        for gender_synth in [["f", dat_f_synth], ["m", dat_m_synth]]:
            train_y_flat = TensorPrepLib().prep_labels(train_y_flat, gender_synth[1],
                                                       z_adj=-0.075, gender=gender_synth[0], is_synth=True,
                                                       loss_vector_type='anglesDC',
                                                       initial_angle_est=False,
                                                       cnn_type='resnet',
                                                       x_y_adjust_mm=[12, -35])


        if train_y_flat[0][157] == 1.:
            model_path = FILEPATH+'smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
            print('loaded female body')
        else:
            model_path = FILEPATH+'smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
            print('loaded male body')

        m = load_model(model_path)





        print(np.shape(synth_xa))



        CTRL_PNL['incl_pmat_cntct_input'] = False
        CTRL_PNL['recon_map_input_est'] = False
        CTRL_PNL['incl_pmat_cntct_input'] = False
        CTRL_PNL['recon_map_labels'] = False
        CTRL_PNL['depth_in'] = True
        CTRL_PNL['depth_out_unet'] = False


        dana_lab_path = FILEPATH+"data_BP/SLP/danaLab/"

        load_real = False
        if load_real == True:
            color_real = {}
            depth_real = {}
            pimage_real = {}
            for bedding in ['uncover', 'cover2']:
                dat_f_slp = SLPPrepLib().load_slp_files_to_database(['00013'], dana_lab_path, PM=bedding, depth=bedding, color=bedding)
                dat_m_slp = SLPPrepLib().load_slp_files_to_database([], dana_lab_path, PM=bedding, depth=bedding, color=bedding)

                x_flat_real_slp = []
                x_flat_real_slp = TensorPrepLib().prep_images(x_flat_real_slp, dat_f_slp, dat_m_slp, num_repeats = 1)
                x_flat_real_slp = PreprocessingLib().preprocessing_blur_images(x_flat_real_slp, (64, 27), sigma=0.5)

                depth_images_real_slp = []
                if bedding == 'cover2' or bedding == 'cover1':
                    depth_images_real_slp = TensorPrepLib().prep_depth_input_images(depth_images_real_slp, dat_f_slp, dat_m_slp,
                                                                                    num_repeats=1,
                                                                                    depth_type='all_meshes')  # 'all_meshes')#'
                elif bedding == 'uncover':
                    depth_images_real_slp = TensorPrepLib().prep_depth_input_images(depth_images_real_slp, dat_f_slp, dat_m_slp,
                                                                                    num_repeats=1,
                                                                                    depth_type='no_blanket')  # 'all_meshes')#'

                synth_xa_real_slp = PreprocessingLib().preprocessing_create_pressure_angle_stack(x_flat_real_slp, (64, 27), CTRL_PNL)

                # stack the depth and contact mesh images (and possibly a pmat contact image) together
                synth_xa_real_slp = TensorPrepLib().append_trainxa_besides_pmat_edges(np.array(synth_xa_real_slp),
                                                                             CTRL_PNL=CTRL_PNL,
                                                                             mesh_reconstruction_maps_input_est=None,
                                                                             mesh_reconstruction_maps=None,
                                                                             depth_images=depth_images_real_slp,
                                                                             depth_images_out_unet=None)
                if bedding == 'uncover':
                    color_real[bedding] = np.array(dat_f_slp['overhead_colorcam_noblanket'])*1
                elif bedding == 'cover2':
                    color_real[bedding] = np.array(dat_f_slp['overhead_colorcam'])*1

                depth_real[bedding] = np.concatenate((np.concatenate((synth_xa_real_slp[:, 2:3, :, :], synth_xa_real_slp[:, 3:4, :, :]), axis=3),
                                             np.concatenate((synth_xa_real_slp[:, 4:5, :, :], synth_xa_real_slp[:, 5:6, :, :]), axis=3)), axis=2)
                depth_real[bedding] = np.squeeze(depth_real[bedding])
                pimage_real[bedding] = np.squeeze(synth_xa_real_slp[:, 0:1, :, :])




        sample_pimg_idx = sample_idx+sample_pimg_idx_addition


        for shape_param in range(10):
            m.betas[shape_param] = train_y_flat[sample_pimg_idx][72+shape_param]
        for pose_param in range(72):
            m.pose[pose_param] = train_y_flat[sample_pimg_idx][82+pose_param]


        human_verts_param = np.array(m.r)
        for s in range(0,3):
            human_verts_param[:, s] += (train_y_flat[sample_pimg_idx][154+s])#this thing is off a bit.
        human_faces_param = np.array(m.f)

        #human_verts_param[:, 0] += 0.0286*3 + 0.0286/2 + 0.012
        #human_verts_param[:, 1] += 0.286 + 0.0286*2 + 0.0286/2 - 0.035
        #human_verts_param[:, 2] += 0.2032
        human_verts_param[:, 0] += 0.0286*3 + 0.0286/2 + 0.012
        human_verts_param[:, 1] += 0.286 + 0.0286*3 + 0.0286/2 - 0.035
        human_verts_param[:, 2] += 0.2032

        #camera_pose[1, 3] = 68 * 0.0286 / 2
        #camera_pose[0, 3] = 33 * 0.0286 / 2
        #camera_pose[2, 3] = 2.101 + 0.2032  # 1.66

        sample_idx += 0


        #print sample_idx
        human_verts = np.array(hv[0, :, :])/2.58872 #np.load(mesh_folder + "human_mesh_verts_py.npy")/2.58872
        human_verts = np.concatenate((human_verts[:, 2:3],human_verts[:, 0:1],human_verts[:, 1:2]), axis = 1)
        human_verts_nohuman = np.concatenate((human_verts[:, 2:3],human_verts[:, 0:1],human_verts[:, 1:2]), axis = 1) + 100.

        #human_verts[:, 1] -= (1.82 - 1.91)
        #human_verts[:,0] -= (0.86 - 0.7722/1.04)

        human_faces = np.array(hf[0, :, :]) #np.load(mesh_folder + "human_mesh_faces_py.npy")
        human_faces = np.concatenate((np.array([[0, 1, 2], [0, 4, 1], [0, 5, 4], [0, 2, 132], [0, 235, 5], [0, 132, 235] ]), human_faces), axis = 0)
        human_vf = [human_verts_param, human_faces_param]
        #human_vf = [human_verts, human_faces]
        human_vf_nohuman = [human_verts_nohuman, human_faces]

        print(np.min(human_verts_param[:,0]), np.min(human_verts[:,0]), np.min(human_verts_param[:,0])- np.min(human_verts[:,0]), 'min 0 comparison')
        print(np.max(human_verts_param[:,0]), np.max(human_verts[:,0]), np.max(human_verts_param[:,0])- np.max(human_verts[:,0]), 'max 0 comparison')
        print(np.min(human_verts_param[:,1]), np.min(human_verts[:,1]), np.min(human_verts_param[:,1])- np.min(human_verts[:,1]), 'min 1 comparison')
        print(np.max(human_verts_param[:,1]), np.max(human_verts[:,1]), np.max(human_verts_param[:,1])- np.max(human_verts[:,1]), 'max 1 comparison')
        print(np.min(human_verts_param[:,2]), np.min(human_verts[:,2]), np.min(human_verts_param[:,2])- np.min(human_verts[:,2]), 'min 2 comparison')
        print(np.max(human_verts_param[:,2]), np.max(human_verts[:,2]), np.max(human_verts_param[:,2])- np.max(human_verts[:,2]), 'max 2 comparison')


        mattress_verts = np.array(mv[0, :, :])/2.58872 #np.load(mesh_folder + "mattress_verts_py.npy")/2.58872
        mattress_verts = np.concatenate((mattress_verts[:, 2:3],mattress_verts[:, 0:1],mattress_verts[:, 1:2]), axis = 1)
        mattress_verts_nomattress = np.concatenate((mattress_verts[:, 2:3],mattress_verts[:, 0:1],mattress_verts[:, 1:2]), axis = 1) + 100.
        mattress_faces = np.array(mf[0, :, :]) #np.load(mesh_folder + "mattress_faces_py.npy")
        mattress_faces = np.concatenate((np.array([[0, 6054, 6055]]), mattress_faces), axis = 0)
        mattress_vf = [mattress_verts, mattress_faces]
        mattress_vf_nomattress = [mattress_verts_nomattress, mattress_faces]


        mat_blanket_verts = np.array(bv[0])/2.58872 #np.load(mesh_folder + "mat_blanket_verts_py.npy")/2.58872
        print(mat_blanket_verts, 'mat blanket verts')
        mat_blanket_verts = np.concatenate((mat_blanket_verts[:, 2:3],mat_blanket_verts[:, 0:1],mat_blanket_verts[:, 1:2]), axis = 1)

        mat_blanket_faces = np.array(bf[0]) #np.load(mesh_folder + "mat_blanket_faces_py.npy")


        #print(np.shape(mat_blanket_verts), mat_blanket_verts)
        #print(np.shape(mat_blanket_faces), mat_blanket_faces, np.max(mat_blanket_faces))

        pmat_faces = []
        for i in range(np.shape(mat_blanket_faces)[0]):
            if np.max(mat_blanket_faces[i, :]) < 2244:#2300:# < 10000:
                pmat_faces.append([mat_blanket_faces[i, 0], mat_blanket_faces[i, 2], mat_blanket_faces[i, 1]])

            #elif np.max(mat_blanket_faces[i, :]) > 3000 and np.max(mat_blanket_faces[i, :])< 3100:  #
            #    stagger = 0
            #    pmat_faces.append([mat_blanket_faces[i, 0]+stagger, mat_blanket_faces[i, 2]+stagger, mat_blanket_faces[i, 1]+stagger])


        pmat_verts = np.copy(mat_blanket_verts)[0:6000]
        pmat_verts_nopmat = np.copy(mat_blanket_verts)[0:6000] + 100.
        pmat_faces = np.array(pmat_faces)

        pmat_faces = np.concatenate((np.array([[0, 69, 1], [0, 68, 69]  ]), pmat_faces), axis = 0)


        pmat_vf = [pmat_verts, pmat_faces]
        pmat_vf_nopmat = [pmat_verts_nopmat, pmat_faces]

        #print(pmat_faces)






        print(np.shape(mat_blanket_faces), np.min(mat_blanket_faces), np.max(mat_blanket_faces))

        blanket_faces = []
        for i in range(np.shape(mat_blanket_faces)[0]):
            if np.max(mat_blanket_faces[i, :]) > 10000:# and np.max(mat_blanket_faces[i, :]) < 50288:
                stagger = -34
                blanket_faces.append([mat_blanket_faces[i, 0]+stagger, mat_blanket_faces[i, 2]+stagger, mat_blanket_faces[i, 1]+stagger])

        blanket_faces_other = []
        for i in range(np.shape(mat_blanket_faces)[0]):
            if np.max(mat_blanket_faces[i, :]) > 10000:# and np.max(mat_blanket_faces[i, :]) < 50288:
                stagger = -34
                blanket_faces_other.append([mat_blanket_faces[i, 0]+stagger, mat_blanket_faces[i, 1]+stagger, mat_blanket_faces[i, 2]+stagger])



        print (np.min(blanket_faces), np.max(blanket_faces))
        blanket_face_start = np.min(blanket_faces)

        blanket_faces = np.array(blanket_faces) - blanket_face_start
        blanket_faces_other = np.array(blanket_faces_other) - blanket_face_start
        blanket_verts = np.copy(mat_blanket_verts)[blanket_face_start:]
        blanket_verts_noblanket = np.copy(mat_blanket_verts)[blanket_face_start:] + 100.

        if np.shape(blanket_verts)[0] <= 10200:
            print("too few blanket verts", np.shape(blanket_verts))
            blanket_verts = np.concatenate((blanket_verts[0:1, :], blanket_verts), axis = 0)
            blanket_verts_noblanket = np.concatenate((blanket_verts_noblanket[0:1, :], blanket_verts_noblanket), axis = 0)

        if np.shape(blanket_verts)[0] <= 10200:
            print("too few blanket verts", np.shape(blanket_verts))
            blanket_verts = np.concatenate((blanket_verts[0:1, :], blanket_verts), axis = 0)
            blanket_verts_noblanket = np.concatenate((blanket_verts_noblanket[0:1, :], blanket_verts_noblanket), axis = 0)


        #blanket_verts[:, 1] -=1.1
        blanket_verts[:, 2] -= np.random.uniform(low = 0, high = 0.0375/2) #vary the thickness of the blanket when we make the depth image by moving it down just a bit


        blanket_vf = [blanket_verts, blanket_faces]
        blanket_vf_noblanket = [blanket_verts_noblanket, blanket_faces]

        blanket_verts_other = np.copy(blanket_verts)
        blanket_verts_other_noblanket = np.copy(blanket_verts_noblanket)
        blanket_vf_other = [blanket_verts_other, blanket_faces_other]
        blanket_vf_other_noblanket = [blanket_verts_other_noblanket, blanket_faces_other]



        print("visualizing sample idx with blanket: ", sample_idx, np.shape(blanket_vf[0]), np.shape(blanket_vf[1]), np.min(blanket_vf[1]), np.max(blanket_vf[1]))
        depth_im_list, color_im_list = pRM.render_3D_data(i, None, human_vf, pmat_vf, blanket_vf, blanket_vf_other, mattress_vf, depth_im_list, color_im_list)

        print("visualizing sample idx without blanket: ", sample_idx)
        depth_im_list_noblanket, color_im_list_noblanket = pRM.render_3D_data(i, None, human_vf, pmat_vf, blanket_vf_noblanket, blanket_vf_other_noblanket, mattress_vf, depth_im_list_noblanket, color_im_list_noblanket)

        print("visualizing sample idx with only human: ", sample_idx)
        depth_im_list_onlyhuman, color_im_list_onlyhuman = pRM.render_3D_data(i, None, human_vf, pmat_vf_nopmat, blanket_vf_noblanket, blanket_vf_other_noblanket, mattress_vf_nomattress, depth_im_list_onlyhuman, color_im_list_onlyhuman)
        #print depth_im_list_noblanket[-1]
        #pRM.render_3D_data(i, None, human_vf, pmat_vf, None, None, mattress_vf)






        if load_real == True:
            pimage_real_curr_uncover = pimage_real['uncover'][sample_idx, :, :]
            pimage_real_curr_cover = pimage_real['cover2'][sample_idx, :, :]

            depth_real_curr_uncover = depth_real['uncover'][sample_idx, :, :]
            depth_real_curr_cover = depth_real['cover2'][sample_idx, :, :]

            color_real_curr_uncover = color_real['uncover'][sample_idx, :, :]
            color_real_curr_cover = color_real['cover2'][sample_idx, :, :]
        else:
            pimage_real_curr_uncover, pimage_real_curr_cover = None, None
            depth_real_curr_uncover, depth_real_curr_cover = None, None
            color_real_curr_uncover, color_real_curr_cover = None, None


        if False:#sample_idx%200 == 0:
            #try:
            pimage_synth_curr = synth_xa[sample_pimg_idx][0]
            #VisualizationLib().plot_pimage_depth(pimage_synth_curr, depth_im_list_noblanket[-1], color_im_list_noblanket[-1], pimage_real_curr_uncover, depth_real_curr_uncover, color_real_curr_uncover, \
            #                                     depth_im_list[-1], color_im_list[-1], pimage_real_curr_cover, depth_real_curr_cover, color_real_curr_cover, 2200, [True, sample_idx, file_name+"_slp"])
            pimage_synth_curr = synth_xa[sample_pimg_idx][0]

            print(len(depth_im_list_onlyhuman), len(color_im_list_onlyhuman), len(depth_im_list), len(color_im_list))

            VisualizationLib().plot_pimage_depth(pimage_synth_curr, depth_im_list_onlyhuman[-1], color_im_list_onlyhuman[-1], pimage_real_curr_uncover, depth_real_curr_uncover, color_real_curr_uncover, \
                                                 depth_im_list[-1], color_im_list[-1], pimage_real_curr_cover, depth_real_curr_cover, color_real_curr_cover, 2200, [True, sample_idx, file_name+"_slp"])
            #except:
            #    pass

        from scipy.ndimage.interpolation import zoom
        depth_im_render = zoom(depth_im_list[-1], 3.435, order=1)*0.98
        print(np.shape(depth_im_render), synth_xa[sample_pimg_idx][0].shape)

        human_verts_flip = np.concatenate((human_verts[:, 1:2], human_verts[:, 0:1], -human_verts[:, 2:3]), axis = 1)
        human_verts_flip[:, 0] -= 0.0286*3
        human_verts_flip[:, 2] -= 0.2

        human_verts_param_flip = np.concatenate((human_verts_param[:, 1:2], human_verts_param[:, 0:1], -human_verts_param[:, 2:3]), axis = 1)
        human_verts_param_flip[:, 0] -= 0.0286*5
        human_verts_param_flip[:, 1] -= 0.0286*4
        #human_verts_param_flip[:, 2] -= 0.0286*3

        import lib_pyrender_depth_savefig as libPyRenderSave
        pyRender = libPyRenderSave.pyRenderMesh(render=True)
        pyRender.render_mesh_pc_bed_pyrender_everything(human_verts_param_flip, human_faces_param, {},
                                                             # smpl_verts_gt = smpl_verts_gt,
                                                             pmat=None, pmat_est=synth_xa[sample_pimg_idx][0],
                                                             color_im_occl=None,
                                                             color_im=None,
                                                             depth_im=depth_im_render,
                                                             # depth should be 440 by 185. pressure should be 64 x 27
                                                             current_pose_type_ct=str(sample_idx),
                                                             participant=file_name)



        all_ims1 = np.stack((depth_im_list_noblanket)).astype(np.int16)
        #print all_ims1.shape
        np.save(mesh_folder + '/depthims_nobkt_slp.npy', all_ims1)

        all_ims2 = np.stack((depth_im_list)).astype(np.int16)
        #print all_ims2.shape
        np.save(mesh_folder + '/depthims_slp.npy', all_ims2)

        all_ims3 = np.stack((depth_im_list_onlyhuman)).astype(np.int16)
        #print all_ims3.shape
        np.save(mesh_folder + '/depthims_onlyhuman_slp.npy', all_ims3)
        break

    print("Saved.")
