


txtfile = open("/home/ganyong/Githubwork/Examples/BodyPressure/FILEPATH.txt")
FILEPATH = txtfile.read().replace("\n", "")
txtfile.close()

import sys
sys.path.insert(0, '../lib_py')
sys.path.insert(-1,FILEPATH+'smpl/smpl_webuser3')
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

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


#MISC
import time as time
import matplotlib.pyplot as plt
import matplotlib.cm as cm #use cm.jet(list)
from lib_py.tensorprep_lib_bp import TensorPrepLib
from lib_py.preprocessing_lib_bp import PreprocessingLib
from lib_py.visualization_lib_bp import VisualizationLib
from lib_py.slp_prep_lib_bp import SLPPrepLib

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

import os




try:
    from smpl.smpl_webuser.serialization import load_model
except:
    print("importing load model3",  sys.path)
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
            #self.pmat_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.6, 0.15, 0.0, 0.0], metallicFactor=0.5, roughnessFactor=0.9)  #
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
        camera_pose[0, 3] =  33 * 0.0286 / 2
        camera_pose[1, 3] = 68 * 0.0286 / 2
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




    def render_3D_data(self, pmat=None, human_vf = None, rot_noise = [0.0, 0.0, 0.0], trans_noise = [0.0, 0.0, 0.0]):
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




            depth = ndimage.zoom(depth, 0.5, order=1)

            depth = np.flipud(np.swapaxes(depth, 0, 1))
            #depth = depth[59:-59, 78:-79] #64 x 0.0286
            depth = depth[53:-53, 72:-73] #1.92
            #depth = depth[44:-44, 64:-65]


            color = np.flipud(np.swapaxes(color, 0, 1))
            #color = color[59:-59, 78:-79] #64 x 0.0286
            color = color[53:-53, 72:-73] #1.92
            #color = color[44:-44, 64:-65]
            #depth = depth[:, 53:53 + 108]



            print(np.shape(color), np.shape(depth), np.sum(depth), np.max(depth), 'shape sum max color depth')

            #print(depth, np.min(depth), np.max(depth), 'minmax depth')


        return depth, color


 
if __name__ == '__main__':
    pRM = pyRenderMesh(True)

    import trimesh as trimesh


    for i in range(1, 103):
        database_file_m = []
        database_file_f = []
        subj_mass_list_m = []
        subj_mass_list_f = []
        some_subject = '%05d' % float(i)
        if i == 7: continue
        phys_arr = np.load('/mnt/DADES2/SLP/SLP/danaLab/physiqueData.npy')
        phys_arr[:, [2, 0]] = phys_arr[:, [0, 2]]
        gender_bin = int(phys_arr[int(i) - 1][2])
        if gender_bin == 1:
            subj_mass_list_m.append([phys_arr[int(i) - 1][0], phys_arr[int(i) - 1][1]])
            database_file_m.append('%05d' % (i))
        else:
            subj_mass_list_f.append([phys_arr[int(i) - 1][0], phys_arr[int(i) - 1][1]])
            database_file_f.append('%05d' % (i))





        CTRL_PNL = {}
        CTRL_PNL['normalize_per_image'] = True
        CTRL_PNL['clip_sobel'] = False
        # load the synthetic pressure images
        #pmat_folder = "./data_BR/synth/general/"


        CTRL_PNL['incl_pmat_cntct_input'] = False
        CTRL_PNL['recon_map_input_est'] = False
        CTRL_PNL['incl_pmat_cntct_input'] = False
        CTRL_PNL['recon_map_labels'] = False
        CTRL_PNL['depth_in'] = True
        CTRL_PNL['depth_out_unet'] = False





        dana_lab_path = "/mnt/DADES2/SLP/SLP/danaLab/"

        dat_f_real = {}
        dat_f_real_u = SLPPrepLib().load_slp_files_to_database(database_file_f, dana_lab_path, PM='uncover', depth='uncover', mass_ht_list = subj_mass_list_f, markers_gt_type = '3D', use_pc = False, pm_adjust_mm = [12, -35])
        for item in dat_f_real_u:
            dat_f_real[item] = []
            for i in range(len(dat_f_real_u[item])): #assign 45 per subject
                dat_f_real[item].append(dat_f_real_u[item][i])

        dat_m_real = {}
        dat_m_real_u = SLPPrepLib().load_slp_files_to_database(database_file_m, dana_lab_path, PM='uncover', depth='uncover', mass_ht_list = subj_mass_list_m, markers_gt_type = '3D', use_pc = False, pm_adjust_mm = [12, -35])
        for item in dat_m_real_u:
            dat_m_real[item] = []
            for i in range(len(dat_m_real_u[item])): #assign 45 per subject
                dat_m_real[item].append(dat_m_real_u[item][i])


        if gender_bin == 1:
            dat_real = dat_m_real
        else:
            dat_real = dat_f_real

        train_y_flat = []  # Initialize the training ground truth list
        for gender_synth in [["f", dat_f_real], ["m", dat_m_real]]:
            train_y_flat = SLPPrepLib().prep_labels_slp(train_y_flat, gender_synth[1], num_repeats = 1,
                                                            z_adj = -0.075, gender = gender_synth[0], is_synth = True,
                                                            markers_gt_type = '3D',
                                                            initial_angle_est = False,
                                                            cnn_type = 'resnetunet')




        print(train_y_flat[0], np.shape(train_y_flat))

        if train_y_flat[0][157] == 1.:
            model_path = FILEPATH+'smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
            print('loaded female body')
        else:
            model_path = FILEPATH+'smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
            print('loaded male body')

        m = load_model(model_path)

        for pose_num in range(1,46):

            O_T_slp = np.load('/home/henry/data/01_init_poses/slp_gt_updated/uncover_' + some_subject + '/images/image_%06d/O_T_slp.npy' % (pose_num))
            slp_T_cam = np.load('/home/henry/data/01_init_poses/slp_gt_updated/uncover_' + some_subject + '/images/image_%06d/slp_T_cam.npy' % (pose_num))

            original_pose_data = load_pickle(FILEPATH+'data_BP/SLP_SMPL_fits/fits/p' + some_subject[-3:] + '/s%02d.pkl' % (pose_num))
            cam_T_Bo = np.array(original_pose_data['transl'])

            for shape_param in range(10):
                m.betas[shape_param] = train_y_flat[pose_num-1][72+shape_param]
            for pose_param in range(72):
                m.pose[pose_param] = train_y_flat[pose_num-1][82+pose_param]


            human_verts = np.array(m.r)
            human_faces = np.array(m.f)

            #human_verts[:, 2] += 0.2
            '''Bo_T_Br = np.array(m.J_transformed[0, 0:3])


            print(train_y_flat[pose_num-1][154:157], 'root gt stored from optimization')
            print(O_T_slp, 'O_T_slp')
            print(slp_T_cam, 'slp_T_cam')
            print(cam_T_Bo, 'cam_T_Bo')
            print(Bo_T_Br, 'Bo_T_Br')
            print(O_T_slp+slp_T_cam, 'real O_T_cam')
            print(O_T_slp+slp_T_cam+cam_T_Bo+Bo_T_Br, 'root gt stored from optimization v2')

            #
            O_T_camp = np.array([33 * 0.0286 / 2, -(68 * 0.0286 / 2), -(2.101 + 0.2032)])
            print(O_T_camp, 'synthetic O_T_cam')
            #print(O_T_camp+cam_T_Bo+Bo_T_Br, 'root gt of synthetic data')

            root_gt_updated_frame = O_T_camp+cam_T_Bo+Bo_T_Br
            root_gt_updated_frame[1] *= -1
            root_gt_updated_frame[2] *= -1

            print(root_gt_updated_frame, 'root gt of synthetic data')'''


            #for s in range(3):
            #    human_verts[:, s] += root_gt_updated_frame[s]#this thing is off a bit.
            for s in range(3):
                human_verts[:, s] += (train_y_flat[pose_num-1][154+s])#this thing is off a bit.

            human_verts[:, 0] += 0.0286*3 + 0.0286/2 + 0.012
            human_verts[:, 1] += 0.286 + 0.0286*3 + 0.0286/2 - 0.035
            human_verts[:, 2] += 0.2032


            #human_verts = np.concatenate((human_verts[:, 2:3],human_verts[:, 0:1],human_verts[:, 1:2]), axis = 1)
            #human_verts_nohuman = np.concatenate((human_verts[:, 2:3],human_verts[:, 0:1],human_verts[:, 1:2]), axis = 1) + 100.

            #human_verts[:, 1] -= (1.82 - 1.91)
            #human_verts[:,0] -= (0.86 - 0.7722/1.04)

            human_vf = [human_verts, human_faces]


            print("visualizing sample idx with only human: ", pose_num)
            depth_im_onlyhuman, color_im_onlyhuman = pRM.render_3D_data(None, human_vf)


            for item in dat_real:
                print(item, np.shape(dat_real[item]))


            pimage_real_curr_uncover = dat_real['images'][pose_num-1]
            pimage_real_curr_cover = dat_real['images'][pose_num-1]

            depth_real_curr_uncover = dat_real['overhead_depthcam_noblanket'][pose_num-1]
            depth_real_curr_cover = dat_real['overhead_depthcam_noblanket'][pose_num-1]



            if pose_num == 1 or pose_num == 11 or pose_num == 31:
                #try:

                VisualizationLib().plot_pimage_depth(pimage_real_curr_uncover, depth_im_onlyhuman, None, pimage_real_curr_uncover, depth_real_curr_uncover, None, \
                                                     depth_real_curr_uncover, None, pimage_real_curr_cover, depth_real_curr_cover, None, 2200, [True, pose_num, some_subject+"_real_slp"])
                #except:
                #    pass


            np.save('/home/henry/data/01_init_poses/slp_gt_updated/uncover_' + some_subject + '/images/image_%06d/' % (pose_num) + 'depth_humanonly.npy', (depth_im_onlyhuman))

        #break
    print("Saved.")
