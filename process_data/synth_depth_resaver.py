

txtfile = open("/home/ganyong/Githubwork/Examples/BodyPressure/FILEPATH.txt")
FILEPATH = txtfile.read().replace("\n", "")
txtfile.close()

import sys
sys.path.insert(0, 'lib_py')
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import trimesh
import pyrender
import pyglet
from scipy import ndimage

import numpy as np
import random
import copy
from time import sleep
import matplotlib.gridspec as gridspec

from tensorprep_lib_br import TensorPrepLib
from preprocessing_lib_br import PreprocessingLib

from visualization_lib_br import VisualizationLib

import math
from random import shuffle
import pickle as pickle

#MISC
import time as time
import matplotlib.pyplot as plt
import matplotlib.cm as cm #use cm.jet(list)

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

import os

from scipy import ndimage



class pyRenderMesh():
    def __init__(self):
        pass


    def resave_slp(self):

        #depthim_type = "depthims_nobkt"
        depthim_type = "depthims_nobkt_slp"
        depthim_onlyhuman_type = "depthims_onlyhuman_slp"
        partition = "slp"
        learning_type = "train"

        folder = FILEPATH+"data_BP/resting_meshes/" + partition + "/"
        folder_pressurepose =  FILEPATH+"data_BP/synth/" + partition + "/"
        folder_pressurepose_depth =  FILEPATH+"data_BP/synth_depth/" + partition + "/"

        depth_dict = {}
        depth_dict['overhead_depthcam_noblanket'] = []
        depth_dict['overhead_depthcam_onlyhuman'] = []
        if depthim_type == "depthims_nobkt_slp":
            depth_dict['overhead_depthcam'] = []



        #file_list = [["f", "lay", 1944, 8549, "1to10", "1to40"], ["f", "lay", 2210, 8549, "11to20", "1to40"],
        #             ["f", "lay", 2201, 8549, "21to30", "1to40"],["f", "lay", 2194, 8549, "31to40", "1to40"]]
        #file_list = [["f", "lside", 1857, 8136, "1to10", "1to40"], ["f", "lside", 2087, 8136, "11to20", "1to40"],
        #             ["f", "lside", 2086, 8136, "21to30", "1to40"],["f", "lside", 2106, 8136, "31to40", "1to40"]]
        #file_list = [["f", "rside", 1805, 7677, "1to10", "1to40"], ["f", "rside", 2001, 7677, "11to20", "1to40"],
        #             ["f", "rside", 1922, 7677, "21to30", "1to40"],["f", "rside", 1949, 7677, "31to40", "1to40"]]

        #file_list = [["m", "lay", 1946, 8493, "1to10", "1to40"], ["m", "lay", 2192, 8493, "11to20", "1to40"],
        #             ["m", "lay", 2178, 8493, "21to30", "1to40"],["m", "lay", 2177, 8493, "31to40", "1to40"]]
        #file_list = [["m", "lside", 1731, 7761, "1to10", "1to40"], ["m", "lside", 2007, 7761, "11to20", "1to40"],
        #             ["m", "lside", 2002, 7761, "21to30", "1to40"],["m", "lside", 2021, 7761, "31to40", "1to40"]]
        #file_list = [["m", "rside", 1704, 7377, "1to10", "1to40"], ["m", "rside", 1927, 7377, "11to20", "1to40"],
        #             ["m", "rside", 1844, 7377, "21to30", "1to40"],["m", "rside", 1902, 7377, "31to40", "1to40"]]

        #file_list = [["f", "lay", 2198, 6608, "41to50", "41to70"], ["f", "lay", 2197, 6608, "51to60", "41to70"],
        #             ["f", "lay", 2213, 6608, "61to70", "41to70"]]
        #file_list = [["f", "lside", 2091, 6158, "41to50", "41to70"], ["f", "lside", 2053, 6158, "51to60", "41to70"],
        #             ["f", "lside", 2014, 6158, "61to70", "41to70"]]
        #file_list = [["f", "rside", 1976, 6006, "41to50", "41to70"], ["f", "rside", 2043, 6006, "51to60", "41to70"],
        #             ["f", "rside", 1987, 6006, "61to70", "41to70"]]

        #file_list = [["m", "lay", 2195, 6597, "41to50", "41to70"], ["m", "lay", 2199, 6597, "51to60", "41to70"],
        #             ["m", "lay", 2203, 6597, "61to70", "41to70"]]
        #file_list = [["m", "lside", 2049, 5935, "41to50", "41to70"], ["m", "lside", 1952, 5935, "51to60", "41to70"],
        #             ["m", "lside", 1934, 5935, "61to70", "41to70"]]
        #file_list = [["m", "rside", 1904, 5817, "41to50", "41to70"], ["m", "rside", 1973, 5817, "51to60", "41to70"],
        #             ["m", "rside", 1940, 5817, "61to70", "41to70"]]


        #file_list = [["f", "lay", 2184, 2184, "71to80", "71to80"]]
        #file_list = [["f", "lside", 2058, 2058, "71to80", "71to80"]]
        #file_list = [["f", "rside", 2010, 2010, "71to80", "71to80"]]
        #file_list = [["m", "lay", 2188, 2188, "71to80", "71to80"]]
        #file_list = [["m", "lside", 2002, 2002, "71to80", "71to80"]]
        file_list = [["m", "rside", 1939, 1939, "71to80", "71to80"]]

        gender = file_list[0][0]
        posture = file_list[0][1]
        total_list_len = file_list[0][3]
        total_set_num = file_list[0][5]

        pressurepose_name = learning_type + "_slp_" +posture + "_" + gender + "_" + total_set_num + "_" + str(total_list_len)



        dat_f_synth = TensorPrepLib().load_files_to_database([[folder_pressurepose + pressurepose_name + ".p"]],\
                                                                 creation_type = 'synth', reduce_data = False)
        dat_m_synth = TensorPrepLib().load_files_to_database([[]], creation_type = 'synth', reduce_data = False)
        test_x = np.zeros((9000, 5, 64, 27)).astype(np.float32)
        synth_xa = TensorPrepLib().prep_images(test_x, None, None, dat_f_synth, dat_m_synth, filter_sigma = 0.5, start_map_idx = 0)


        sample_pimg_idx = 10
        overall_ct = 1
        for highct_setnum in file_list:

            list_len = highct_setnum[2]
            set_num = highct_setnum[4]

            depth_init_file_name = "slp_" + set_num + "_" + posture + "_" + gender + "_" + str(list_len) + "_filtered/"


            depth_im_list_file1 = np.load(folder + depth_init_file_name + depthim_type + '.npy')
            depth_im_list_file3 = np.load(folder + depth_init_file_name + depthim_onlyhuman_type + '.npy')
            if depthim_type == "depthims_nobkt_slp":
                depth_im_list_file2 = np.load(folder + depth_init_file_name + 'depthims_slp.npy')



            print(np.shape(depth_im_list_file1))
            for i in range(int(list_len)):

                if len(depth_dict['overhead_depthcam_noblanket']) >= int(total_list_len):
                    break

                if overall_ct % 20 == True:
                    print(overall_ct)


                depth_dict['overhead_depthcam_noblanket'].append(depth_im_list_file1[i, :, :])
                depth_dict['overhead_depthcam_onlyhuman'].append(depth_im_list_file3[i, :, :])
                try:
                    depth_dict['overhead_depthcam'].append(depth_im_list_file2[i, :, :])
                except:
                    pass
                overall_ct += 1



            pimage_synth_curr = synth_xa[sample_pimg_idx][0]

            if depthim_type == "depthims_nobkt_slp":
                VisualizationLib().plot_pimage_depth(pimage_synth_curr, depth_dict['overhead_depthcam_onlyhuman'][sample_pimg_idx], None, None, None, None, \
                                                     depth_dict['overhead_depthcam'][sample_pimg_idx], None, None, None, None, 2200, [True, sample_pimg_idx, pressurepose_name])
                try:
                    pimage_synth_curr = synth_xa[sample_pimg_idx+1600][0]
                    VisualizationLib().plot_pimage_depth(pimage_synth_curr, depth_dict['overhead_depthcam_onlyhuman'][sample_pimg_idx+1600], None, None,None, None,  \
                                                     depth_dict['overhead_depthcam'][sample_pimg_idx+1600], None, None, None, None, 2200, [True, sample_pimg_idx+1600, pressurepose_name])
                except:
                    pass

            #else:
            #    VisualizationLib().plot_pimage_depth(pimage_synth_curr, depth_dict['overhead_depthcam_noblanket'][sample_pimg_idx], None, None, None, None, \
            #                                         None, None, None, None, None, 1700, [True, sample_pimg_idx, pressurepose_name])
            #    try:
            #        pimage_synth_curr = synth_xa[sample_pimg_idx+250][0]
            #        VisualizationLib().plot_pimage_depth(pimage_synth_curr, depth_dict['overhead_depthcam_noblanket'][sample_pimg_idx+250], None, None,None, None,  \
            #                                         None, None, None, None, None, 1700, [True, sample_pimg_idx+250, pressurepose_name])
            #    except:
            #        pass

            sample_pimg_idx += int(list_len)




        print (np.shape(depth_dict['overhead_depthcam_noblanket']), len(depth_dict['overhead_depthcam_noblanket']), 'shape and length')
        print (np.shape(synth_xa), 'pressurepose')
        pickle.dump(depth_dict, open(folder_pressurepose_depth + pressurepose_name + '_' + depthim_type + '.p', 'wb'))
        print ("saved")




if __name__ == '__main__':
    pRM = pyRenderMesh()

    import trimesh as trimesh

    pRM.resave_slp()


