
import trimesh
from scipy import ndimage

import numpy as np
import random
import copy

txtfile = open("/home/ganyong/Githubwork/Examples/BodyPressure/FILEPATH.txt")
FILEPATH = txtfile.read().replace("\n", "")
txtfile.close()


import sys
sys.path.insert(0, '../lib_py')
sys.path.insert(-1,FILEPATH+'smpl/smpl_webuser3')
print(sys.path)



try:
    from smpl.smpl_webuser.serialization import load_model
except:
    from smpl.smpl_webuser3.serialization import load_model


from time import sleep

DATASET_CREATE_TYPE = 1

import math
from random import shuffle
import torch
import torch.nn as nn
import matplotlib.cm as cm
#MISC
import time as time

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
        self.pic_num = 0






    def resave_npart_with_vol(self, m, train_database_file_f):


        human_mesh_vtx_all = [np.array(m.r)]
        human_mesh_face_all = [np.array(m.f)]


        smpl_trimesh_0 = trimesh.Trimesh(vertices=human_mesh_vtx_all[0], faces=human_mesh_face_all[0])
        female_vol_0 = smpl_trimesh_0.volume
        print(female_vol_0)


        time.sleep(100)

        print(train_database_file_f)

        for item in train_database_file_f:
            print(item)
            next_file = load_pickle(item)
            for entry in next_file:
                try:
                    print(entry, np.shape(next_file[entry]), next_file[entry][0].dtype)
                except:
                    print(entry, np.shape(next_file[entry]))


            #next_file = load_pickle(item+'blah.p')
            #for entry in next_file:
            #    try:
            #        print(entry, np.shape(next_file[entry]), next_file[entry][0].dtype)
            #    except:
            #        print(entry, np.shape(next_file[entry]))

            next_file['body_volume'] = []

            for i in range(np.shape(next_file['body_nparticles'])[0]):
                print(i, next_file['body_shape'][i])

                for shape_param in range(10):
                    m.betas[shape_param] = float(next_file['body_shape'][i][shape_param])

                human_mesh_vtx_all_mod = [np.array(m.r)]

                smpl_trimesh_mod = trimesh.Trimesh(vertices=human_mesh_vtx_all_mod[0], faces=human_mesh_face_all[0])
                female_vol_mod = smpl_trimesh_mod.volume
                print(next_file['body_nparticles'][i], female_vol_mod)
                next_file['body_volume'].append(female_vol_mod)


            for entry in next_file:
                print(entry, np.shape(next_file[entry]))

            pkl.dump(next_file, open(item, 'wb'))




    def resave_slpsynth_with_vol(self, m, train_database_file_f):


        human_mesh_vtx_all = [np.array(m.r)]
        human_mesh_face_all = [np.array(m.f)]


        smpl_trimesh_0 = trimesh.Trimesh(vertices=human_mesh_vtx_all[0], faces=human_mesh_face_all[0])
        female_vol_0 = smpl_trimesh_0.volume
        print(female_vol_0)


        #time.sleep(100)

        print(train_database_file_f)

        for item in train_database_file_f:
            print(item)
            next_file = load_pickle(item)
            for entry in next_file:
                try:
                    print(entry, np.shape(next_file[entry]), next_file[entry][0].dtype)
                except:
                    print(entry, np.shape(next_file[entry]))


            #next_file = load_pickle(item+'blah.p')
            #for entry in next_file:
            #    try:
            #        print(entry, np.shape(next_file[entry]), next_file[entry][0].dtype)
            #    except:
            #        print(entry, np.shape(next_file[entry]))

            next_file['body_volume'] = []

            for i in range(np.shape(next_file['body_height'])[0]):
                #print(i, next_file['body_shape'][i])

                for shape_param in range(10):
                    m.betas[shape_param] = float(next_file['body_shape'][i][shape_param])

                human_mesh_vtx_all_mod = [np.array(m.r)]

                smpl_trimesh_mod = trimesh.Trimesh(vertices=human_mesh_vtx_all_mod[0], faces=human_mesh_face_all[0])
                vol_mod = smpl_trimesh_mod.volume
                #print(next_file['body_height'][i], vol_mod)
                next_file['body_volume'].append(vol_mod)

                print i, next_file['body_volume'][i], vol_mod


            for entry in next_file:
                print(entry, np.shape(next_file[entry]))

            pkl.dump(next_file, open(item, 'wb'))





if __name__ ==  "__main__":



    import optparse

    from optparse_lib import get_depthnet_options
    from filename_input_lib_br import FileNameInputLib

    p = optparse.OptionParser()

    p = get_depthnet_options(p)

    p.add_option('--mod', action='store', type = 'int', dest='mod', default=1,
                 help='Choose a network.')


    p.add_option('--viz', action='store_true', dest='visualize', default=False,  help='Visualize training.')
    opt, args = p.parse_args()



    if opt.hd == True:
        dana_lab_path = '/media/henry/multimodal_data_2/data/SLP/danaLab/'
    else:
        dana_lab_path = '/mnt/DADES2/SLP/SLP/danaLab/'




    FileNameInputLib1 = FileNameInputLib(opt, depth = False)

    #train_database_file_f, train_database_file_m = FileNameInputLib1.get_184K_pressurepose(True, '')#_nonoise')
    #test_database_file_f, test_database_file_m = FileNameInputLib1.get_184K_pressurepose(False, '')#_nonoise')
    train_database_file_f, train_database_file_m = FileNameInputLib1.get_slpsynth_pressurepose(True, '')#_nonoise')
    test_database_file_f, test_database_file_m = FileNameInputLib1.get_slpsynth_pressurepose(False, '')#_nonoise')



    pRM = pyRenderMesh(None)

    gender = 'f'
    model_path = FILEPATH+'smpl/models/basicModel_'+gender+'_lbs_10_207_0_v1.0.0.pkl'
    mf = load_model(model_path)

    #pRM.resave_npart_with_vol(mf, train_database_file_f)
    #pRM.resave_npart_with_vol(mf, test_database_file_f)
    #pRM.resave_slpsynth_with_vol(mf, test_database_file_f)
    #pRM.resave_slpsynth_with_vol(mf, train_database_file_f)


    gender = 'm'
    model_path = FILEPATH+'smpl/models/basicmodel_'+gender+'_lbs_10_207_0_v1.0.0.pkl'
    mm = load_model(model_path)

    pRM.resave_slpsynth_with_vol(mm, train_database_file_m)
    pRM.resave_slpsynth_with_vol(mm, test_database_file_m)