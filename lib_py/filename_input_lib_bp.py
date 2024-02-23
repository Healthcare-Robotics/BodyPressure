#!/usr/bin/env python

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import *

# PyTorch libraries
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import transforms
from torch.autograd import Variable


MAT_WIDTH = 0.762 #metres
MAT_HEIGHT = 1.854 #metres
MAT_HALF_WIDTH = MAT_WIDTH/2
NUMOFTAXELS_X = 64#73 #taxels
NUMOFTAXELS_Y = 27#30
INTER_SENSOR_DISTANCE = 0.0286#metres
LOW_TAXEL_THRESH_X = 0
LOW_TAXEL_THRESH_Y = 0
HIGH_TAXEL_THRESH_X = (NUMOFTAXELS_X - 1)
HIGH_TAXEL_THRESH_Y = (NUMOFTAXELS_Y - 1)


txtfile = open("/home/ganyong/Githubwork/Examples/BodyPressure/FILEPATH.txt")
FILEPATH = txtfile.read().replace("\n", "")
txtfile.close()


class FileNameInputLib():
    def __init__(self, opt, depth = False):
        if opt.hd == False:
            self.data_fp_prefix = FILEPATH + "data_BP/"
        else:
            self.data_fp_prefix = "/media/henry/git/multimodal_data_2/data_BR/"

        self.data_fp_suffix = ''

        self.opt = opt

        if opt.mod == 2:  # or opt.quick_test == True:
            if opt.X_is == 'W':
                self.data_fp_suffix = '_resnet34_1_' + str(opt.losstype)
            if opt.X_is == 'B':
                self.data_fp_suffix = '_resnetunet34_1_' + str(opt.losstype)


            if opt.small == True:
                self.data_fp_suffix += '_46000ct'
            elif opt.slp == 'synth':
                #self.data_fp_suffix += '_85114ct'
                self.data_fp_suffix += '_97495ct'
            elif opt.slp == 'real':
                #self.data_fp_suffix += '_9315ct'
                self.data_fp_suffix += '_10665ct'
            elif opt.slp == 'mixed':
                self.data_fp_suffix += '_183114ct'
            elif opt.slp == 'mixedreal':
                #self.data_fp_suffix += '_94429ct'
                self.data_fp_suffix += '_108160ct'
            else:
                self.data_fp_suffix += '_184000ct'

            self.data_fp_suffix += '_128b_x1pm'

            if opt.no_reg_angles == False:
                self.data_fp_suffix += '_rgangs'

            if opt.no_loss_betas == False:
                self.data_fp_suffix += '_lb'
            if opt.noloss_htwt == True:
                self.data_fp_suffix += '_nlhw'

            if opt.no_blanket == False or opt.slp == 'real':
                self.data_fp_suffix += '_slpb'
            else:
                self.data_fp_suffix += '_slpnb'

            if opt.no_depthnoise == False:
                self.data_fp_suffix += '_dpns'
            if opt.slpnoise == True:
                self.data_fp_suffix += '_slpns'
            if opt.no_loss_root == False:
                self.data_fp_suffix += '_rt'
            if opt.X_is == 'B':
                self.data_fp_suffix += '_dou'
            if opt.slp == True:
                self.data_fp_suffix += '_slp'

            self.data_fp_suffix += '_100e_' + str(0.0001) + 'lr'

        elif opt.mod == 1:
            self.data_fp_suffix = ''  # _hb'

        else:
            print("Please choose a valid network. You can specify '--net 1' or '--net 2'.")
            sys.exit()


        if depth == True:
            self.synth_folder_suffix = "_depth"
            if opt.slp_depth == True:
                self.data_fp_suffix = "_depthims"
            else:
                self.data_fp_suffix = "_depthims_nobkt"

        else:
            self.synth_folder_suffix = ""




    def get_qt_slp(self, is_train = True):

        database_file_m = []
        database_file_f = []
        subj_mass_list_m = []
        subj_mass_list_f = []

        if is_train == True:
            for i in range(1,3):
                if i == 7: continue
                phys_arr = np.load('/mnt/DADES2/SLP/SLP/danaLab/physiqueData.npy')
                phys_arr[:, [2, 0]] = phys_arr[:, [0, 2]]
                gender_bin = int(phys_arr[int(i) - 1][2])
                if gender_bin == 1:
                    subj_mass_list_m.append(phys_arr[int(i) - 1][0])
                    database_file_m.append('%05d' % (i))
                else:
                    subj_mass_list_f.append(phys_arr[int(i) - 1][0])
                    database_file_f.append('%05d' % (i))
        else:
            for i in range(91,92):
                phys_arr = np.load('/mnt/DADES2/SLP/SLP/danaLab/physiqueData.npy')
                phys_arr[:, [2, 0]] = phys_arr[:, [0, 2]]
                gender_bin = int(phys_arr[int(i) - 1][2])
                if gender_bin == 1:
                    subj_mass_list_m.append(phys_arr[int(i) - 1][0])
                    database_file_m.append('%05d' % (i))
                else:
                    subj_mass_list_f.append(phys_arr[int(i) - 1][0])
                    database_file_f.append('%05d' % (i))

        return database_file_f, database_file_m, subj_mass_list_f, subj_mass_list_m



    def get_dana_slp(self, is_train = True):

        database_file_m = []
        database_file_f = []
        subj_mass_list_m = []
        subj_mass_list_f = []

        if is_train == True:
            #for i in range(1, 71):
            for i in range(1, 81):
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
        else:
            for i in range(81,103):
            #for i in range(71, 81):
                phys_arr = np.load('/mnt/DADES2/SLP/SLP/danaLab/physiqueData.npy')
                phys_arr[:, [2, 0]] = phys_arr[:, [0, 2]]
                gender_bin = int(phys_arr[int(i) - 1][2])
                if gender_bin == 1:
                    subj_mass_list_m.append([phys_arr[int(i) - 1][0], phys_arr[int(i) - 1][1]])
                    database_file_m.append('%05d' % (i))
                else:
                    subj_mass_list_f.append([phys_arr[int(i) - 1][0], phys_arr[int(i) - 1][1]])
                    database_file_f.append('%05d' % (i))


        return database_file_f, database_file_m, subj_mass_list_f, subj_mass_list_m



    def get_qt_dana_slp(self, is_train = True):

        database_file_m = []
        database_file_f = []
        subj_mass_list_m = []
        subj_mass_list_f = []

        if is_train == True:
            for i in range(1, 2):
                #if i == 7: continue
                phys_arr = np.load('/mnt/DADES2/SLP/SLP/danaLab/physiqueData.npy')
                phys_arr[:, [2, 0]] = phys_arr[:, [0, 2]]
                gender_bin = int(phys_arr[int(i) - 1][2])
                if gender_bin == 1:
                    subj_mass_list_m.append([phys_arr[int(i) - 1][0], phys_arr[int(i) - 1][1]])
                    database_file_m.append('%05d' % (i))
                else:
                    subj_mass_list_f.append([phys_arr[int(i) - 1][0], phys_arr[int(i) - 1][1]])
                    database_file_f.append('%05d' % (i))
        else:
            #for i in range(91,103):
            for i in range(71, 72):
                phys_arr = np.load('/mnt/DADES2/SLP/SLP/danaLab/physiqueData.npy')
                phys_arr[:, [2, 0]] = phys_arr[:, [0, 2]]
                gender_bin = int(phys_arr[int(i) - 1][2])
                if gender_bin == 1:
                    subj_mass_list_m.append([phys_arr[int(i) - 1][0], phys_arr[int(i) - 1][1]])
                    database_file_m.append('%05d' % (i))
                else:
                    subj_mass_list_f.append([phys_arr[int(i) - 1][0], phys_arr[int(i) - 1][1]])
                    database_file_f.append('%05d' % (i))


        return database_file_f, database_file_m, subj_mass_list_f, subj_mass_list_m







    def get_slpsynth_pressurepose(self, is_train = True, extra_suffix = ''):

        database_file_m = []
        database_file_f = []

        if len(self.data_fp_suffix) > 20:
            synth_folder = 'mod1est_synth'
        else:
            synth_folder = 'synth'

        if is_train == True:
            # SLP partition - xx,xxx train + xx,xxx test
            database_file_f.append(self.data_fp_prefix + synth_folder+self.synth_folder_suffix+'/train_slp_lay_f_1to40_8549' + self.data_fp_suffix + extra_suffix + '.p')
            database_file_f.append(self.data_fp_prefix + synth_folder+self.synth_folder_suffix+'/train_slp_lside_f_1to40_8136' + self.data_fp_suffix + extra_suffix + '.p')
            database_file_f.append(self.data_fp_prefix + synth_folder+self.synth_folder_suffix+'/train_slp_rside_f_1to40_7677' + self.data_fp_suffix + extra_suffix + '.p')
            database_file_m.append(self.data_fp_prefix + synth_folder+self.synth_folder_suffix+'/train_slp_lay_m_1to40_8493' + self.data_fp_suffix + extra_suffix + '.p')
            database_file_m.append(self.data_fp_prefix + synth_folder+self.synth_folder_suffix+'/train_slp_lside_m_1to40_7761' + self.data_fp_suffix + extra_suffix + '.p')
            database_file_m.append(self.data_fp_prefix + synth_folder+self.synth_folder_suffix+'/train_slp_rside_m_1to40_7377' + self.data_fp_suffix + extra_suffix + '.p')

            database_file_f.append(self.data_fp_prefix + synth_folder+self.synth_folder_suffix+'/train_slp_lay_f_41to70_6608' + self.data_fp_suffix + extra_suffix + '.p')
            database_file_f.append(self.data_fp_prefix + synth_folder+self.synth_folder_suffix+'/train_slp_lside_f_41to70_6158' + self.data_fp_suffix + extra_suffix + '.p')
            database_file_f.append(self.data_fp_prefix + synth_folder+self.synth_folder_suffix+'/train_slp_rside_f_41to70_6006' + self.data_fp_suffix + extra_suffix + '.p')
            database_file_m.append(self.data_fp_prefix + synth_folder+self.synth_folder_suffix+'/train_slp_lay_m_41to70_6597' + self.data_fp_suffix + extra_suffix + '.p')
            database_file_m.append(self.data_fp_prefix + synth_folder+self.synth_folder_suffix+'/train_slp_lside_m_41to70_5935' + self.data_fp_suffix + extra_suffix + '.p')
            database_file_m.append(self.data_fp_prefix + synth_folder+self.synth_folder_suffix+'/train_slp_rside_m_41to70_5817' + self.data_fp_suffix + extra_suffix + '.p')

            database_file_f.append(self.data_fp_prefix + synth_folder+self.synth_folder_suffix+'/train_slp_lay_f_71to80_2184' + self.data_fp_suffix + extra_suffix + '.p')
            database_file_f.append(self.data_fp_prefix + synth_folder+self.synth_folder_suffix+'/train_slp_lside_f_71to80_2058' + self.data_fp_suffix + extra_suffix + '.p')
            database_file_f.append(self.data_fp_prefix + synth_folder+self.synth_folder_suffix+'/train_slp_rside_f_71to80_2010' + self.data_fp_suffix + extra_suffix + '.p')
            database_file_m.append(self.data_fp_prefix + synth_folder+self.synth_folder_suffix+'/train_slp_lay_m_71to80_2188' + self.data_fp_suffix + extra_suffix + '.p')
            database_file_m.append(self.data_fp_prefix + synth_folder+self.synth_folder_suffix+'/train_slp_lside_m_71to80_2002' + self.data_fp_suffix + extra_suffix + '.p')
            database_file_m.append(self.data_fp_prefix + synth_folder+self.synth_folder_suffix+'/train_slp_rside_m_71to80_1939' + self.data_fp_suffix + extra_suffix + '.p')

        # print(self.opt.mod)

        elif self.opt.mod==2 and self.opt.X_is=='B':
            # pass
            database_file_f.append(self.data_fp_prefix + synth_folder+self.synth_folder_suffix+'/train_slp_lay_f_71to80_2184' + self.data_fp_suffix + extra_suffix + '.p')
            database_file_f.append(self.data_fp_prefix + synth_folder+self.synth_folder_suffix+'/train_slp_lside_f_71to80_2058' + self.data_fp_suffix + extra_suffix + '.p')
            database_file_f.append(self.data_fp_prefix + synth_folder+self.synth_folder_suffix+'/train_slp_rside_f_71to80_2010' + self.data_fp_suffix + extra_suffix + '.p')
            database_file_m.append(self.data_fp_prefix + synth_folder+self.synth_folder_suffix+'/train_slp_lay_m_71to80_2188' + self.data_fp_suffix + extra_suffix + '.p')
            database_file_m.append(self.data_fp_prefix + synth_folder+self.synth_folder_suffix+'/train_slp_lside_m_71to80_2002' + self.data_fp_suffix + extra_suffix + '.p')
            database_file_m.append(self.data_fp_prefix + synth_folder+self.synth_folder_suffix+'/train_slp_rside_m_71to80_1939' + self.data_fp_suffix + extra_suffix + '.p')
        
        else:
            pass

        return database_file_f, database_file_m



    def get_qt_slpsynth_pressurepose(self, is_train = True, extra_suffix = ''):

        database_file_m = []
        database_file_f = []

        if len(self.data_fp_suffix) > 20:
            synth_folder = 'mod1est_synth'
        else:
            synth_folder = 'synth'

        if is_train == True:
            # SLP partition - xx,xxx train + xx,xxx test
            database_file_f.append(self.data_fp_prefix + synth_folder+self.synth_folder_suffix+'/train_slp_lay_f_1to40_8549' + self.data_fp_suffix + extra_suffix + '.p')
            database_file_m.append(self.data_fp_prefix + synth_folder+self.synth_folder_suffix+'/train_slp_lay_m_1to40_8493' + self.data_fp_suffix + extra_suffix + '.p')

        else:
            pass
            #database_file_f.append(self.data_fp_prefix + synth_folder+self.synth_folder_suffix+'/train_slp_lay_f_71to80_2184' + self.data_fp_suffix + extra_suffix + '.p')
            #database_file_m.append(self.data_fp_prefix + synth_folder+self.synth_folder_suffix+'/train_slp_lay_m_71to80_2188' + self.data_fp_suffix + extra_suffix + '.p')

        return database_file_f, database_file_m

