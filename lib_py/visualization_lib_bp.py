#!/usr/bin/env python
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import *
import matplotlib.gridspec as gridspec
import math
from matplotlib import rc

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=False)

import random

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
NUMOFTAXELS_X = 74#73 #taxels
NUMOFTAXELS_Y = 27#30
INTER_SENSOR_DISTANCE = 0.0286#metres
LOW_TAXEL_THRESH_X = 0
LOW_TAXEL_THRESH_Y = 0
HIGH_TAXEL_THRESH_X = (NUMOFTAXELS_X - 1)
HIGH_TAXEL_THRESH_Y = (NUMOFTAXELS_Y - 1)




class VisualizationLib():

    def print_error_train(self, target, score, output_size, loss_vector_type = None, data = None, printerror = True):

        if target[0, 0] == target[0, 9]:
            for i in range(24):
                target[:, 2+i*3] = 0
                score[:, 2+i*3] = 0
            for i in [0,3,6,9,10,11,13,14,22,23]:
                target[:, 0+i*3] = 0
                target[:, 1+i*3] = 0
                score[:, 0+i*3] = 0
                score[:, 1+i*3] = 0

        error = (score - target)

        error = np.reshape(error, (error.shape[0], output_size[0], output_size[1]))

        error_norm = np.expand_dims(np.linalg.norm(error, axis = 2),2)
        error = np.concatenate((error, error_norm), axis = 2)

        error_avg = np.mean(error, axis=0) / 10 #convert from mm to cm

       # for i in error_avg[:, 3]*10:
       #     print i

        error_avg = np.reshape(error_avg, (output_size[0], output_size[1]+1))
        error_avg_print = np.reshape(np.array(["%.2f" % w for w in error_avg.reshape(error_avg.size)]),
                                     (output_size[0], output_size[1] + 1))


        error_avg_print = np.transpose(np.concatenate(([['Average Error for Last Batch', '       ',
                                                        'Pelvis ', 'L Hip  ', 'R Hip  ', 'Spine 1', 'L Knee ', 'R Knee ',
                                                        'Spine 2', 'L Ankle', 'R Ankle', 'Spine 3', 'L Foot ', 'R Foot ',
                                                        'Neck   ', 'L Sh.in', 'R Sh.in', 'Head   ', 'L Sh.ou', 'R Sh.ou',
                                                        'L Elbow', 'R Elbow', 'L Wrist', 'R Wrist', 'L Hand ', 'R Hand ']], np.transpose(
            np.concatenate(([['', '', '', ''], [' x, cm ', ' y, cm ', ' z, cm ', '  norm ']], error_avg_print))))))
        # if printerror == True:
        #     print(data, error_avg_print)


        error_std = np.std(error, axis=0) / 10

        #for i in error_std[:, 3]*10:
        #    print i

        error_std = np.reshape(error_std, (output_size[0], output_size[1] + 1))
        error_std_print = np.reshape(np.array(["%.2f" % w for w in error_std.reshape(error_std.size)]),
                                     (output_size[0], output_size[1] + 1))

        error_std_print = np.transpose(np.concatenate(([['Error Standard Deviation for Last Batch', '       ',
                                                        'Pelvis ', 'L Hip  ', 'R Hip  ', 'Spine 1', 'L Knee ', 'R Knee ',
                                                        'Spine 2', 'L Ankle', 'R Ankle', 'Spine 3', 'L Foot ', 'R Foot ',
                                                        'Neck   ', 'L Sh.in', 'R Sh.in', 'Head   ', 'L Sh.ou', 'R Sh.ou',
                                                        'L Elbow', 'R Elbow', 'L Wrist', 'R Wrist', 'L Hand ', 'R Hand ']], np.transpose(
                np.concatenate(([['', '', '', ''], ['x, cm', 'y, cm', 'z, cm', '  norm ']], error_std_print))))))
        #if printerror == True:
        #    print data, error_std_print
        error_norm = np.squeeze(error_norm, axis = 2)

        #return error_avg[:,3], error_std[:,3]
        return error_norm, error_avg[:,3], error_std[:,3]

    def print_error_val(self, target, score, output_size, loss_vector_type = None, data = None, printerror = True):

        if target.shape[1] == 72:
            target = target.reshape(-1, 24, 3)
            target = np.stack((target[:, 15, :],
                               target[:, 3, :],
                               target[:, 19, :],
                               target[:, 18, :],
                               target[:, 21, :],
                               target[:, 20, :],
                               target[:, 5, :],
                               target[:, 4, :],
                               target[:, 8, :],
                               target[:, 7, :],), axis = 1)

            score = score.reshape(-1, 24, 3)
            score = np.stack((score[:, 15, :],
                               score[:, 3, :],
                               score[:, 19, :],
                               score[:, 18, :],
                               score[:, 21, :],
                               score[:, 20, :],
                               score[:, 5, :],
                               score[:, 4, :],
                               score[:, 8, :],
                               score[:, 7, :],), axis = 1)


        error = (score - target)

        #print error.shape
        error = np.reshape(error, (error.shape[0], output_size[0], output_size[1]))

        #print error.shape

        error_norm = np.expand_dims(np.linalg.norm(error, axis = 2),2)
        error = np.concatenate((error, error_norm), axis = 2)

        error_avg = np.mean(error, axis=0) / 10

        #for i in error_avg[:, 3]*10:
        #    print i

        error_avg = np.reshape(error_avg, (output_size[0], output_size[1]+1))
        error_avg_print = np.reshape(np.array(["%.2f" % w for w in error_avg.reshape(error_avg.size)]),
                                     (output_size[0], output_size[1] + 1))


        error_avg_print = np.transpose(np.concatenate(([['Average Error for Last Batch', '       ', 'Head   ',
                                                   'Torso  ', 'R Elbow', 'L Elbow', 'R Hand ', 'L Hand ',
                                                   'R Knee ', 'L Knee ', 'R Foot ', 'L Foot ']], np.transpose(
            np.concatenate(([['', '', '', ''], [' x, cm ', ' y, cm ', ' z, cm ', '  norm ']], error_avg_print))))))
        # if printerror == True:
        #     print(data, error_avg_print)


        error_std = np.std(error, axis=0) / 10

        #for i in error_std[:, 3]*10:
        #    print i

        error_std = np.reshape(error_std, (output_size[0], output_size[1] + 1))
        error_std_print = np.reshape(np.array(["%.2f" % w for w in error_std.reshape(error_std.size)]),
                                     (output_size[0], output_size[1] + 1))

        error_std_print = np.transpose(np.concatenate(([['Error Standard Deviation for Last Batch', '       ', 'Head   ', 'Torso  ',
                              'R Elbow', 'L Elbow', 'R Hand ', 'L Hand ', 'R Knee ', 'L Knee ',
                              'R Foot ', 'L Foot ']], np.transpose(
                np.concatenate(([['', '', '', ''], ['x, cm', 'y, cm', 'z, cm', '  norm ']], error_std_print))))))
        #if printerror == True:
        #    print data, error_std_print
        error_norm = np.squeeze(error_norm, axis = 2)

        #return error_avg[:,3], error_std[:,3]
        return error_norm, error_avg[:,3], error_std[:,3]

    def visualize_error_from_distance(self, bed_distance, error_norm):
        plt.close()
        fig = plt.figure()
        ax = []
        for joint in range(0, bed_distance.shape[1]):
            ax.append(joint)
            if bed_distance.shape[1] <= 5:
                ax[joint] = fig.add_subplot(1, bed_distance.shape[1], joint + 1)
            else:
                #print math.ceil(bed_distance.shape[1]/2.)
                ax[joint] = fig.add_subplot(2, math.ceil(bed_distance.shape[1]/2.), joint + 1)
            ax[joint].set_xlim([0, 0.7])
            ax[joint].set_ylim([0, 1])
            ax[joint].set_title('Joint ' + str(joint) + ' error')
            ax[joint].plot(bed_distance[:, joint], error_norm[:, joint], 'r.')
        plt.show()

    def make_popup_2D_plot(self, x1, y1, x2=None, y2=None):



        plt.plot(x1, y1)
        if x2 is not None:
            plt.plot(x2, y2, 'r')

        plt.xlim(0, 500)
        plt.ylim(0, 2000)


        plt.show()


    def get_standard_viz_maps(self, im_display_idx, INPUT_DICT, OUTPUT_DICT, VIZ_DICT, CTRL_PNL):
        if CTRL_PNL['mesh_recon_map_labels'] == True:  # pmr regression
            VIZ_DICT['cntct_in'] = INPUT_DICT['batch_images'][im_display_idx, 0, :].squeeze().cpu()   # contact
            VIZ_DICT['pimage_in'] = INPUT_DICT['batch_images'][im_display_idx, 1, :].squeeze().cpu() # pmat
            VIZ_DICT['sobel_in'] = INPUT_DICT['batch_images'][im_display_idx, 2, :].squeeze().cpu()  # sobel
            VIZ_DICT['pmap_recon'] = (OUTPUT_DICT['batch_mdm_est'][im_display_idx, :, :].squeeze() * -1).cpu().data  # est depth output
            VIZ_DICT['cntct_recon'] = (OUTPUT_DICT['batch_cm_est'][im_display_idx, :, :].squeeze()).cpu().data  # est depth output
            VIZ_DICT['pmap_recon_gt'] = (INPUT_DICT['batch_mdm_gt'][im_display_idx, :, :].squeeze() * -1).cpu().data  # ground truth depth
            VIZ_DICT['cntct_recon_gt'] = (INPUT_DICT['batch_cm_gt'][im_display_idx, :, :].squeeze()).cpu().data / 100.  # ground truth depth
        else:
            VIZ_DICT['cntct_in'] = INPUT_DICT['batch_images'][im_display_idx, 0, :].squeeze().cpu()  # contact
            VIZ_DICT['pimage_in'] = INPUT_DICT['batch_images'][im_display_idx, 1, :].squeeze().cpu()  # pmat
            VIZ_DICT['sobel_in'] = INPUT_DICT['batch_images'][im_display_idx, 2, :].squeeze().cpu()  # sobel
            VIZ_DICT['pmap_recon'] = None
            VIZ_DICT['cntct_recon'] = None
            VIZ_DICT['pmap_recon_gt'] = None
            VIZ_DICT['cntct_recon_gt'] = None



        if CTRL_PNL['recon_map_input_est'] == True:  # this is a network 2 option ONLY
            VIZ_DICT['pmap_recon_in'] = INPUT_DICT['batch_images'][im_display_idx, 1, :].squeeze().cpu() # pmat
            VIZ_DICT['cntct_recon_in'] = INPUT_DICT['batch_images'][im_display_idx, 2, :].squeeze().cpu() # pmat
            VIZ_DICT['pimage_in'] = INPUT_DICT['batch_images'][im_display_idx, 3, :].squeeze().cpu()  # pmat
            VIZ_DICT['sobel_in'] = INPUT_DICT['batch_images'][im_display_idx, 4, :].squeeze().cpu()   # sobel
            if CTRL_PNL['depth_in'] == True:
                VIZ_DICT['depth_in'] = INPUT_DICT['batch_images'][im_display_idx, 5, :].squeeze().cpu()  #sobel
            else:
                VIZ_DICT['depth_in'] = None
        else:
            VIZ_DICT['pmap_recon_in'] = None
            VIZ_DICT['cntct_recon_in'] = None
            if CTRL_PNL['depth_in'] == True:
                VIZ_DICT['depth_in'] = INPUT_DICT['batch_images'][im_display_idx, 3, :].squeeze().cpu()  #sobel
            else:
                VIZ_DICT['depth_in'] = None

        return VIZ_DICT




    def get_depthnet_viz_maps(self, im_display_idx, INPUT_DICT, OUTPUT_DICT, VIZ_DICT, CTRL_PNL):
        VIZ_DICT['depth_in'] = (INPUT_DICT['batch_images'][im_display_idx, 0, :, :].squeeze()).cpu().data
        if CTRL_PNL['mesh_recon_map_output'] == True and CTRL_PNL['train_only_CAL'] == False and CTRL_PNL['train_only_betanet'] == False:
            VIZ_DICT['pmap_recon'] = (OUTPUT_DICT['batch_mdm_est'][im_display_idx, :, :].squeeze() * -1).cpu().data  # est depth output
            VIZ_DICT['cntct_recon'] = (OUTPUT_DICT['batch_cm_est'][im_display_idx, :, :].squeeze()).cpu().data  # est depth output
        else:
            VIZ_DICT['pmap_recon'] = None
            VIZ_DICT['cntct_recon'] = None

        if CTRL_PNL['mesh_recon_map_labels'] == True:
            VIZ_DICT['pmap_recon_gt'] = (INPUT_DICT['batch_mdm_gt'][im_display_idx, :, :].squeeze() * -1).cpu().data  # ground truth depth
            VIZ_DICT['cntct_recon_gt'] = (INPUT_DICT['batch_cm_gt'][im_display_idx, :, :].squeeze()).cpu().data / 100.  # ground truth depth
        else:
            VIZ_DICT['pmap_recon_gt'] = None
            VIZ_DICT['cntct_recon_gt'] = None

        if CTRL_PNL['recon_map_input_est'] == True:
            VIZ_DICT['pmap_recon_in'] = (INPUT_DICT['batch_images'][im_display_idx, 1, :, :].squeeze() ).cpu().data
            VIZ_DICT['cntct_recon_in'] = (INPUT_DICT['batch_images'][im_display_idx, 2, :, :].squeeze()).cpu().data / 100.
        else:
            VIZ_DICT['pmap_recon_in'] = None
            VIZ_DICT['cntct_recon_in'] = None


        #print(torch.max(VIZ_DICT['p_img_cntct']), torch.max(VIZ_DICT['cntct_recon']), torch.max(VIZ_DICT['cntct_recon_gt']), 'cntct')

        return VIZ_DICT




    def get_fcn_recon_viz_maps(self, im_display_idx, INPUT_DICT, OUTPUT_DICT, VIZ_DICT, CTRL_PNL):

        try:
            VIZ_DICT['p_img'] = (INPUT_DICT['batch_pimg'][im_display_idx, :, :].squeeze()).cpu().data  #
            VIZ_DICT['p_img_cntct'] = (INPUT_DICT['batch_pimg_cntct'][im_display_idx, :, :].squeeze()).cpu().data #
        except:
            pass

        if CTRL_PNL['mesh_recon_map_labels'] == True or CTRL_PNL['mesh_recon_map_output'] == True or CTRL_PNL['CNN'] == 'resnetunet':
            if CTRL_PNL['train_only_betanet'] == False:
                VIZ_DICT['p_img_est'] = OUTPUT_DICT['batch_pimg_est'][im_display_idx, :, :].squeeze().cpu().data  # est depth output
                VIZ_DICT['p_img_cntct_est'] = OUTPUT_DICT['batch_pimg_cntct_est'][im_display_idx, :, :].squeeze().cpu().data  # est depth output
        else:
            VIZ_DICT['p_img_est'] = None
            VIZ_DICT['p_img_cntct_est'] = None



        if CTRL_PNL['depth_out_unet'] == True:
            VIZ_DICT['dimg_est'] = OUTPUT_DICT['batch_dimg_est'][im_display_idx, :].squeeze().detach().cpu().numpy()
            VIZ_DICT['dimg_cntct_est'] = OUTPUT_DICT['batch_dimg_cntct_est'][im_display_idx, :].squeeze().detach().cpu().numpy()
        else:
            VIZ_DICT['dimg_est'] = None
            VIZ_DICT['dimg_cntct_est'] = None


        if CTRL_PNL['depth_out_unet'] == True and CTRL_PNL['onlyhuman_labels'] == True:
            VIZ_DICT['dimg_gt'] = INPUT_DICT['batch_dimg_noblanket_gt'][im_display_idx, :].squeeze().detach().cpu().numpy()
            VIZ_DICT['dimg_cntct_gt'] = INPUT_DICT['batch_dimg_noblanket_cntct_gt'][im_display_idx, :].squeeze().detach().cpu().numpy()
        else:
            VIZ_DICT['dimg_gt'] = None
            VIZ_DICT['dimg_cntct_gt'] = None



        return VIZ_DICT




    def visualize_depth_net(self, VIZ_DICT,
                                targets_raw=None, scores_net1 = None, scores_net2 = None,
                                block = False, max_depth = 2200, is_testing = False):


        pimage_mult = 1.
        pmap_recon_in_mult = 1.
        cntct_recon_in_mult = 1.
        depth_in_mult = 1.


        if VIZ_DICT['depth_in'].shape[0] == 128: depth_in_mult = 2.

        plt.close()
        plt.pause(0.0001)

        # set options
        num_subplots = 5
        if VIZ_DICT['pmap_recon'] is not None or ['dimg_est'] is not None:
            num_subplots += 2
            shift_pressuremap = 2
        else:
            shift_pressuremap = 0

        depth_fcn_recon_col = 2
        depth_fcn_recon_gt_col = 5


        mult_bottom_row_size = 1.0
        unet_row = 1



        fig = plt.figure(tight_layout=True, figsize = (1.5*num_subplots*.8, 5*.8*mult_bottom_row_size))
        gs = gridspec.GridSpec(1+unet_row, num_subplots)


        plt.pause(0.0001)
        # use LaTeX fonts in the plot
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')


        ax1 = fig.add_subplot(gs[:, 0:2])
        ax1.set_xlim([-10.0*depth_in_mult, 37.0*depth_in_mult])
        ax1.set_ylim([74.0*depth_in_mult, -10.0*depth_in_mult])
        ax1.set_facecolor('cyan')

        ax1.imshow(VIZ_DICT['depth_in'], interpolation='nearest', cmap=plt.cm.inferno, origin='upper', vmin=max_depth - 700, vmax=max_depth)
        ax1.set_title('Training Sample \n Depth Image, \n Targets and Estimates')
        ax1.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)


        ax13 = fig.add_subplot(gs[1, 3+shift_pressuremap])
        ax13.set_xlim([-pimage_mult, 27.0*pimage_mult])
        ax13.set_ylim([64.0*pimage_mult, -pimage_mult])
        ax13.imshow(VIZ_DICT['p_img'], interpolation='nearest', cmap=plt.cm.jet, origin='upper', vmin=0, vmax=35)
        ax13.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)
        #ax13.set_ylabel('INPUT')
        ax13.set_title(r'$\mathcal{P}$')
        #ax13.set_title(r'$\mathcal{P}_m$')

        ax13 = fig.add_subplot(gs[1, 4+shift_pressuremap])
        ax13.set_xlim([-pimage_mult, 27.0*pimage_mult])
        ax13.set_ylim([64.0*pimage_mult, -pimage_mult])
        ax13.imshow(100-100*VIZ_DICT['p_img_cntct'], interpolation='nearest', cmap= plt.cm.gray, origin='upper', vmin=0, vmax=100)
        ax13.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)
        #ax13.set_ylabel('INPUT')
        #ax13.set_title(r'$\mathcal{C}$')
        ax13.set_title(r'$\mathcal{C}_p$')




        if VIZ_DICT['p_img_est'] is not None:
            ax33 = fig.add_subplot(gs[0, 3+shift_pressuremap])
            ax33.set_xlim([-pimage_mult, 27.0*pimage_mult])
            ax33.set_ylim([64.0*pimage_mult, -pimage_mult])
            ax33.imshow(VIZ_DICT['p_img_est'], interpolation='nearest', cmap=plt.cm.jet, origin='upper', vmin=0, vmax=35)
            ax33.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)
            #ax33.set_ylabel('INPUT')
            ax33.set_title(r'$\widehat{\mathcal{P}}$')

            ax33 = fig.add_subplot(gs[0, 4+shift_pressuremap])
            ax33.set_xlim([-pimage_mult, 27.0*pimage_mult])
            ax33.set_ylim([64.0*pimage_mult, -pimage_mult])

            ax33.imshow(100-100*VIZ_DICT['p_img_cntct_est'], interpolation='nearest', cmap= plt.cm.gray, origin='upper', vmin=0, vmax=100)
            ax33.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)
            #ax33.set_ylabel('INPUT')
            ax33.set_title(r'$\widehat{\mathcal{C}}_p$')




        q_out_subscr = str(1)
        if VIZ_DICT['pmap_recon_in'] is not None:
            q_out_subscr = str(2)

            if VIZ_DICT['dimg_est'] is not None:
                ax23b = fig.add_subplot(gs[0, 2])
                ax23b.set_xlim([-depth_in_mult, depth_in_mult*27.0])
                ax23b.set_ylim([depth_in_mult*64.0, -depth_in_mult])
                ax23b.imshow(VIZ_DICT['pmap_recon_in'], interpolation='nearest', cmap=plt.cm.inferno, origin='upper', vmin=max_depth - 700, vmax=max_depth)
                ax23b.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)
                ax23b.set_ylabel('INPUT')
                ax23b.set_title(r'$\widehat{D}_1$')
            else:
                ax23b = fig.add_subplot(gs[0, 2])
                ax23b.set_xlim([-depth_in_mult, depth_in_mult*27.0])
                ax23b.set_ylim([depth_in_mult*64.0, -depth_in_mult])
                #ax23b.imshow(3000 - VIZ_DICT['pmap_recon_in'], interpolation='nearest', cmap= plt.cm.viridis, origin='upper', vmin=0, vmax=3000)
                ax23b.imshow(-VIZ_DICT['pmap_recon_in'], interpolation='nearest', cmap= plt.cm.inferno, origin='upper', vmin=-900, vmax=100)
                ax23b.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)
                ax23b.set_ylabel('INPUT')
                #ax23b.set_title(r'$\widehat{Q}^{+}_1$')
                ax23b.set_title(r'$\widehat{\mathcal{D}}^{+}$')

            ax24b = fig.add_subplot(gs[1, 2])
            ax24b.set_xlim([-depth_in_mult, depth_in_mult*27.0])
            ax24b.set_ylim([depth_in_mult*64.0, -depth_in_mult])
            ax24b.imshow(100-VIZ_DICT['cntct_recon_in']*100, interpolation='nearest', cmap= plt.cm.gray, origin='upper', vmin=0, vmax=100)
            ax24b.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)
            ax24b.set_ylabel('INPUT')
            #ax24b.set_title(r'$\widehat{C}_{O,1}$')
            ax24b.set_title(r'$\widehat{\mathcal{C}}_{d^+}$')


        if VIZ_DICT['pmap_recon'] is not None:
            ax23 = fig.add_subplot(gs[0, 3])
            ax23.set_xlim([-1.0, 27.0])
            ax23.set_ylim([64.0, -1.0])
            ax23.imshow(VIZ_DICT['pmap_recon'], interpolation='nearest', cmap= plt.cm.viridis, origin='upper', vmin=0, vmax=100)
            ax23.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)
            ax23.set_ylabel('OUTPUT')
            #ax23.set_title(r'$\widehat{Q}^{-}_'+q_out_subscr+'$')
            ax23.set_title(r'$\widehat{\mathcal{P}}^{+}$')

            ax24 = fig.add_subplot(gs[0, 4])
            ax24.set_xlim([-1.0, 27.0])
            ax24.set_ylim([64.0, -1.0])
            ax24.imshow(100 - VIZ_DICT['cntct_recon']*100, interpolation='nearest', cmap= plt.cm.gray, origin='upper', vmin=0, vmax=100)
            ax24.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)
            #ax24.set_title(r'$\widehat{C}_{O,'+q_out_subscr+'}$')
            ax24.set_title(r'$\widehat{\mathcal{C}}_{p^+}$')

        elif VIZ_DICT['dimg_est'] is not None:
            ax23 = fig.add_subplot(gs[0, 3])
            ax23.set_xlim([-depth_in_mult, depth_in_mult*27.0])
            ax23.set_ylim([depth_in_mult*64.0, -depth_in_mult])
            ax23.imshow(VIZ_DICT['dimg_est'], interpolation='nearest', cmap=plt.cm.inferno, origin='upper', vmin=max_depth - 700, vmax=max_depth)
            ax23.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)
            ax23.set_ylabel('OUTPUT')
            ax23.set_title(r'$\widehat{D}_'+q_out_subscr+'$')

            ax24 = fig.add_subplot(gs[0, 4])
            ax24.set_xlim([-depth_in_mult, depth_in_mult*27.0])
            ax24.set_ylim([depth_in_mult*64.0, -depth_in_mult])
            ax24.imshow(100 - VIZ_DICT['dimg_cntct_est']*100, interpolation='nearest', cmap= plt.cm.gray, origin='upper', vmin=0, vmax=100)
            ax24.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)
            ax24.set_title(r'$\widehat{C}_{O,'+q_out_subscr+'}$')



        if VIZ_DICT['pmap_recon_gt'] is not None:
            ax27 = fig.add_subplot(gs[1, 3])
            ax27.set_xlim([-1.0, 27.0])
            ax27.set_ylim([64.0, -1.0])
            ax27.imshow(VIZ_DICT['pmap_recon_gt'], interpolation='nearest', cmap= plt.cm.viridis, origin='upper', vmin=0, vmax=100)
            ax27.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)
            ax27.set_ylabel('GROUND TRUTH')
            #ax27.set_title(r'$Q^{-}$')
            ax27.set_title(r'$\mathcal{P}^{+}$')

            ax28 = fig.add_subplot(gs[1, 4])
            ax28.set_xlim([-1.0, 27.0])
            ax28.set_ylim([64.0, -1.0])
            ax28.imshow(100 - VIZ_DICT['cntct_recon_gt']*100, interpolation='nearest', cmap= plt.cm.gray, origin='upper', vmin=0, vmax=100)
            ax28.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)
            #ax28.set_title(r'$C_{O}$')
            ax28.set_title(r'$\mathcal{C}_{p^+}$')

        elif VIZ_DICT['dimg_gt'] is not None:
            ax27 = fig.add_subplot(gs[1, 3])
            ax27.set_xlim([-depth_in_mult, depth_in_mult*27.0])
            ax27.set_ylim([depth_in_mult*64.0, -depth_in_mult])
            ax27.imshow(VIZ_DICT['dimg_gt'], interpolation='nearest', cmap=plt.cm.inferno, origin='upper', vmin=max_depth - 700, vmax=max_depth)
            ax27.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)
            ax27.set_ylabel('OUTPUT')
            ax27.set_title(r'$D_{nb}$')

            ax28 = fig.add_subplot(gs[1, 4])
            ax28.set_xlim([-depth_in_mult, depth_in_mult*27.0])
            ax28.set_ylim([depth_in_mult*64.0, -depth_in_mult])
            ax28.imshow(100 - VIZ_DICT['dimg_cntct_gt']*100, interpolation='nearest', cmap= plt.cm.gray, origin='upper', vmin=0, vmax=100)
            ax28.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)
            ax28.set_title(r'$C_{d,nb}$')

        #print(scores_net1.size(), targets_raw.size(), 'viz scores and targets shape')

        # Visualize estimated from training set
        self.plot_joint_markers(scores_net1, depth_in_mult, ax1, 'yellow')

        # plot the skeleton links for the SLP dataset, which has 14 2D joint labels
        if scores_net1 is not None:
            if scores_net1.size()[0] == 24 or scores_net1.size()[0] == 25:
                self.plot_skeleton_links(scores_net1, depth_in_mult, ax1, 'yellow')


        # Visualize targets of training set
        self.plot_joint_markers(targets_raw, depth_in_mult, ax1, 'green')

        #plot the skeleton links for the SLP dataset, which has 14 2D joint labels
        if targets_raw is not None:
            self.plot_skeleton_links(targets_raw, depth_in_mult, ax1, 'green')


        #fig.savefig('/home/henry/data/blah.png', dpi=400)
        plt.show(block=block)





    def visualize_pressure_map(self, VIZ_DICT,
                                targets_raw=None, scores_net1 = None, scores_net2 = None,
                                block = False, max_depth = 2200):



        max_vals = []
        max_vals.append(float(torch.max(VIZ_DICT['pimage_in']).cpu().data.numpy()))
        sum_vals = []
        sum_vals.append(float(torch.sum(VIZ_DICT['pimage_in']).cpu().data.numpy()))
        print("max: ", max_vals, "     sum: ", sum_vals)


        pimage_in_mult = 1.
        cntct_in_mult = 1.
        sobel_in_mult = 1.
        pmap_recon_in_mult = 1.
        cntct_recon_in_mult = 1.
        depth_in_mult = 1.

        if VIZ_DICT['pimage_in'].shape[0] == 128: pimage_in_mult = 2.
        elif VIZ_DICT['pimage_in'].shape[0] == 256: pimage_in_mult = 4.
        if VIZ_DICT['cntct_in'] is not None:
            if VIZ_DICT['cntct_in'].shape[0] == 128: cntct_in_mult = 2.
            elif VIZ_DICT['cntct_in'].shape[0] == 256: cntct_in_mult = 4.
        if VIZ_DICT['sobel_in'] is not None:
            if VIZ_DICT['sobel_in'].shape[0] == 128: sobel_in_mult = 2.
        if VIZ_DICT['pmap_recon_in'] is not None:
            if VIZ_DICT['pmap_recon_in'].shape[0] == 128: pmap_recon_in_mult = 2.
        if VIZ_DICT['cntct_recon_in'] is not None:
            if VIZ_DICT['cntct_recon_in'].shape[0] == 128: cntct_recon_in_mult = 2.
        if VIZ_DICT['depth_in'] is not None:
            if VIZ_DICT['depth_in'].shape[0] == 128: depth_in_mult = 2.

        plt.close()
        # set options
        num_subplots = 5
        if VIZ_DICT['pmap_recon_in'] is not None:
            num_subplots += 3
        elif VIZ_DICT['pmap_recon_gt'] is not None:
            num_subplots += 2
        elif VIZ_DICT['depth_in'] is not None or VIZ_DICT['pmap_fcn_recon_gt'] is not None or VIZ_DICT['depth_fcn_recon_gt'] is not None:
            num_subplots += 1
        pmap_fcn_recon_col = 2
        depth_fcn_recon_col = 2
        pmap_fcn_recon_gt_col = 4
        depth_fcn_recon_gt_col = 5
        if VIZ_DICT['pmap_fcn_recon_gt'] is not None and VIZ_DICT['depth_fcn_recon_gt'] is not None:
            if num_subplots <= 8: num_subplots += 1
            depth_fcn_recon_col = 3
            pmap_fcn_recon_gt_col = 5
            depth_fcn_recon_gt_col = 6


        mult_bottom_row_size = 1.0
        unet_row = 1
        if VIZ_DICT['pmap_recon']  is not None or VIZ_DICT['pmap_recon_gt'] is not None:
            if VIZ_DICT['pmap_fcn_recon'] is not None or VIZ_DICT['depth_fcn_recon'] is not None:
                mult_bottom_row_size = 1.5
                unet_row = 2



        fig = plt.figure(tight_layout=True, figsize = (1.5*num_subplots*.8, 5*.8*mult_bottom_row_size))
        gs = gridspec.GridSpec(1+unet_row, num_subplots)


        plt.pause(0.0001)
        # use LaTeX fonts in the plot
        plt.rc('text', usetex=False)
        plt.rc('font', family='serif')
        plt.pause(0.0001)



        ax1 = fig.add_subplot(gs[:, 0:2])
        ax1.set_xlim([-10.0*pimage_in_mult, 37.0*pimage_in_mult])
        ax1.set_ylim([74.0*pimage_in_mult, -10.0*pimage_in_mult])
        ax1.set_facecolor('cyan')
        if torch.sum(VIZ_DICT['pimage_in']) != 0.:
            ax1.imshow(VIZ_DICT['pimage_in'], interpolation='nearest', cmap=
            plt.cm.jet, origin='upper', vmin=0, vmax=100)
            ax1.set_title('Training Sample \n Pressure Image, \n Targets and Estimates')
        else:
            ax1.imshow(VIZ_DICT['depth_in'], interpolation='nearest', cmap=
            plt.cm.inferno, origin='upper', vmin=0, vmax=500)
            ax1.set_title('Training Sample \n Depth Image, \n Targets and Estimates')
        ax1.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)
        ax1.set_title('Training Sample \n Pressure Image, \n Targets and Estimates')



        ax13 = fig.add_subplot(gs[0, 2])
        ax13.set_xlim([-pimage_in_mult, 27.0*pimage_in_mult])
        ax13.set_ylim([64.0*pimage_in_mult, -pimage_in_mult])
        ax13.imshow(VIZ_DICT['pimage_in'], interpolation='nearest', cmap=
        plt.cm.jet, origin='upper', vmin=0, vmax=100)
        ax13.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)
        ax13.set_ylabel('INPUT')
        ax13.set_title(r'$\mathcal{P}$')

        ax14 = fig.add_subplot(gs[0, 3])
        ax14.set_xlim([-cntct_in_mult, 27.0 * cntct_in_mult])
        ax14.set_ylim([64.0 * cntct_in_mult, -cntct_in_mult])
        ax14.imshow(100 - VIZ_DICT['cntct_in'], interpolation='nearest', cmap=
        plt.cm.gray, origin='upper', vmin=0, vmax=100)
        ax14.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)
        ax14.set_title(r'$\mathcal{C}_I$')

        ax15 = fig.add_subplot(gs[0, 4])
        ax15.set_xlim([-sobel_in_mult, 27.0 * sobel_in_mult])
        ax15.set_ylim([64.0 * sobel_in_mult, -sobel_in_mult])
        ax15.imshow(VIZ_DICT['sobel_in'], interpolation='nearest', cmap=
        plt.cm.jet, origin='upper', vmin=0, vmax=100)
        ax15.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)
        ax15.set_title(r'$\mathcal{E}$')


        if VIZ_DICT['depth_in'] is not None:
            ax18 = fig.add_subplot(gs[0, 5])
            ax18.set_xlim([-depth_in_mult, 27.0 * depth_in_mult])
            ax18.set_ylim([64.0 * depth_in_mult, -depth_in_mult])
            ax18.imshow(VIZ_DICT['depth_in'], interpolation='nearest', cmap=
            plt.cm.inferno, origin='upper', vmin=max_depth - 700, vmax=max_depth)
            ax18.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom=False, top=False)
            ax18.set_title(r'$D$')


        if VIZ_DICT['pmap_recon_in'] is not None:
            ax16 = fig.add_subplot(gs[0, 6])
            ax16.set_xlim([-pmap_recon_in_mult, 27.0 * pmap_recon_in_mult])
            ax16.set_ylim([64.0 * pmap_recon_in_mult, -pmap_recon_in_mult])
            ax16.imshow(VIZ_DICT['pmap_recon_in'], interpolation='nearest', cmap=
            plt.cm.viridis, origin='upper', vmin=0, vmax=100)
            ax16.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom=False, top=False)
            ax16.set_title(r'$\widehat{Q}^{-}_1$')

            ax17 = fig.add_subplot(gs[0, 7])
            ax17.set_xlim([-cntct_recon_in_mult, 27.0 * cntct_recon_in_mult])
            ax17.set_ylim([64.0 * cntct_recon_in_mult, -cntct_recon_in_mult])
            ax17.imshow(100 - VIZ_DICT['cntct_recon_in']*100, interpolation='nearest', cmap=
            plt.cm.gray, origin='upper', vmin=0, vmax=100)
            ax17.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom=False, top=False)
            ax17.set_title(r'$\widehat{C}_{O,1}$')



        if VIZ_DICT['pmap_recon'] is not None:
            q_out_subscr = str(1)
            if VIZ_DICT['pmap_recon_in'] is not None:
                q_out_subscr = str(2)

            ax23 = fig.add_subplot(gs[1, 2])
            ax23.set_xlim([-1.0, 27.0])
            ax23.set_ylim([64.0, -1.0])
            ax23.imshow(VIZ_DICT['pmap_recon'], interpolation='nearest', cmap= plt.cm.viridis, origin='upper', vmin=0, vmax=100)
            ax23.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)
            ax23.set_ylabel('OUTPUT')
            ax23.set_title(r'$\widehat{Q}^{-}_'+q_out_subscr+'$')

            ax24 = fig.add_subplot(gs[1, 3])
            ax24.set_xlim([-1.0, 27.0])
            ax24.set_ylim([64.0, -1.0])
            ax24.imshow(100 - VIZ_DICT['cntct_recon']*100, interpolation='nearest', cmap= plt.cm.gray, origin='upper', vmin=0, vmax=100)
            ax24.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)
            ax24.set_title(r'$\widehat{C}_{O,'+q_out_subscr+'}$')

            #ax25 = fig.add_subplot(gs[1, 4])
            #ax25.set_xlim([-1.0, 27.0])
            #ax25.set_ylim([64.0, -1.0])
            #ax25.imshow(hover_recon, interpolation='nearest', cmap= plt.cm.viridis, origin='upper', vmin=0, vmax=100)
            #ax25.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)
            #ax25.set_title(r'$\widehat{Q}^{+}_'+q_out_subscr+'$')

        if VIZ_DICT['pmap_recon_gt'] is not None:
            ax27 = fig.add_subplot(gs[1, 5])
            ax27.set_xlim([-1.0, 27.0])
            ax27.set_ylim([64.0, -1.0])
            ax27.imshow(VIZ_DICT['pmap_recon_gt'], interpolation='nearest', cmap= plt.cm.viridis, origin='upper', vmin=0, vmax=100)
            ax27.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)
            ax27.set_ylabel('GROUND TRUTH')
            ax27.set_title(r'$Q^{-}$')

            ax28 = fig.add_subplot(gs[1, 6])
            ax28.set_xlim([-1.0, 27.0])
            ax28.set_ylim([64.0, -1.0])
            ax28.imshow(100 - VIZ_DICT['cntct_recon_gt']*100, interpolation='nearest', cmap= plt.cm.gray, origin='upper', vmin=0, vmax=100)
            ax28.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)
            ax28.set_title(r'$C_{O}$')


        if VIZ_DICT['depth_fcn_recon'] is not None:
            ax24 = fig.add_subplot(gs[unet_row, depth_fcn_recon_col])
            ax24.set_xlim([-depth_in_mult, 27.0 * depth_in_mult])
            ax24.set_ylim([64.0 * depth_in_mult, -depth_in_mult])
            ax24.imshow(VIZ_DICT['depth_fcn_recon'], interpolation='nearest', cmap=
            plt.cm.inferno, origin='upper', vmin=0, vmax=500)
            ax24.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom=False, top=False)
            if VIZ_DICT['pmap_fcn_recon'] is None: ax24.set_ylabel('UNET OUTPUT')
            ax24.set_title(r'$\widehat{D_{nb}}$')



            ax28 = fig.add_subplot(gs[unet_row, depth_fcn_recon_gt_col])
            ax28.set_xlim([-depth_in_mult, 27.0 * depth_in_mult])
            ax28.set_ylim([64.0 * depth_in_mult, -depth_in_mult])
            ax28.imshow(VIZ_DICT['depth_fcn_recon_gt'], interpolation='nearest', cmap=
            plt.cm.inferno, origin='upper', vmin=0, vmax=500)
            ax28.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom=False, top=False)
            if VIZ_DICT['pmap_fcn_recon_gt'] is None: ax28.set_ylabel('UNET GND TRUTH')
            ax28.set_title(r'$D_{nb}$')

        #print(scores_net1.size(), targets_raw.size(), 'viz scores and targets shape')

        # Visualize estimated from training set
        self.plot_joint_markers(scores_net1, pimage_in_mult, ax1, 'yellow')

        # plot the skeleton links for the SLP dataset, which has 14 2D joint labels

        if scores_net1 is not None:
            if scores_net1.size()[0] == 24 or scores_net1.size()[0] == 25:
                self.plot_skeleton_links(scores_net1, pimage_in_mult, ax1, 'yellow')




        # Visualize targets of training set
        self.plot_joint_markers(targets_raw, pimage_in_mult, ax1, 'green')

        #plot the skeleton links for the SLP dataset, which has 14 2D joint labels
        if targets_raw is not None:
            self.plot_skeleton_links(targets_raw, pimage_in_mult, ax1, 'green')


        #fig.savefig('/home/henry/data/blah.png', dpi=400)
        plt.show(block=block)







    def visualize_pressure_map_slp1(self, VIZ_DICT,
                                targets_raw=None, scores_net1 = None, scores_net2 = None,
                                block = False, title = ' '):

        scores_net1 = None

        pimage_in_mult = 1.
        cntct_in_mult = 1.
        sobel_in_mult = 1.
        pmap_recon_in_mult = 1.
        cntct_recon_in_mult = 1.
        depth_in_mult = 1.

        if VIZ_DICT['pimage_in'].shape[0] == 128: pimage_in_mult = 2.
        elif VIZ_DICT['pimage_in'].shape[0] == 256: pimage_in_mult = 4.
        if VIZ_DICT['cntct_in'] is not None:
            if VIZ_DICT['cntct_in'].shape[0] == 128: cntct_in_mult = 2.
            elif VIZ_DICT['cntct_in'].shape[0] == 256: cntct_in_mult = 4.
        if VIZ_DICT['sobel_in'] is not None:
            if VIZ_DICT['sobel_in'].shape[0] == 128: sobel_in_mult = 2.
        if VIZ_DICT['pmap_recon_in'] is not None:
            if VIZ_DICT['pmap_recon_in'].shape[0] == 128: pmap_recon_in_mult = 2.
        if VIZ_DICT['cntct_recon_in'] is not None:
            if VIZ_DICT['cntct_recon_in'].shape[0] == 128: cntct_recon_in_mult = 2.
        if VIZ_DICT['depth_in'] is not None:
            if VIZ_DICT['depth_in'].shape[0] == 128: depth_in_mult = 2.

        plt.close()
        plt.pause(0.0001)

        # set options
        num_subplots = 5
        if VIZ_DICT['pmap_recon_in'] is not None:
            num_subplots += 3
        elif VIZ_DICT['pmap_recon_gt'] is not None:
            num_subplots += 2
        elif VIZ_DICT['depth_in'] is not None or VIZ_DICT['pmap_fcn_recon_gt'] is not None or VIZ_DICT['depth_fcn_recon_gt'] is not None:
            num_subplots += 1
        pmap_fcn_recon_col = 2
        depth_fcn_recon_col = 2
        pmap_fcn_recon_gt_col = 4
        depth_fcn_recon_gt_col = 5
        if VIZ_DICT['pmap_fcn_recon_gt'] is not None and VIZ_DICT['depth_fcn_recon_gt'] is not None:
            if num_subplots <= 8: num_subplots += 1
            depth_fcn_recon_col = 3
            pmap_fcn_recon_gt_col = 5
            depth_fcn_recon_gt_col = 6


        mult_bottom_row_size = 1.0
        unet_row = 1
        if VIZ_DICT['pmap_recon']  is not None or VIZ_DICT['pmap_recon_gt'] is not None:
            if VIZ_DICT['pmap_fcn_recon'] is not None or VIZ_DICT['depth_fcn_recon'] is not None:
                mult_bottom_row_size = 1.5
                unet_row = 2



        fig = plt.figure(tight_layout=True, figsize = (1.5*num_subplots*.8, 5*.8*mult_bottom_row_size))
        gs = gridspec.GridSpec(1+unet_row, num_subplots)


        plt.pause(0.0001)
        # use LaTeX fonts in the plot
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')


        ax1 = fig.add_subplot(gs[:, 0:2])
        ax1.set_xlim([-10.0*pimage_in_mult, 37.0*pimage_in_mult])
        ax1.set_ylim([74.0*pimage_in_mult, -10.0*pimage_in_mult])
        ax1.set_facecolor('cyan')
        if torch.sum(VIZ_DICT['pimage_in']) != 0.:
            ax1.imshow(VIZ_DICT['pimage_in'], interpolation='nearest', cmap=
            plt.cm.jet, origin='upper', vmin=0, vmax=100)
            ax1.set_title('Pressure')
        else:
            ax1.imshow(VIZ_DICT['depth_in'], interpolation='nearest', cmap=
            plt.cm.inferno, origin='upper', vmin=0, vmax=500)
            ax1.set_title('Pressure')
        ax1.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)
        ax1.set_title('Pressure')



        ax2 = fig.add_subplot(gs[:, 2:4])
        ax2.set_xlim([-10.0*pimage_in_mult, 37.0*pimage_in_mult])
        ax2.set_ylim([74.0*pimage_in_mult, -10.0*pimage_in_mult])
        ax2.set_facecolor('black')
        ax2.imshow(VIZ_DICT['depth_in'], interpolation='nearest', cmap=
        plt.cm.inferno, origin='upper', vmin=1100, vmax=1900)
        ax2.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)
        ax2.set_title('Depth')




        # Visualize estimated from training set
        self.plot_joint_markers(scores_net1, pimage_in_mult, ax1, 'yellow')

        # plot the skeleton links for the SLP dataset, which has 14 2D joint labels
        if scores_net1 is not None:
            if scores_net1.size()[0] == 24:
                self.plot_skeleton_links(scores_net1, pimage_in_mult, ax1, 'yellow')




        # Visualize targets of training set
        self.plot_joint_markers(targets_raw, pimage_in_mult, ax1, 'green')
        self.plot_joint_markers(targets_raw, pimage_in_mult, ax2, 'green')

        #plot the skeleton links for the SLP dataset, which has 14 2D joint labels
        if targets_raw is not None:
            self.plot_skeleton_links(targets_raw, pimage_in_mult, ax1, 'green')
            self.plot_skeleton_links(targets_raw, pimage_in_mult, ax2, 'green')




        #fig.savefig('/home/henry/data/blah.png', dpi=400)
        plt.show(block=block)




    def plot_pimage_depth(self, pimage_synth, depth_synth_uncover, color_synth_uncover,
                          pimage_real_uncover, depth_real_uncover, color_real_uncover,
                          depth_synth, color_synth_cover,
                          pimage_real_cover, depth_real_cover, color_real_cover,
                          max_depth = 2200, save_criteria = [False, 0, '_']):

        import matplotlib.pyplot as plt
        # plt.ion()
        #fig, ax = plt.subplots()

        plt.close()
        plt.pause(0.0001)

        fig = plt.figure(tight_layout=True, figsize = (8.0, 6.0))
        gs = gridspec.GridSpec(2, 6)



        plt.pause(0.0001)
        # use LaTeX fonts in the plot
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')


        pimage_in_mult = 1.
        if color_synth_uncover is not None:
            ax9 = fig.add_subplot(gs[0, 0])
            ax9.set_xlim([-2.0, 54])
            ax9.set_ylim([128, -2.0])
            ax9.set_facecolor('cyan')
            ax9.imshow(color_synth_uncover)
            ax9.set_title('Synth \n Color \n Uncover')
            ax9.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)

        if pimage_synth is not None:
            ax7 = fig.add_subplot(gs[0, 1])
            ax7.set_xlim([-1.0*pimage_in_mult, 27.0*pimage_in_mult])
            ax7.set_ylim([64.0*pimage_in_mult, -1.0*pimage_in_mult])
            ax7.set_facecolor('cyan')
            ax7.imshow(pimage_synth, interpolation='nearest', cmap=plt.cm.jet, origin='upper', vmin=0, vmax=35)
            ax7.set_title('Synth \n Pressure')
            ax7.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)

        if depth_synth_uncover is not None:
            ax8 = fig.add_subplot(gs[0, 2])
            ax8.set_xlim([-2.0, 54])
            ax8.set_ylim([128, -2.0])
            ax8.set_facecolor('cyan')
            ax8.imshow(depth_synth_uncover, interpolation='nearest', cmap=plt.cm.inferno, origin='upper', vmin=max_depth - 700, vmax=max_depth)
            ax8.set_title('Synth \n Depth \n Uncover')
            ax8.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)

        if color_real_uncover is not None:
            ax9 = fig.add_subplot(gs[1, 0])
            ax9.set_xlim([-2.0, 54])
            ax9.set_ylim([128, -2.0])
            ax9.set_facecolor('cyan')
            ax9.imshow(color_real_uncover)
            ax9.set_title('Real \n Color \n Uncover')
            ax9.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)

        if pimage_real_uncover is not None:
            ax5 = fig.add_subplot(gs[1, 1])
            ax5.set_xlim([-1.0*pimage_in_mult, 27.0*pimage_in_mult])
            ax5.set_ylim([64.0*pimage_in_mult, -1.0*pimage_in_mult])
            ax5.set_facecolor('cyan')
            ax5.imshow(pimage_real_uncover, interpolation='nearest', cmap=plt.cm.jet, origin='upper', vmin=0, vmax=35)
            ax5.set_title('Real \n Pressure \n Uncover')
            ax5.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)

        if depth_real_uncover is not None:
            ax6 = fig.add_subplot(gs[1, 2])
            ax6.set_xlim([-2.0, 54])
            ax6.set_ylim([128, -2.0])
            ax6.set_facecolor('cyan')
            ax6.imshow(depth_real_uncover, interpolation='nearest', cmap=plt.cm.inferno, origin='upper', vmin=max_depth - 700, vmax=max_depth)
            ax6.set_title('Real \n Depth \n Uncover')
            ax6.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)

        if color_synth_cover is not None:
            ax10 = fig.add_subplot(gs[0, 3])
            ax10.set_xlim([-2.0, 54])
            ax10.set_ylim([128, -2.0])
            ax10.set_facecolor('cyan')
            ax10.imshow(color_synth_cover)
            ax10.set_title('Synth \n Color \n Cover')
            ax10.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)

        if pimage_synth is not None:
            ax1 = fig.add_subplot(gs[0, 4])
            ax1.set_xlim([-1.0*pimage_in_mult, 27.0*pimage_in_mult])
            ax1.set_ylim([64.0*pimage_in_mult, -1.0*pimage_in_mult])
            ax1.set_facecolor('cyan')
            ax1.imshow(pimage_synth, interpolation='nearest', cmap=plt.cm.jet, origin='upper', vmin=0, vmax=35)
            ax1.set_title('Synth \n Pressure')
            ax1.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)

        if depth_synth is not None:
            ax2 = fig.add_subplot(gs[0, 5])
            ax2.set_xlim([-2.0, 54])
            ax2.set_ylim([128, -2.0])
            ax2.set_facecolor('cyan')
            ax2.imshow(depth_synth, interpolation='nearest', cmap=plt.cm.inferno, origin='upper', vmin=max_depth - 700, vmax=max_depth)
            ax2.set_title('Synth \n Depth \n Cover')
            ax2.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)

        if color_real_cover is not None:
            ax9 = fig.add_subplot(gs[1, 3])
            ax9.set_xlim([-2.0, 54])
            ax9.set_ylim([128, -2.0])
            ax9.set_facecolor('cyan')
            ax9.imshow(color_real_cover)
            ax9.set_title('Real \n Color \n Cover2')
            ax9.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)

        if pimage_real_cover is not None:
            ax3 = fig.add_subplot(gs[1, 4])
            ax3.set_xlim([-1.0*pimage_in_mult, 27.0*pimage_in_mult])
            ax3.set_ylim([64.0*pimage_in_mult, -1.0*pimage_in_mult])
            ax3.set_facecolor('cyan')
            ax3.imshow(pimage_real_cover, interpolation='nearest', cmap=plt.cm.jet, origin='upper', vmin=0, vmax=35)
            ax3.set_title('Real \n Pressure \n Cover2')
            ax3.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)

        if depth_real_cover is not None:
            ax4 = fig.add_subplot(gs[1, 5])
            ax4.set_xlim([-2.0, 54])
            ax4.set_ylim([128, -2.0])
            ax4.set_facecolor('cyan')
            ax4.imshow(depth_real_cover, interpolation='nearest', cmap=plt.cm.inferno, origin='upper', vmin=max_depth - 700, vmax=max_depth)
            ax4.set_title('Real \n Depth \n Cover2')
            ax4.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)


        #im = ax.imshow(depth_synth, interpolation='nearest', vmin=1000, vmax=1600, cmap=plt.cm.viridis, origin='upper')
        # fig.colorbar(im, orientation='vertical')

        # plt.imshow(color)


        if save_criteria[0] == True:
            print('saving fig')
            fig.savefig('/home/henry/Downloads/'+save_criteria[2]+'_'+str(save_criteria[1])+'.png', dpi=300)
        else:
            plt.show(block=True)


    def visualize_fcn_map(self, VIZ_DICT, VIZ_DICT_VAL, block = False, title = ' '):



        pimage_in_mult = 1.
        cntct_in_mult = 1.
        sobel_in_mult = 1.
        pmap_recon_in_mult = 1.
        cntct_recon_in_mult = 1.
        depth_in_mult = 1.

        if VIZ_DICT['pimage_in'].shape[0] == 128: pimage_in_mult = 2.
        if VIZ_DICT['cntct_in'] is not None:
            if VIZ_DICT['cntct_in'].shape[0] == 128: cntct_in_mult = 2.
        if VIZ_DICT['sobel_in'] is not None:
            if VIZ_DICT['sobel_in'].shape[0] == 128: sobel_in_mult = 2.
        if VIZ_DICT['pmap_recon_in'] is not None:
            if VIZ_DICT['pmap_recon_in'].shape[0] == 128: pmap_recon_in_mult = 2.
        if VIZ_DICT['cntct_recon_in'] is not None:
            if VIZ_DICT['cntct_recon_in'].shape[0] == 128: cntct_recon_in_mult = 2.
        if VIZ_DICT['depth_in'] is not None:
            if VIZ_DICT['depth_in'].shape[0] == 128: depth_in_mult = 2.

        plt.close()
        plt.pause(0.0001)

        # set options
        num_subplots = 3
        if VIZ_DICT['pmap_recon_in'] is not None:
            num_subplots += 3
        elif VIZ_DICT['pmap_recon_gt'] is not None:
            num_subplots += 2
        elif VIZ_DICT['depth_in'] is not None or VIZ_DICT['pmap_fcn_recon_gt'] is not None or VIZ_DICT['depth_fcn_recon_gt'] is not None:
            num_subplots += 1
        pmap_fcn_recon_col = 2
        depth_fcn_recon_col = 2
        pmap_fcn_recon_gt_col = 4
        depth_fcn_recon_gt_col = 5
        if VIZ_DICT['pmap_fcn_recon_gt'] is not None and VIZ_DICT['depth_fcn_recon_gt'] is not None:
            if num_subplots <= 8: num_subplots += 1
            depth_fcn_recon_col = 3
            pmap_fcn_recon_gt_col = 5
            depth_fcn_recon_gt_col = 6


        mult_bottom_row_size = 1.0
        unet_row = 1
        if VIZ_DICT['pmap_recon']  is not None or VIZ_DICT['pmap_recon_gt'] is not None:
            if VIZ_DICT['pmap_fcn_recon'] is not None or VIZ_DICT['depth_fcn_recon'] is not None:
                mult_bottom_row_size = 1.5
                unet_row = 2



        fig = plt.figure(tight_layout=True, figsize = (1.5*num_subplots*.8, 5*.8*mult_bottom_row_size))
        gs = gridspec.GridSpec(1+unet_row, num_subplots)


        plt.pause(0.0001)
        # use LaTeX fonts in the plot
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')



        if VIZ_DICT['depth_in'] is not None:
            ax18 = fig.add_subplot(gs[0, 0])
            ax18.set_xlim([-depth_in_mult, 27.0 * depth_in_mult])
            ax18.set_ylim([64.0 * depth_in_mult, -depth_in_mult])
            ax18.imshow(VIZ_DICT['depth_in'], interpolation='nearest', cmap=
            plt.cm.inferno, origin='upper', vmin=0, vmax=500)
            ax18.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom=False, top=False)
            ax18.set_ylabel('INPUT')
            ax18.set_title(r'$D$')

            ax18 = fig.add_subplot(gs[1, 0])
            ax18.set_xlim([-depth_in_mult, 27.0 * depth_in_mult])
            ax18.set_ylim([64.0 * depth_in_mult, -depth_in_mult])
            ax18.imshow(VIZ_DICT_VAL['depth_in'], interpolation='nearest', cmap=
            plt.cm.inferno, origin='upper', vmin=0, vmax=500)
            ax18.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom=False, top=False)
            ax18.set_ylabel('INPUT VAL')
            ax18.set_title(r'$D$')


        if VIZ_DICT['depth_fcn_recon'] is not None:
            ax24 = fig.add_subplot(gs[0, 2])
            ax24.set_xlim([-depth_in_mult, 27.0 * depth_in_mult])
            ax24.set_ylim([64.0 * depth_in_mult, -depth_in_mult])
            ax24.imshow(VIZ_DICT['depth_fcn_recon'], interpolation='nearest', cmap=
            plt.cm.inferno, origin='upper', vmin=0, vmax=500)
            ax24.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom=False, top=False)
            ax24.set_ylabel('FCN OUTPUT')
            ax24.set_title(r'$\widehat{D_{nb}}$')

            ax25 = fig.add_subplot(gs[1, 2])
            ax25.set_xlim([-depth_in_mult, 27.0 * depth_in_mult])
            ax25.set_ylim([64.0 * depth_in_mult, -depth_in_mult])
            ax25.imshow(VIZ_DICT_VAL['depth_fcn_recon'], interpolation='nearest', cmap=
            plt.cm.inferno, origin='upper', vmin=0, vmax=500)
            ax25.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom=False, top=False)
            ax25.set_ylabel('FCN OUTPUT VAL')
            ax25.set_title(r'$\widehat{D_{nb}}$')


            ax28 = fig.add_subplot(gs[0, 3])
            ax28.set_xlim([-depth_in_mult, 27.0 * depth_in_mult])
            ax28.set_ylim([64.0 * depth_in_mult, -depth_in_mult])
            ax28.imshow(VIZ_DICT['depth_fcn_recon_gt'], interpolation='nearest', cmap=
            plt.cm.inferno, origin='upper', vmin=0, vmax=500)
            ax28.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom=False, top=False)
            ax28.set_ylabel('GND TRUTH BLANKET')
            ax28.set_title(r'$D_{nb}$')


            ax29 = fig.add_subplot(gs[1, 3])
            ax29.set_xlim([-depth_in_mult, 27.0 * depth_in_mult])
            ax29.set_ylim([64.0 * depth_in_mult, -depth_in_mult])
            ax29.imshow(VIZ_DICT_VAL['depth_fcn_recon_gt'], interpolation='nearest', cmap=
            plt.cm.inferno, origin='upper', vmin=0, vmax=500)
            ax29.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom=False, top=False)
            ax29.set_ylabel('GND TRUTH BLANKET VAL')
            ax29.set_title(r'$D_{nb}$')



        plt.pause(0.0001)
        #fig.savefig('/home/henry/data/blah.png', dpi=400)
        plt.show(block=block)




    def plot_joint_markers(self, markers, p_map_mult, ax, color):
        if markers is not None:
            if len(np.shape(markers)) == 1:
                markers = np.reshape(markers, (int(len(markers) / 3), 3))

            target_coord = np.array(markers[:, :2])
            target_coord[:, 1] += (1.92 - 64.*0.0286)
            target_coord[:, 0] += (0.84 - 27.*0.0286)
            target_coord /= (INTER_SENSOR_DISTANCE * 1.92/(64.*0.0286))
            #target_coord[:, 0] -= 10
            target_coord[:, 1] += 10
            target_coord[:, 1] -= (NUMOFTAXELS_X - 1)# + 2)
            target_coord[:, 1] *= -1.0
            target_coord*=p_map_mult
            ax.plot(target_coord[:, 0], target_coord[:, 1], marker = 'o', linestyle='None', markerfacecolor = color, markeredgecolor='black', ms=4)

            if markers.size()[0] == 15:
                ax.plot(target_coord[14, 0], target_coord[14, 1], marker = 'o', linestyle='None', markerfacecolor = 'red', markeredgecolor='black', ms=4)
        plt.pause(0.0001)




    def plot_skeleton_links(self, markers, p_map_mult, ax, color):
        if len(np.shape(markers)) == 1:
            markers = np.reshape(markers, (int(len(markers) / 3), 3))


        target_coord = np.array(markers[:, :2])
        target_coord[:, 1] += (1.92 - 64.*0.0286)
        target_coord[:, 0] += (0.84 - 27.*0.0286)
        target_coord /= (INTER_SENSOR_DISTANCE * 1.92/(64.*0.0286))
        #print(markers)
        #target_coord[:, 0] -= 10
        target_coord[:, 1] += 10
        target_coord[:, 1] -= (NUMOFTAXELS_X - 1)# + 2)
        target_coord[:, 1] *= -1.0
        target_coord*=p_map_mult

        if np.shape(markers)[0] == 14 or np.shape(markers)[0] == 15:
            ax.plot([target_coord[0, 0], target_coord[1, 0]], [target_coord[0, 1],target_coord[1, 1] ], color = color, markeredgecolor='black', ms=8) #r ankle to knee
            ax.plot([target_coord[1, 0], target_coord[2, 0]], [target_coord[1, 1],target_coord[2, 1] ], color = color, markeredgecolor='black', ms=8) #r knee to hip
            ax.plot([target_coord[6, 0], target_coord[7, 0]], [target_coord[6, 1],target_coord[7, 1] ], color = color, markeredgecolor='black', ms=8) #r wrist to elbow
            ax.plot([target_coord[7, 0], target_coord[8, 0]], [target_coord[7, 1],target_coord[8, 1] ], color = color, markeredgecolor='black', ms=8) #r elbow to shoulder
            ax.plot([target_coord[8, 0], target_coord[12, 0]], [target_coord[8, 1],target_coord[12, 1] ], color = color, markeredgecolor='black', ms=8) #r shoulder to thorax

            ax.plot([target_coord[5, 0], target_coord[4, 0]], [target_coord[5, 1],target_coord[4, 1] ], color = '#88419d', markeredgecolor='black', ms=8) #l ankle to knee
            ax.plot([target_coord[4, 0], target_coord[3, 0]], [target_coord[4, 1],target_coord[3, 1] ], color = '#88419d', markeredgecolor='black', ms=8) #l knee to shoulder
            ax.plot([target_coord[11, 0], target_coord[10, 0]], [target_coord[11, 1],target_coord[10, 1] ], color = '#88419d', markeredgecolor='black', ms=8) #l wrist to elbow
            ax.plot([target_coord[10, 0], target_coord[9, 0]], [target_coord[10, 1],target_coord[9, 1] ], color = '#88419d', markeredgecolor='black', ms=8) #l elbow to shoulder
            ax.plot([target_coord[9, 0], target_coord[12, 0]], [target_coord[9, 1],target_coord[12, 1] ], color = '#88419d', markeredgecolor='black', ms=8) #l shoulder to thorax

            ax.plot([target_coord[12, 0], target_coord[13, 0]], [target_coord[12, 1],target_coord[13, 1] ], color = color, markeredgecolor='black', ms=8) #thorax to head top

        if np.shape(markers)[0] == 24 or np.shape(markers)[0] == 25:
            if color == 'green':
                color2 = '#88419d'
            else:
                color2 = '#e6550d'

            if np.linalg.norm(target_coord[0, :] - target_coord[3, :]) == 0:
                ax.plot([target_coord[1, 0], target_coord[4, 0]], [target_coord[1, 1],target_coord[4, 1] ], color = color, markeredgecolor='black', ms=8)
                ax.plot([target_coord[4, 0], target_coord[7, 0]], [target_coord[4, 1],target_coord[7, 1] ], color = color, markeredgecolor='black', ms=8)
                ax.plot([target_coord[2, 0], target_coord[5, 0]], [target_coord[2, 1],target_coord[5, 1] ], color = color2, markeredgecolor='black', ms=8)
                ax.plot([target_coord[5, 0], target_coord[8, 0]], [target_coord[5, 1],target_coord[8, 1] ], color = color2, markeredgecolor='black', ms=8)
                ax.plot([target_coord[12, 0], target_coord[15, 0]], [target_coord[12, 1],target_coord[15, 1] ], color = color, markeredgecolor='black', ms=8)

                ax.plot([target_coord[12, 0], target_coord[16, 0]], [target_coord[12, 1],target_coord[16, 1] ], color = color, markeredgecolor='black', ms=8)
                ax.plot([target_coord[16, 0], target_coord[18, 0]], [target_coord[16, 1],target_coord[18, 1] ], color = color, markeredgecolor='black', ms=8)
                ax.plot([target_coord[18, 0], target_coord[20, 0]], [target_coord[18, 1],target_coord[20, 1] ], color = color, markeredgecolor='black', ms=8)
                ax.plot([target_coord[12, 0], target_coord[17, 0]], [target_coord[12, 1],target_coord[17, 1] ], color = color2, markeredgecolor='black', ms=8)
                ax.plot([target_coord[17, 0], target_coord[19, 0]], [target_coord[17, 1],target_coord[19, 1] ], color = color2, markeredgecolor='black', ms=8)
                ax.plot([target_coord[19, 0], target_coord[21, 0]], [target_coord[19, 1],target_coord[21, 1] ], color = color2, markeredgecolor='black', ms=8)
            else:
                ax.plot([target_coord[0, 0], target_coord[1, 0]], [target_coord[0, 1],target_coord[1, 1] ], color = color2, markeredgecolor='black', ms=8)
                ax.plot([target_coord[1, 0], target_coord[4, 0]], [target_coord[1, 1],target_coord[4, 1] ], color = color2, markeredgecolor='black', ms=8)
                ax.plot([target_coord[4, 0], target_coord[7, 0]], [target_coord[4, 1],target_coord[7, 1] ], color = color2, markeredgecolor='black', ms=8)
                ax.plot([target_coord[7, 0], target_coord[10, 0]], [target_coord[7, 1],target_coord[10, 1] ], color = color2, markeredgecolor='black', ms=8)
                ax.plot([target_coord[0, 0], target_coord[2, 0]], [target_coord[0, 1],target_coord[2, 1] ], color = color, markeredgecolor='black', ms=8)
                ax.plot([target_coord[2, 0], target_coord[5, 0]], [target_coord[2, 1],target_coord[5, 1] ], color = color, markeredgecolor='black', ms=8)
                ax.plot([target_coord[5, 0], target_coord[8, 0]], [target_coord[5, 1],target_coord[8, 1] ], color = color, markeredgecolor='black', ms=8)
                ax.plot([target_coord[8, 0], target_coord[11, 0]], [target_coord[8, 1],target_coord[11, 1] ], color = color, markeredgecolor='black', ms=8)
                ax.plot([target_coord[0, 0], target_coord[3, 0]], [target_coord[0, 1],target_coord[3, 1] ], color = color, markeredgecolor='black', ms=8)
                ax.plot([target_coord[3, 0], target_coord[6, 0]], [target_coord[3, 1],target_coord[6, 1] ], color = color, markeredgecolor='black', ms=8)
                ax.plot([target_coord[6, 0], target_coord[9, 0]], [target_coord[6, 1],target_coord[9, 1] ], color = color, markeredgecolor='black', ms=8)
                ax.plot([target_coord[9, 0], target_coord[12, 0]], [target_coord[9, 1],target_coord[12, 1] ], color = color, markeredgecolor='black', ms=8)
                ax.plot([target_coord[12, 0], target_coord[15, 0]], [target_coord[12, 1],target_coord[15, 1] ], color = color, markeredgecolor='black', ms=8)

                ax.plot([target_coord[9, 0], target_coord[13, 0]], [target_coord[9, 1],target_coord[13, 1] ], color = color2, markeredgecolor='black', ms=8)
                ax.plot([target_coord[13, 0], target_coord[16, 0]], [target_coord[13, 1],target_coord[16, 1] ], color = color2, markeredgecolor='black', ms=8)
                ax.plot([target_coord[16, 0], target_coord[18, 0]], [target_coord[16, 1],target_coord[18, 1] ], color = color2, markeredgecolor='black', ms=8)
                ax.plot([target_coord[18, 0], target_coord[20, 0]], [target_coord[18, 1],target_coord[20, 1] ], color = color2, markeredgecolor='black', ms=8)
                ax.plot([target_coord[20, 0], target_coord[22, 0]], [target_coord[20, 1],target_coord[22, 1] ], color = color2, markeredgecolor='black', ms=8)

                ax.plot([target_coord[9, 0], target_coord[14, 0]], [target_coord[9, 1],target_coord[14, 1] ], color = color, markeredgecolor='black', ms=8)
                ax.plot([target_coord[14, 0], target_coord[17, 0]], [target_coord[14, 1],target_coord[17, 1] ], color = color, markeredgecolor='black', ms=8)
                ax.plot([target_coord[17, 0], target_coord[19, 0]], [target_coord[17, 1],target_coord[19, 1] ], color = color, markeredgecolor='black', ms=8)
                ax.plot([target_coord[19, 0], target_coord[21, 0]], [target_coord[19, 1],target_coord[21, 1] ], color = color, markeredgecolor='black', ms=8)
                ax.plot([target_coord[21, 0], target_coord[23, 0]], [target_coord[21, 1],target_coord[23, 1] ], color = color, markeredgecolor='black', ms=8)


        plt.pause(0.0001)

