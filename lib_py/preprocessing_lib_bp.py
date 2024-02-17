#!/usr/bin/env python

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import *

import random
from scipy import ndimage
import scipy.stats as ss
from scipy.ndimage.interpolation import zoom

from scipy.ndimage.filters import gaussian_filter


# PyTorch libraries
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import transforms
from torch.autograd import Variable
import cv2


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



class PreprocessingLib():
    def __init__(self):

        self.camera_to_bed_dist = 1.645 - 0.2032
        #zero_location += 0.5
        #zero_location = zero_location.astype(int)

        self.x = np.arange(0, 54).astype(float)
        self.x = np.tile(self.x, (128, 1))
        self.y = np.arange(0, 128).astype(float)
        self.y = np.tile(self.y, (54, 1)).T

        self.x_coord_from_camcenter = self.x - 26.5  # self.depthcam_midpixel[0]
        self.y_coord_from_camcenter = self.y - 63.5  # self.depthcam_midpixel[1]


        depth = np.zeros((128, 54))
        depth2 = np.zeros((128, 54))

        for i in range(128):
            depth[i, :] = -i*(i-127)*0.01/40.32

        for i in range(53):
            depth2[:, i] = -i*(i-53)*0.01/7.02

        self.full_mattress_comp = depth*depth2*101.6



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

    def preprocessing_add_image_noise(self, images, pmat_chan_idx, norm_std_coeffs):

        queue = np.copy(images[:, pmat_chan_idx:pmat_chan_idx+2, :, :])
        queue[queue != 0] = 1.


        x = np.arange(-10, 10)
        xU, xL = x + 0.5, x - 0.5
        prob = ss.norm.cdf(xU, scale=1) - ss.norm.cdf(xL,scale=1)  # scale is the standard deviation using a cumulative density function
        prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
        image_noise = np.random.choice(x, size=(images.shape[0], 2, images.shape[2], images.shape[3]), p=prob)

        
        image_noise = image_noise*queue
        image_noise = image_noise.astype(float)
        #image_noise[:, 0, :, :] /= 11.70153502792190
        #image_noise[:, 1, :, :] /= 45.61635847182483
        image_noise[:, 0, :, :] *= norm_std_coeffs[3]
        image_noise[:, 1, :, :] *= norm_std_coeffs[4]

        images[:, pmat_chan_idx:pmat_chan_idx+2, :, :] += image_noise

        #print images[0, 0, 50, 10:25], 'added noise'

        #clip noise so we dont go outside sensor limits
        #images[:, pmat_chan_idx, :, :] = np.clip(images[:, pmat_chan_idx, :, :], 0, 100/11.70153502792190)
        #images[:, pmat_chan_idx, :, :] = np.clip(images[:, pmat_chan_idx, :, :], 0, 100.*norm_std_coeffs[4])
        images[:, pmat_chan_idx, :, :] = np.clip(images[:, pmat_chan_idx, :, :], 0, 10000.)
        #images[:, pmat_chan_idx+1, :, :] = np.clip(images[:, pmat_chan_idx+1, :, :], 0, 10000)
        return images


    def preprocessing_add_calibration_noise(self, images, pmat_chan_idx, norm_std_coeffs, is_training, noise_amount, normalize_per_image):
        time_orig = time.time()
        if is_training == True:
            variation_amount = float(noise_amount)
            print ("ADDING CALIB NOISE", variation_amount)

            #pmat_contact_orig = np.copy(images[:, pmat_chan_idx, :, :])
            #pmat_contact_orig[pmat_contact_orig != 0] = 1.
            #sobel_contact_orig = np.copy(images[:, pmat_chan_idx+1, :, :])
            #sobel_contact_orig[sobel_contact_orig != 0] = 1.

            for map_index in range(images.shape[0]):

                pmat_contact_orig = np.copy(images[map_index, pmat_chan_idx, :, :])
                pmat_contact_orig[pmat_contact_orig != 0] = 1.
                sobel_contact_orig = np.copy(images[map_index, pmat_chan_idx + 1, :, :])
                sobel_contact_orig[sobel_contact_orig != 0] = 1.


                # first multiply
                amount_to_mult_im = random.normalvariate(mu = 1.0, sigma = variation_amount) #mult a variation of 10%
                amount_to_mult_sobel = random.normalvariate(mu = 1.0, sigma = variation_amount) #mult a variation of 10%
                images[map_index, pmat_chan_idx, :, :] = images[map_index, pmat_chan_idx, :, :] * amount_to_mult_im
                images[map_index, pmat_chan_idx+1, :, :] = images[map_index, pmat_chan_idx+1, :, :] * amount_to_mult_sobel

                # then add
                #amount_to_add_im = random.normalvariate(mu = 0.0, sigma = (1./11.70153502792190)*(98.666 - 0.0)*0.1) #add a variation of 10% of the range
                #amount_to_add_sobel = random.normalvariate(mu = 0.0, sigma = (1./45.61635847182483)*(386.509 - 0.0)*0.1) #add a variation of 10% of the range

                if normalize_per_image == True:
                    amount_to_add_im = random.normalvariate(mu = 0.0, sigma = norm_std_coeffs[3]*(70. - 0.0)*variation_amount) #add a variation of 10% of the range
                    amount_to_add_sobel = random.normalvariate(mu = 0.0, sigma = norm_std_coeffs[4]*(70. - 0.0)*variation_amount) #add a variation of 10% of the range
                else:
                    amount_to_add_im = random.normalvariate(mu = 0.0, sigma = norm_std_coeffs[3]*(98.666 - 0.0)*variation_amount) #add a variation of 10% of the range
                    amount_to_add_sobel = random.normalvariate(mu = 0.0, sigma = norm_std_coeffs[4]*(386.509 - 0.0)*variation_amount) #add a variation of 10% of the range

                images[map_index, pmat_chan_idx, :, :] = images[map_index, pmat_chan_idx, :, :] + amount_to_add_im
                images[map_index, pmat_chan_idx+1, :, :] = images[map_index, pmat_chan_idx+1, :, :] + amount_to_add_sobel
                images[map_index, pmat_chan_idx, :, :] = np.clip(images[map_index, pmat_chan_idx, :, :], a_min = 0., a_max = 10000)
                images[map_index, pmat_chan_idx+1, :, :] = np.clip(images[map_index, pmat_chan_idx+1, :, :], a_min = 0., a_max = 10000)

                #cut out the background. need to do this after adding.
                images[map_index, pmat_chan_idx, :, :] *= pmat_contact_orig#[map_index, :, :]
                images[map_index, pmat_chan_idx+1, :, :] *= sobel_contact_orig#[map_index, :, :]


                amount_to_gauss_filter_im = random.normalvariate(mu = 0.5, sigma = variation_amount)
                amount_to_gauss_filter_sobel = random.normalvariate(mu = 0.5, sigma = variation_amount)
                images[map_index, pmat_chan_idx, :, :] = gaussian_filter(images[map_index, pmat_chan_idx, :, :], sigma= amount_to_gauss_filter_im) #pmap
                images[map_index, pmat_chan_idx+1, :, :] = gaussian_filter(images[map_index, pmat_chan_idx+1, :, :], sigma= amount_to_gauss_filter_sobel) #sobel #NOW


        else:  #if its NOT training we should still blur things by 0.5
            print (pmat_chan_idx, np.shape(images), 'pmat chan idx')

            for map_index in range(images.shape[0]):
               # print pmat_chan_idx, images.shape, 'SHAPE'
                images[map_index, pmat_chan_idx, :, :] = gaussian_filter(images[map_index, pmat_chan_idx, :, :], sigma= 0.5) #pmap
                images[map_index, pmat_chan_idx+1, :, :] = gaussian_filter(images[map_index, pmat_chan_idx+1, :, :], sigma= 0.5) #sobel

        #images[:, pmat_chan_idx, :, :] = np.clip(images[:, pmat_chan_idx, :, :], 0, 100/11.70153502792190)
            if normalize_per_image == False:
                images[:, pmat_chan_idx, :, :] = np.clip(images[:, pmat_chan_idx, :, :], 0, 100.*norm_std_coeffs[3])
            else:
                images[:, pmat_chan_idx, :, :] = np.clip(images[:, pmat_chan_idx, :, :], 0, 10000.)



        #now calculate the contact map AFTER we've blurred it
        pmat_contact = np.copy(images[:, pmat_chan_idx:pmat_chan_idx+1, :, :])
        #pmat_contact[pmat_contact != 0] = 100./41.80684362163343
        pmat_contact[pmat_contact != 0] = 100.*norm_std_coeffs[0]
        images = np.concatenate((pmat_contact, images), axis = 1)

        #for i in range(0, 20):
        #    VisualizationLib().visualize_pressure_map(images[i, 0, :, :] * 20., None, None,
        #                                              images[i, 1, :, :] * 20., None, None,
        #                                              block=False)
        #    time.sleep(0.5)


        #print time.time() - time_orig

        return images




    def preprocessing_add_slp_noise(self, images, batch_depth, batch1, pmat_chan_idx, norm_std_coeffs, is_training, normalize_per_image):
        time_orig = time.time()

        batch1  = batch1.cpu().numpy().reshape(-1, 24, 3)


        if batch_depth is not None:
            depth_ims = np.array(batch_depth.numpy())
            map_idx_len = depth_ims.shape[0]
        else:
            map_idx_len = images.shape[0]

        if is_training == True:
            print ("ADDING SLP NOISE")


            for map_index in range(map_idx_len):

                #DO SLP STUFF HERE: flip, scale, rotate, occlude
                input_shape = (images.shape[2], images.shape[3])
                scale, rot, do_flip, _, do_occlusion = self.get_aug_config()
                #scale, rot, do_flip, _, do_occlusion = 1.0, 0.0, False, [1.0, 1.0, 1.0], False



                bb_c_x = input_shape[1]/2 + 0.5
                bb_c_y = input_shape[0]/2
                bb_width = input_shape[1]
                bb_height = input_shape[0]

                # synthetic occlusion
                if do_occlusion:
                    while True:
                        area_min = 0.0
                        area_max = 0.7
                        synth_area = (random.random() * (area_max - area_min) + area_min) * bb_width * bb_height

                        ratio_min = 0.3
                        ratio_max = 1 / 0.3
                        synth_ratio = (random.random() * (ratio_max - ratio_min) + ratio_min)

                        synth_h = math.sqrt(synth_area * synth_ratio)
                        synth_w = math.sqrt(synth_area / synth_ratio)
                        synth_xmin = random.random() * (bb_width - synth_w - 1) + 0#bbox[0]
                        synth_ymin = random.random() * (bb_height - synth_h - 1) + 0#bbox[1]

                        if synth_xmin >= 0 and synth_ymin >= 0 and synth_xmin + synth_w < bb_width and synth_ymin + synth_h < bb_height:
                            xmin = int(synth_xmin)
                            ymin = int(synth_ymin)
                            w = int(synth_w)
                            h = int(synth_h)
                            images[map_index, pmat_chan_idx, ymin:ymin + h, xmin:xmin + w] = np.random.rand(h, w) * np.max(images[map_index, pmat_chan_idx, :, :])
                            try:
                                images[map_index, pmat_chan_idx+1, ymin:ymin + h, xmin:xmin + w] = np.random.rand(h, w) * np.max(images[map_index, pmat_chan_idx+1, :, :])
                            except:
                                pass
                            depth_ims[map_index, 0, ymin*2:ymin*2 + h*2, xmin*2:xmin*2 + w*2] = np.random.rand(h*2, w*2) * np.max(depth_ims[map_index, 0, :, :])
                            break



                if do_flip:
                    images[map_index, pmat_chan_idx, :, :] = images[map_index, pmat_chan_idx, :, ::-1]
                    try:
                        images[map_index, pmat_chan_idx+1, :, :] = images[map_index, pmat_chan_idx+1, :, ::-1]
                    except:
                        pass
                    depth_ims[map_index, 0, :, :] = depth_ims[map_index, 0, :, ::-1]
                    bb_c_x = bb_width - bb_c_x





                trans = self.gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, input_shape[1], input_shape[0],
                                                scale, rot, inv=False)  # is bb aspect needed? yes, otherwise patch distorted
                trans_labels = self.gen_trans_from_patch_cv((2*286.+27*28.6)/2, (2*286.+64*28.6)/2, 2*286.+27*28.6, 2*286.+64*28.6, 2*286.+27*28.6, 2*286.+64*28.6,
                                                scale, -rot, inv=False)  # is bb aspect needed? yes, otherwise patch distorted
                trans_depth = self.gen_trans_from_patch_cv(bb_c_x*2, bb_c_y*2, bb_width*2, bb_height*2, input_shape[1]*2, input_shape[0]*2,
                                                scale, rot, inv=False)  # is bb aspect needed? yes, otherwise patch distorted
                #print("trans", trans)

                images[map_index, pmat_chan_idx, :, :] = cv2.warpAffine(images[map_index, pmat_chan_idx, :, :],
                                                        trans, (int(input_shape[1]), int(input_shape[0])),
                                                        flags=cv2.INTER_LINEAR)  # is there channel requirements
                try:
                    images[map_index, pmat_chan_idx+1, :, :] = cv2.warpAffine(images[map_index, pmat_chan_idx+1, :, :],
                                                            trans, (int(input_shape[1]), int(input_shape[0])),
                                                            flags=cv2.INTER_LINEAR)  # is there channel requirements
                except:
                    pass
                depth_ims[map_index, 0, :, :] = cv2.warpAffine(depth_ims[map_index, 0, :, :],
                                                        trans_depth, (int(input_shape[1]*2), int(input_shape[0]*2)),
                                                        flags=cv2.INTER_LINEAR)  # is there channel requirements

                if do_flip:
                    batch1[map_index, :, 0] = - batch1[map_index, :, 0] + 27*28.6 + 286*2# + 33*28.6 -

                for i in range(24):  # jt trans
                    # print(i, joints_pch[i, 0:2])
                    batch1[map_index, i, 0:2] = self.trans_point2d(batch1[map_index, i, 0:2], trans_labels)  # this is what causes things to be weird


                images[map_index, pmat_chan_idx, :, :] = gaussian_filter(images[map_index, pmat_chan_idx, :, :], sigma= 0.5) #pmap
                try:
                    images[map_index, pmat_chan_idx+1, :, :] = gaussian_filter(images[map_index, pmat_chan_idx+1, :, :], sigma= 0.5) #sobel
                except:
                    pass


        else:  #if its NOT training we should still blur things by 0.5

            for map_index in range(images.shape[0]):
               # print pmat_chan_idx, images.shape, 'SHAPE'
                images[map_index, pmat_chan_idx, :, :] = gaussian_filter(images[map_index, pmat_chan_idx, :, :], sigma= 0.5) #pmap
                try:
                    images[map_index, pmat_chan_idx+1, :, :] = gaussian_filter(images[map_index, pmat_chan_idx+1, :, :], sigma= 0.5) #sobel
                except:
                    pass

        #images[:, pmat_chan_idx, :, :] = np.clip(images[:, pmat_chan_idx, :, :], 0, 100/11.70153502792190)
            if normalize_per_image == False:
                images[:, pmat_chan_idx, :, :] = np.clip(images[:, pmat_chan_idx, :, :], 0, 100.*norm_std_coeffs[3])
            else:
                images[:, pmat_chan_idx, :, :] = np.clip(images[:, pmat_chan_idx, :, :], 0, 10000.)



        #now calculate the contact map AFTER we've blurred it
        pmat_contact = np.copy(images[:, pmat_chan_idx:pmat_chan_idx+1, :, :])
        #pmat_contact[pmat_contact != 0] = 100./41.80684362163343
        pmat_contact[pmat_contact != 0] = 100.*norm_std_coeffs[0]
        images = np.concatenate((pmat_contact, images), axis = 1)


        if batch_depth is not None:
            batch_depth_return = torch.Tensor(depth_ims)
        else:
            batch_depth_return = None


        batch1 = torch.Tensor(batch1.reshape(-1, 72))

        return images, batch_depth_return, batch1

    def get_aug_config(self):
        scale_factor = 0.25
        rot_factor = 30
        color_factor = 0.2

        scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
        rot = np.clip(np.random.randn(), -2.0,
                      2.0) * rot_factor if random.random() <= 0.6 else 0  # -60 to 60
        do_flip = random.random() <= 0.5
        c_up = 1.0 + color_factor
        c_low = 1.0 - color_factor
        color_scale = [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)]

        do_occlusion = random.random() <= 0.5

        return scale, rot, do_flip, color_scale, do_occlusion


    def gen_trans_from_patch_cv(self, c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
        # augment size with scale
        src_w = src_width * scale
        src_h = src_height * scale
        src_center = np.array([c_x, c_y], dtype=np.float32)

        # augment rotation
        rot_rad = np.pi * rot / 180
        src_downdir = self.rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
        src_rightdir = self.rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

        dst_w = dst_width
        dst_h = dst_height
        dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
        dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
        dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = src_center
        src[1, :] = src_center + src_downdir
        src[2, :] = src_center + src_rightdir

        dst = np.zeros((3, 2), dtype=np.float32)
        dst[0, :] = dst_center
        dst[1, :] = dst_center + dst_downdir
        dst[2, :] = dst_center + dst_rightdir

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return trans

    def trans_point2d(self, pt_2d, trans):
        src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
        dst_pt = np.dot(trans, src_pt)
        return dst_pt[0:2]


    def rotate_2d(self, pt_2d, rot_rad):
        x = pt_2d[0]
        y = pt_2d[1]
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        xx = x * cs - y * sn
        yy = x * sn + y * cs
        return np.array([xx, yy], dtype=np.float32)

    def multiply_along_axis(self, A, B, axis):
        A = np.array(A)
        B = np.array(B)
        # shape check
        if axis >= A.ndim:
            raise AxisError(axis, A.ndim)
        if A.shape[axis] != B.size:
            raise ValueError("'A' and 'B' must have the same length along the given axis")
        # Expand the 'B' according to 'axis':
        # 1. Swap the given axis with axis=0 (just need the swapped 'shape' tuple here)
        swapped_shape = A.swapaxes(0, axis).shape
        # 2. Repeat:
        # loop through the number of A's dimensions, at each step:
        # a) repeat 'B':
        #    The number of repetition = the length of 'A' along the
        #    current looping step;
        #    The axis along which the values are repeated. This is always axis=0,
        #    because 'B' initially has just 1 dimension
        # b) reshape 'B':
        #    'B' is then reshaped as the shape of 'A'. But this 'shape' only
        #     contains the dimensions that have been counted by the loop
        for dim_step in range(A.ndim - 1):
            B = B.repeat(swapped_shape[dim_step + 1], axis=0) \
                .reshape(swapped_shape[:dim_step + 2])
        # 3. Swap the axis back to ensure the returned 'B' has exactly the
        # same shape of 'A'
        B = B.swapaxes(0, axis)
        return A * B


    def clean_depth_images(self, batch):

        # mode_val = stats.mode(batch[-1].squeeze().cpu().numpy())
        orig_image = batch[-1].clone().squeeze().cpu().numpy()
        overall_counts_all = batch[-1].clone().squeeze().cpu().numpy()[0:20, :].flatten().astype(np.int64)

        overall_counts = np.bincount(overall_counts_all)
        overall_mode_val = np.argmax(overall_counts)

        for horizidx in range(54):
            overwrite_ct = 1
            overwrite_ctB = 1
            for batchidx in range(batch[-1].size()[0]):
                counts_all = orig_image[0:20, np.max([0, horizidx - 10]):np.min([53, horizidx + 10])].flatten().astype(
                    np.int64)
                counts_all = counts_all[counts_all < overall_mode_val + 50]
                counts_all = counts_all[counts_all > overall_mode_val - 50]
                counts = np.bincount(counts_all)
                # try:
                try:
                    mode_val = np.argmax(counts)

                    for i in range(15):
                        if torch.min(batch[-1][batchidx, 0, i, horizidx]) < 1800 or torch.max(
                                batch[-1][batchidx, 0, i, horizidx]) > 2200:
                            overwrite_ct += 1
                    for i in range(overwrite_ct):
                        batch[-1][batchidx, 0, i, horizidx] = batch[-1][
                                                                  batchidx, 0, overwrite_ct, horizidx].clone() * 0.0 + float(
                            mode_val)
                    # except:
                    #    pass

                    counts_all = orig_image[-21:-1,
                                 np.max([0, horizidx - 10]):np.min([53, horizidx + 10])].flatten().astype(np.int64)
                    counts_all = counts_all[counts_all < overall_mode_val + 40]
                    counts_all = counts_all[counts_all > overall_mode_val - 40]
                    counts = np.bincount(counts_all)
                    try:
                        mode_val = np.argmax(counts)

                        for i in range(15):
                            if torch.min(batch[-1][batchidx, 0, -i, horizidx]) < 1700 or torch.max(
                                    batch[-1][batchidx, 0, -i, horizidx]) > 2300:
                                overwrite_ctB += 1
                            else:
                                pass  # break
                        for i in range(overwrite_ctB):
                            batch[-1][batchidx, 0, -i, horizidx] = batch[-1][
                                                                       batchidx, 0, overwrite_ctB, horizidx].clone() * 0.0 + float(
                                mode_val)
                    except:
                        pass
                except:
                    pass

                # print(overwrite_ct, "OVERWRITING THIS MANY LINES AT TOP")
        pre_hair_np = batch[-1].clone().squeeze().cpu().numpy()
        print('checking for hair')
        for batchidx in range(batch[-1].size()[0]):
            for horizidx in range(2, 52):
                for vertidx in range(2, 40):
                    if pre_hair_np[vertidx, horizidx] < 1800:
                        box_surrounding = pre_hair_np[vertidx - 2:vertidx + 2, horizidx - 2:horizidx + 2].flatten()
                        # print(box_surrounding)
                        box_surrounding = box_surrounding[box_surrounding >= 1800]
                        # print(box_surrounding)
                        # print(vertidx, horizidx, 'probably hair', batch[-1][batchidx, 0, vertidx, horizidx])

                        try:
                            batch[-1][batchidx, 0, vertidx, horizidx] = batch[-1][
                                                                            batchidx, 0, vertidx, horizidx].clone() * 0.0 + np.min(
                                box_surrounding)
                        except:
                            pass

        post_hair_np = batch[-1].clone().squeeze().unsqueeze(0).cpu().numpy()
        # for batchidx in range(batch[-1].size()[0]):
        #    print('smoothing')
        #    batch[-1][batchidx, 0, :, :] = torch.Tensor(gaussian_filter(post_hair_np[batchidx, :, :], sigma = 0.5)).unsqueeze(0)

        return batch

    def preprocessing_add_depth_calnoise(self, depth_tensor, targets, root_xyz):

        depth_images = depth_tensor.data.numpy().astype(np.int16)
        variation_amount = float(0.2)

        # print ("ADDING DEPTH NOISE")



        root_pos_in_taxels = targets[:, 0:2].clone().cpu().numpy()
        root_pos_in_taxels -= 1000*0.286 #0 point is 10 taxels below the bottom left corner in both x and y
        root_pos_in_taxels /= 28.6 #convert from mm to taxels
        root_pos_in_taxels *= 2 #double resolution for depth
        root_pos_in_taxels = root_pos_in_taxels.astype(np.int16)
        root_pos_in_taxels[:, 0] = np.clip(root_pos_in_taxels[:, 0], a_min = 0, a_max=53)
        root_pos_in_taxels[:, 1] = np.clip(root_pos_in_taxels[:, 1], a_min = 0, a_max=127)

        targets = targets.view(-1, 24, 3)

        #print(np.shape(self.full_mattress_comp), np.max(self.full_mattress_comp))
        #print(np.shape(depth_images))

        #print(root_xyz[0, 2], 'init gt root height')
        bed_vertical_shift_list = []
        same_as_slp_noise = False

        for i in range(depth_images.shape[0]):
            #print root_pos_in_taxels[i, :]

            #print(i, targets[i, :], root_xyz[i, :], np.max(self.full_mattress_comp))

            if same_as_slp_noise == True:
                sampled_mattress_comp = 0*self.full_mattress_comp  * np.random.uniform(0, 1)
            else:
                sampled_mattress_comp = self.full_mattress_comp  * np.random.uniform(0, 1)
            additive_factor = np.random.uniform(-1, 1) * 101.6 / 2
            sampled_mattress_comp += additive_factor

            root_drop = -sampled_mattress_comp[root_pos_in_taxels[i, 1], root_pos_in_taxels[i, 0]]

            bed_vertical_shift_list.append(root_drop)


            targets[i, :, 2] += root_drop
            root_xyz[i, 2] += root_drop/1000. #this is to adjust the ground truth mesh and markers. they are global.

            #if i == 0: print(targets[i, 0, 2], root_xyz[i, 2], 'after')

            #print(i, targets[i, :], root_xyz[i, :], np.max(sampled_mattress_comp))
            #print self.full_mattress_comp.shape, root_pos_in_taxels[i, :], root_drop


            depth_images[i, 0, :, :] += sampled_mattress_comp.astype(np.int16)





            #p = np.abs(np.random.normal(0, 0.05))
            p = np.abs(np.random.normal(0, 0.05))#use this for blankets
            image_dropout = np.random.binomial(1, 1-p, 6912).reshape(depth_images.shape[2], depth_images.shape[3])
            depth_images[i, 0, :, :]*= image_dropout



            x = np.arange(-20, 21)
            xU, xL = x + 0.5, x - 0.5
            #prob = ss.norm.cdf(xU, scale=3) - ss.norm.cdf(xL, scale=3)  # scale is the standard deviation using a cumulative density function
            prob = ss.norm.cdf(xU, scale=1) - ss.norm.cdf(xL, scale=1)  # scale is the standard deviation using a cumulative density function #use this for blankets
            prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
            image_noise = np.random.choice(x, size=(depth_images.shape[2], depth_images.shape[3]), p=prob)*10* np.random.uniform(0, 1)
            #image_noise = np.random.choice(x, size=(depth_images.shape[2], depth_images.shape[3]), p=prob)*5* np.random.uniform(0, 1) #use this for blankets
            #image_noise = np.random.normal(0, 10, 6912).reshape(depth_images.shape[2], depth_images.shape[3])
            depth_images[i, 0, :, :] += image_noise.astype(np.int16)



            if same_as_slp_noise == True:
                prob_occlude = 1.0
            else:
                prob_occlude = np.random.uniform(0, 1)

            if prob_occlude > 0.5:
                bb_width = 27
                bb_height = 64
                while True:
                    area_min = 0.0
                    area_max = 0.7
                    synth_area = (random.random() * (area_max - area_min) + area_min) * bb_width * bb_height

                    ratio_min = 0.3
                    ratio_max = 1 / 0.3
                    synth_ratio = (random.random() * (ratio_max - ratio_min) + ratio_min)

                    synth_h = math.sqrt(synth_area * synth_ratio)
                    synth_w = math.sqrt(synth_area / synth_ratio)
                    synth_xmin = random.random() * (bb_width - synth_w - 1) + 0  # bbox[0]
                    synth_ymin = random.random() * (bb_height - synth_h - 1) + 0  # bbox[1]

                    if synth_xmin >= 0 and synth_ymin >= 0 and synth_xmin + synth_w < bb_width and synth_ymin + synth_h < bb_height:
                        xmin = int(synth_xmin)
                        ymin = int(synth_ymin)
                        w = int(synth_w)
                        h = int(synth_h)
                        #print(np.shape(np.random.rand(h * 2, w * 2)), np.shape(depth_images[i, 0, ymin * 2:ymin * 2 + h * 2, xmin * 2:xmin * 2 + w * 2]))

                        depth_images[i, 0, ymin * 2:ymin * 2 + h * 2, xmin * 2:xmin * 2 + w * 2] = np.random.rand(h * 2, w * 2) * np.max(depth_images[i, 0, :, :])
                        break



            #if i == 1:
            #    break

        #print(root_xyz[0, 2], 'noisey gt root height')


        targets = targets.view(-1, 72)


        #print root_pos_in_taxels
        #print targets.size(), root_xyz.size()




        #batch_fake_floor = np.random.uniform(low = 1219.5, high = 2438.5, size = depth_images.shape[0]).astype(np.int16) #vary floor between 4 and 8 feet from camera
        #add_background = self.multiply_along_axis(mask_background, batch_fake_floor,  axis = 0)

        #depth_images += add_background


        depth_images = np.clip(depth_images, 0, 10000.)


        depth_tensor = torch.Tensor(depth_images)


        return depth_tensor, targets, root_xyz, torch.Tensor(bed_vertical_shift_list).unsqueeze(1)







    def preprocessing_blur_images(self, x_data, mat_size, sigma):

        x_data_return = []
        for map_index in range(len(x_data)):
            p_map = np.reshape(x_data[map_index], mat_size)

            p_map = gaussian_filter(p_map, sigma= sigma)

            x_data_return.append(p_map.flatten())

        return x_data_return




    def preprocessing_create_pressure_angle_stack(self,x_data, mat_size, CTRL_PNL):
        '''This is for creating a 2-channel input using the height of the bed. '''

        #if CTRL_PNL['verbose']:
        print (np.shape(x_data), np.max(x_data), 'max p im')
        x_data = np.clip(x_data, 0, 100)

        print( "normalizing per image", CTRL_PNL['normalize_per_image'])

        p_map_dataset = []
        for map_index in range(len(x_data)):
            # print map_index, self.mat_size, 'mapidx'
            # Resize mat to make into a matrix

            p_map = np.reshape(x_data[map_index], mat_size)

            if CTRL_PNL['normalize_per_image'] == True:
                p_map = p_map * (20000./np.sum(p_map))

            if mat_size == (84, 47):
                p_map = p_map[10:74, 10:37]

            # this makes a sobel edge on the image
            sx = ndimage.sobel(p_map, axis=0, mode='constant')
            sy = ndimage.sobel(p_map, axis=1, mode='constant')
            p_map_inter = np.hypot(sx, sy)
            if CTRL_PNL['clip_sobel'] == True:
                p_map_inter = np.clip(p_map_inter, a_min=0, a_max = 100)

            if CTRL_PNL['normalize_per_image'] == True:
                p_map_inter = p_map_inter * (20000. / np.sum(p_map_inter))

            #print np.sum(p_map), 'sum after norm'
            p_map_dataset.append([p_map, p_map_inter])

        return p_map_dataset


    def preprocessing_create_pressure_only_stack(self,x_data, mat_size, CTRL_PNL):
        '''This is for creating a 1-channel input using the height of the bed. '''

        print (np.shape(x_data), np.max(x_data), 'max p im')
        x_data = np.clip(x_data, 0, 100)


        p_map_dataset = []
        for map_index in range(np.shape(x_data)[0]):
            # print map_index, self.mat_size, 'mapidx'
            # Resize mat to make into a matrix

            p_map = np.reshape(x_data[map_index], mat_size)


            p_map_dataset.append([p_map])

        print(np.shape(p_map_dataset), 'shape test xa pressure only stack')

        return p_map_dataset


    def preprocessing_pressure_map_upsample(self, data, multiple, order=1):
        '''Will upsample an incoming pressure map dataset'''
        p_map_highres_dataset = []


        if len(np.shape(data)) == 3:
            for map_index in range(len(data)):
                #Upsample the current map using bilinear interpolation
                p_map_highres_dataset.append(
                        ndimage.zoom(data[map_index], multiple, order=order))
        elif len(np.shape(data)) == 4:
            for map_index in range(len(data)):
                p_map_highres_dataset_subindex = []
                for map_subindex in range(len(data[map_index])):
                    #Upsample the current map using bilinear interpolation
                    p_map_highres_dataset_subindex.append(ndimage.zoom(data[map_index][map_subindex], multiple, order=order))
                p_map_highres_dataset.append(p_map_highres_dataset_subindex)

        return p_map_highres_dataset



    def pad_pressure_mats(self,NxHxWimages):
        padded = np.zeros((NxHxWimages.shape[0],NxHxWimages.shape[1]+20,NxHxWimages.shape[2]+20))
        padded[:,10:74,10:37] = NxHxWimages
        NxHxWimages = padded
        return NxHxWimages


    def preprocessing_per_im_norm(self, images, CTRL_PNL):

        if CTRL_PNL['depth_map_input_est'] == True:
            pmat_sum = 1./(torch.sum(torch.sum(images[:, 3, :, :], dim=1), dim=1)/100000.)
            sobel_sum = 1./(torch.sum(torch.sum(images[:, 4, :, :], dim=1), dim=1)/100000.)

            print ("ConvNet input size: ", images.size(), pmat_sum.size())
            for i in range(images.size()[1]):
                print (i, torch.min(images[0, i, :, :]), torch.max(images[0, i, :, :]))

            images[:, 3, :, :] = (images[:, 3, :, :].permute(1, 2, 0)*pmat_sum).permute(2, 0, 1)
            images[:, 4, :, :] = (images[:, 4, :, :].permute(1, 2, 0)*sobel_sum).permute(2, 0, 1)

        else:
            pmat_sum = 1./(torch.sum(torch.sum(images[:, 1, :, :], dim=1), dim=1)/100000.)
            sobel_sum = 1./(torch.sum(torch.sum(images[:, 2, :, :], dim=1), dim=1)/100000.)

            print ("ConvNet input size: ", images.size(), pmat_sum.size())
            for i in range(images.size()[1]):
                print (i, torch.min(images[0, i, :, :]), torch.max(images[0, i, :, :]))


            images[:, 1, :, :] = (images[:, 1, :, :].permute(1, 2, 0)*pmat_sum).permute(2, 0, 1)
            images[:, 2, :, :] = (images[:, 2, :, :].permute(1, 2, 0)*sobel_sum).permute(2, 0, 1)



        #do this ONLY to pressure and sobel. scale the others to get them in a reasonable range, by a constant factor.


        return images
