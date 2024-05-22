import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import scipy.stats as ss
import torchvision
import time


from visualization_lib_bp import VisualizationLib
from kinematics_lib_bp import KinematicsLib
from mesh_depth_lib_bp import MeshDepthLib


class SMPL_PMR(nn.Module):
    def __init__(self, loss_vector_type, batch_size, verts_list, CTRL_PNL = None):
        '''
        Create components of a CNN classifier and initialize their weights.

        Arguments:
            mat_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            out_size (int): Number of classes to score
        '''

        super(SMPL_PMR, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        #print mat_size
        self.loss_vector_type = loss_vector_type
        print ('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

        self.count = 0



        if torch.cuda.is_available():
            self.GPU = True
            # Use for self.GPU
            dtype = torch.cuda.FloatTensor
            print('######################### CUDA is available! #############################')

        else:
            self.GPU = False
            # Use for CPU
            dtype = torch.FloatTensor
            print('############################## USING CPU #################################')

        self.dtype = dtype

        self.verts_list = verts_list
        self.meshDepthLib = MeshDepthLib(loss_vector_type, batch_size, use_cuda = self.GPU, verts_list = self.verts_list, CTRL_PNL = CTRL_PNL)

        self.zeros_z = torch.zeros(128, 24, 1).type(self.dtype)










    def forward_kinematic_angles_ptB(self, images, scores, gender_switch, synth_real_switch, CTRL_PNL, OUTPUT_DICT, OUTPUT_EST_DICT, INPUT_DICT = None,
                                 targets=None, is_training = True, betas=None, angles_gt = None, root_shift = None):


        reg_angles = CTRL_PNL['regr_angles']

        self.GPU = CTRL_PNL['GPU']
        self.dtype = CTRL_PNL['dtype']


        if CTRL_PNL['first_pass'] == False:
            x = self.meshDepthLib.bounds
            #print blah
            #self.GPU = False
            #self.dtype = torch.FloatTensor

        else:
            if CTRL_PNL['GPU'] == True:
                self.GPU = True
                self.dtype = torch.cuda.FloatTensor
            else:
                self.GPU = False
                self.dtype = torch.FloatTensor
            if CTRL_PNL['mesh_recon_output'] == True:
                self.verts_list = "all"
            else:
                self.verts_list = [1325, 336, 1032, 4515, 1374, 4848, 1739, 5209, 1960, 5423]
            self.meshDepthLib = MeshDepthLib(loss_vector_type=self.loss_vector_type,
                                             batch_size=scores.size(0), use_cuda = self.GPU, verts_list = self.verts_list)

        #print ("ConvNet input size: ", images.size())





        # weight the outputs, which are already centered around 0. First make them uniformly smaller than the direct output, which is too large.
        if CTRL_PNL['adjust_ang_from_est'] == True:
            scores = torch.mul(scores.clone(), 0.01)
        else:
            scores = torch.mul(scores.clone(), 0.01)




        if CTRL_PNL['CNN'] == 'resnet':

            if CTRL_PNL['adjust_ang_from_est'] == True:
                scores[:, 88:89] = scores[:, 88:89].clone() + OUTPUT_EST_DICT['bed_vertical_shift']

            if CTRL_PNL['compute_forward_maps'] == True and is_training == True and CTRL_PNL['CNN'] == 'resnet':
                print('RANDOM ADDITION')
                total_noise_factor = torch.Tensor(np.clip(np.random.normal(loc=1, scale=0.5, size=scores.size()[0]), a_min = 0, a_max = 2)).unsqueeze(1).type(CTRL_PNL['dtype'])
                random_addition = torch.Tensor(np.random.normal(0, 0.01 / 2, scores.size()[0])).type(CTRL_PNL['dtype']).unsqueeze(1)
                scores[:, 88:89] = scores[:, 88:89].clone() + random_addition*total_noise_factor

            OUTPUT_DICT['bed_vertical_shift_est'] = scores[:, 88:89].clone()
            scores = scores[:, 0:88]



        #normalize the output of the network based on the range of the parameters
        #if self.GPU == True:
        #    output_norm = 10*[6.0] + [0.91, 1.98, 0.15] + 6*[2.0] + list(torch.abs(self.meshDepthLib.bounds.view(72, 2)[3:, 1] - self.meshDepthLib.bounds.view(72,2)[3:, 0]).cpu().numpy())
        #else:
        #    output_norm = 10*[6.0] + [0.91, 1.98, 0.15] + 6*[2.0] + list(torch.abs(self.meshDepthLib.bounds.view(72, 2)[3:, 1] - self.meshDepthLib.bounds.view(72, 2)[3:, 0]).numpy())
        #for i in range(88):
        #    scores[:, i] = torch.mul(scores[:, i].clone(), output_norm[i])


        #add a factor so the model starts close to the home position. Has nothing to do with weighting.

        if CTRL_PNL['adjust_ang_from_est'] == True:
            pass
        else:
            scores[:, 10] = torch.add(scores[:, 10].clone(), 0.6-0.286)
            scores[:, 11] = torch.add(scores[:, 11].clone(), 1.2-0.286)
            scores[:, 12] = torch.add(scores[:, 12].clone(), 0.1)

        #scores[:, 12] = torch.add(scores[:, 12].clone(), 0.06)

        scores = scores.unsqueeze(0)
        scores = scores.unsqueeze(0)
        scores = F.pad(scores, (0, 3, 0, 0))
        scores = scores.squeeze(0)
        scores = scores.squeeze(0)

        if CTRL_PNL['adjust_ang_from_est'] == True:
            scores[:, 13:19] = scores[:, 13:19].clone() + OUTPUT_EST_DICT['root_atan2']

        if CTRL_PNL['compute_forward_maps'] == True and is_training == True and CTRL_PNL['CNN'] == 'resnet':
            pose_mult = 0.25 #was 0.25
            random_addition = torch.Tensor(np.stack((np.random.normal(0, 0.0036975095458499085*pose_mult/2, scores.size()[0]),
                                                     np.random.normal(0, 0.7379464882409956*pose_mult/2, scores.size()[0]),
                                                     np.random.normal(0, 0.034447491335482146*pose_mult/2, scores.size()[0]),
                                                     np.random.normal(0, 0.033630596327447373*pose_mult/2, scores.size()[0]),
                                                     np.random.normal(0, 0.5292844654709228*pose_mult/2, scores.size()[0]),
                                                     np.random.normal(0, 0.24211357487578994*pose_mult/2, scores.size()[0])))).type(CTRL_PNL['dtype']).permute(1,0)

            scores[:, 13:19] = scores[:, 13:19].clone() + random_addition*total_noise_factor




        scores[:, 22:91] = scores[:, 19:88].clone()

        if CTRL_PNL['nosmpl'] == True:
            scores = scores.clone()*0.0 + 0.1

        scores[:, 19] = torch.atan2(scores[:, 16].clone(), scores[:, 13].clone()) #pitch x, y
        scores[:, 20] = torch.atan2(scores[:, 17].clone(), scores[:, 14].clone()) #roll x, y
        scores[:, 21] = torch.atan2(scores[:, 18].clone(), scores[:, 15].clone()) #yaw x, y

        if CTRL_PNL['nosmpl'] == True:
            scores = scores.clone()*0.0 + 0.1


        #scores[:, 0:2] = scores[:, 0:2].clone()#-3
        #print scores[0, 0:10]
        if CTRL_PNL['adjust_ang_from_est'] == True:
            scores[:, 0:10] = scores[:, 0:10].clone() + OUTPUT_EST_DICT['betas']
            scores[:, 10:13] = scores[:, 10:13].clone() + OUTPUT_EST_DICT['root_shift']
            scores[:, 22:91] = scores[:, 22:91].clone() + OUTPUT_EST_DICT['angles'][:, 3:72]


        if CTRL_PNL['compute_forward_maps'] == True and is_training == True and CTRL_PNL['CNN'] == 'resnet':
            beta_mult = 0.25 #was 0.25
            random_addition1 = torch.Tensor(np.stack((np.random.normal(0, 1.72*beta_mult, scores.size()[0]),
                                                     np.random.normal(0, 1.72*beta_mult, scores.size()[0]),
                                                     np.random.normal(0, 1.72*beta_mult, scores.size()[0]),
                                                     np.random.normal(0, 1.72*beta_mult, scores.size()[0]),
                                                     np.random.normal(0, 1.72*beta_mult, scores.size()[0]),
                                                     np.random.normal(0, 1.72*beta_mult, scores.size()[0]),
                                                     np.random.normal(0, 1.72*beta_mult, scores.size()[0]),
                                                     np.random.normal(0, 1.72*beta_mult, scores.size()[0]),
                                                     np.random.normal(0, 1.72*beta_mult, scores.size()[0]),
                                                     np.random.normal(0, 1.72*beta_mult, scores.size()[0])))).type(CTRL_PNL['dtype']).permute(1,0)
            scores[:, 0:10] = scores[:, 0:10].clone() + random_addition1*total_noise_factor

            random_addition2 = torch.Tensor(np.stack((np.random.normal(0, 0.12020718153797369*pose_mult/2, scores.size()[0]),
                                                     np.random.normal(0, 0.11797362316016279*pose_mult/2, scores.size()[0]),
                                                     np.random.normal(0, 0.028776631603567603*pose_mult/2, scores.size()[0])))).type(CTRL_PNL['dtype']).permute(1,0)
            scores[:, 10:13] = scores[:, 10:13].clone() + random_addition2*total_noise_factor

            random_addition3 = torch.Tensor(np.stack((np.random.normal(0, 0.3684436716790615*pose_mult, scores.size()[0]),
                                                     np.random.normal(0, 0.3461566270768993*pose_mult, scores.size()[0]),
                                                     np.random.normal(0, 0.283979432140635*pose_mult, scores.size()[0]),
                                                     np.random.normal(0, 0.3617009781949601*pose_mult, scores.size()[0]),
                                                     np.random.normal(0, 0.3396290193491969*pose_mult, scores.size()[0]),
                                                     np.random.normal(0, 0.2871432548549813*pose_mult, scores.size()[0]),

                                                     np.random.normal(0, 0.07750461846574795*pose_mult, scores.size()[0]),
                                                     np.random.normal(0, 0.0442710571084925*pose_mult, scores.size()[0]),
                                                     np.random.normal(0, 0.042485319993514*pose_mult, scores.size()[0]),

                                                     np.random.normal(0, 0.6465153387153821*pose_mult, scores.size()[0]),
                                                     np.random.normal(0, 0.0, scores.size()[0]),
                                                     np.random.normal(0, 0.0, scores.size()[0]),
                                                     np.random.normal(0, 0.642104379832039*pose_mult, scores.size()[0]),
                                                     np.random.normal(0, 0.0, scores.size()[0]),
                                                     np.random.normal(0, 0.0, scores.size()[0]),

                                                     np.random.normal(0, 0.07371467374780329*pose_mult, scores.size()[0]),
                                                     np.random.normal(0, 0.03705504489195931*pose_mult, scores.size()[0]),
                                                     np.random.normal(0, 0.03291796657740526*pose_mult, scores.size()[0]),

                                                     np.random.normal(0, 0.12154123378141235*pose_mult, scores.size()[0]),
                                                     np.random.normal(0, 0.19086128815588266*pose_mult, scores.size()[0]),
                                                     np.random.normal(0, 0.10400887872770424*pose_mult, scores.size()[0]),
                                                     np.random.normal(0, 0.11746638553388697*pose_mult, scores.size()[0]),
                                                     np.random.normal(0, 0.18808133569678517*pose_mult, scores.size()[0]),
                                                     np.random.normal(0, 0.10640904240789341*pose_mult, scores.size()[0]),

                                                     np.random.normal(0, 0.06926121898087002*pose_mult, scores.size()[0]),
                                                     np.random.normal(0, 0.03860052288166030*pose_mult, scores.size()[0]),
                                                     np.random.normal(0, 0.03418401341066271*pose_mult, scores.size()[0]),

                                                     np.random.normal(0, 0.0, scores.size()[0]),
                                                     np.random.normal(0, 0.0, scores.size()[0]),
                                                     np.random.normal(0, 0.0, scores.size()[0]),
                                                     np.random.normal(0, 0.0, scores.size()[0]),
                                                     np.random.normal(0, 0.0, scores.size()[0]),
                                                     np.random.normal(0, 0.0, scores.size()[0]),

                                                     np.random.normal(0, 0.08909258750678141*pose_mult, scores.size()[0]),
                                                     np.random.normal(0, 0.039229170885139455*pose_mult, scores.size()[0]),
                                                     np.random.normal(0, 0.04644843816738842*pose_mult, scores.size()[0]),

                                                     np.random.normal(0, 0.3026503032494492*pose_mult, scores.size()[0]),
                                                     np.random.normal(0, 0.3085936435263838*pose_mult, scores.size()[0]),
                                                     np.random.normal(0, 0.3216202738080313*pose_mult, scores.size()[0]),
                                                     np.random.normal(0, 0.30065405442959997*pose_mult, scores.size()[0]),
                                                     np.random.normal(0, 0.31184399715220695*pose_mult, scores.size()[0]),
                                                     np.random.normal(0, 0.3198511876929381*pose_mult, scores.size()[0]),

                                                     np.random.normal(0, 0.0632549671559082*pose_mult, scores.size()[0]),
                                                     np.random.normal(0, 0.032347247930993826*pose_mult, scores.size()[0]),
                                                     np.random.normal(0, 0.036873838115728946*pose_mult, scores.size()[0]),

                                                     np.random.normal(0, 0.48783899401813735*pose_mult, scores.size()[0]),
                                                     np.random.normal(0, 0.5889956373868273*pose_mult, scores.size()[0]),
                                                     np.random.normal(0, 0.5840850783711136*pose_mult, scores.size()[0]),
                                                     np.random.normal(0, 0.4841440309204227*pose_mult, scores.size()[0]),
                                                     np.random.normal(0, 0.5723436483116511*pose_mult, scores.size()[0]),
                                                     np.random.normal(0, 0.5738215310556809*pose_mult, scores.size()[0]),

                                                     np.random.normal(0, 0.0, scores.size()[0]),
                                                     np.random.normal(0, 0.5879644106105493*pose_mult, scores.size()[0]),
                                                     np.random.normal(0, 0.0, scores.size()[0]),
                                                     np.random.normal(0, 0.0, scores.size()[0]),
                                                     np.random.normal(0, 0.588753560356266*pose_mult, scores.size()[0]),
                                                     np.random.normal(0, 0.0, scores.size()[0]),

                                                     np.random.normal(0, 0.2306703695307409*pose_mult, scores.size()[0]),
                                                     np.random.normal(0, 0.16403956391447416*pose_mult, scores.size()[0]),
                                                     np.random.normal(0, 0.16708480272678095*pose_mult, scores.size()[0]),
                                                     np.random.normal(0, 0.23029305970470534*pose_mult, scores.size()[0]),
                                                     np.random.normal(0, 0.1606822521531769*pose_mult, scores.size()[0]),
                                                     np.random.normal(0, 0.1620146908612652*pose_mult, scores.size()[0]),

                                                     np.random.normal(0, 0.0, scores.size()[0]),
                                                     np.random.normal(0, 0.0, scores.size()[0]),
                                                     np.random.normal(0, 0.0, scores.size()[0]),
                                                     np.random.normal(0, 0.0, scores.size()[0]),
                                                     np.random.normal(0, 0.0, scores.size()[0]),
                                                     np.random.normal(0, 0.0, scores.size()[0])))).type(CTRL_PNL['dtype']).permute(1,0)
            scores[:, 22:91] = scores[:, 22:91].clone() + random_addition3*total_noise_factor






        OUTPUT_DICT['batch_betas_est'] = scores[:, 0:10].clone().data
        OUTPUT_DICT['batch_root_xyz_est'] = scores[:, 10:13].clone().data
        OUTPUT_DICT['batch_root_atan2_est'] = scores[:, 13:19].clone().data
        OUTPUT_DICT['batch_angles_est']  = scores[:, 19:91].clone().data


        if reg_angles == True:
            add_idx = 72
        else:
            add_idx = 0


        if CTRL_PNL['clip_betas'] == True:
            scores[:, 0:10] /= 3.
            scores[:, 0:10] = scores[:, 0:10].tanh()
            scores[:, 0:10] *= 3.

        test_ground_truth = False#can only use True when the dataset is entirely synthetic AND when we use anglesDC

        #is_training = True

        if test_ground_truth == False or is_training == False:
            # make sure the estimated betas are reasonable.

            #betas_est = scores[:, 0:10].clone()#.detach() #make sure to detach so the gradient flow of joints doesn't corrupt the betas
            root_shift_est = scores[:, 10:13].clone()


            # normalize for tan activation function
            scores[:, 19:91] -= torch.mean(self.meshDepthLib.bounds[0:72, 0:2], dim=1)
            scores[:, 19:91] *= (2. / torch.abs(self.meshDepthLib.bounds[0:72, 0] - self.meshDepthLib.bounds[0:72, 1]))
            scores[:, 19:91] = scores[:, 19:91].tanh()
            scores[:, 19:91] /= (2. / torch.abs(self.meshDepthLib.bounds[0:72, 0] - self.meshDepthLib.bounds[0:72, 1]))
            scores[:, 19:91] += torch.mean(self.meshDepthLib.bounds[0:72, 0:2], dim=1)


            if CTRL_PNL['align_procr'] == True:
                print ("aligning procrustes")
                root_shift_est = root_shift
                scores[:, 19:22] = angles_gt[:, 0:3].clone()

            if self.loss_vector_type == 'anglesDC':

                Rs_est = KinematicsLib().batch_rodrigues(scores[:, 19:91].view(-1, 24, 3).clone()).view(-1, 24, 3, 3)

            elif self.loss_vector_type == 'anglesEU':

                Rs_est = KinematicsLib().batch_euler_to_R(scores[:, 19:91].view(-1, 24, 3).clone(), self.meshDepthLib.zeros_cartesian, self.meshDepthLib.ones_cartesian).view(-1, 24, 3, 3)

        else:
            scores[:, 0:10] = betas.clone()
            scores[:, 19:91] = angles_gt.clone()
            root_shift_est = root_shift

            #OUTPUT_DICT['bed_vertical_shift_est'] = OUTPUT_DICT['bed_vertical_shift_est'].clone()*0 + INPUT_DICT['bed_vertical_shift'].clone()
            OUTPUT_DICT['bed_vertical_shift_est'] =  INPUT_DICT['bed_vertical_shift'].clone()

            if self.loss_vector_type == 'anglesDC':
                Rs_est = KinematicsLib().batch_rodrigues(scores[:, 19:91].view(-1, 24, 3).clone()).view(-1, 24, 3, 3)


        OUTPUT_DICT['batch_betas_est_post_clip'] = scores[:, 0:10].clone()
        if self.loss_vector_type == 'anglesEU':
            OUTPUT_DICT['batch_angles_est_post_clip']  = KinematicsLib().batch_dir_cos_angles_from_euler_angles(scores[:, 19:91].view(-1, 24, 3).clone(), self.meshDepthLib.zeros_cartesian, self.meshDepthLib.ones_cartesian)
        elif self.loss_vector_type == 'anglesDC':
            OUTPUT_DICT['batch_angles_est_post_clip']  = scores[:, 19:91].view(-1, 24, 3).clone()
        OUTPUT_DICT['batch_root_xyz_est_post_clip'] = scores[:, 10:13].clone().data

        gender_switch = gender_switch.unsqueeze(1)
        current_batch_size = gender_switch.size()[0]


        if CTRL_PNL['mesh_recon_output'] == True:
            # break things up into sub batches and pass through the mesh
            num_normal_sub_batches = int(current_batch_size / self.meshDepthLib.N)
            if current_batch_size % self.meshDepthLib.N != 0:
                sub_batch_incr_list = num_normal_sub_batches * [int(self.meshDepthLib.N)] + [
                    current_batch_size % self.meshDepthLib.N]
            else:
                sub_batch_incr_list = num_normal_sub_batches * [int(self.meshDepthLib.N)]
            start_incr, end_incr = 0, 0

            for sub_batch_incr in sub_batch_incr_list:
                end_incr += sub_batch_incr
                verts_sub, J_est_sub, targets_est_sub = self.meshDepthLib.HMR(gender_switch, OUTPUT_DICT['batch_betas_est_post_clip'],
                                                                              Rs_est, root_shift_est,
                                                                              start_incr, end_incr, self.GPU)
                if start_incr == 0:
                    verts = verts_sub.clone()
                    J_est = J_est_sub.clone()
                    targets_est = targets_est_sub.clone()
                else:
                    verts = torch.cat((verts, verts_sub), dim=0)
                    J_est = torch.cat((J_est, J_est_sub), dim=0)
                    targets_est = torch.cat((targets_est, targets_est_sub), dim=0)
                start_incr += sub_batch_incr

            bed_ang_idx = -1
            if CTRL_PNL['incl_ht_wt_channels'] == True: bed_ang_idx -= 2
            bed_angle_batch = torch.mean(images[:, bed_ang_idx, 1:3, 0], dim=1)*0.

            if CTRL_PNL['compute_forward_maps'] == True:
                get_mesh_bottom_dist = False
            else:
                get_mesh_bottom_dist = True


            if CTRL_PNL['mesh_recon_map_output'] == True and CTRL_PNL['depth_out_unet'] == False:
                OUTPUT_DICT['batch_mdm_est'], OUTPUT_DICT['batch_cm_est'] = self.meshDepthLib.PMR(verts, bed_angle_batch, OUTPUT_DICT['bed_vertical_shift_est'],
                                                                                                  CTRL_PNL, get_mesh_bottom_dist = get_mesh_bottom_dist, is_training = is_training)
                OUTPUT_DICT['batch_mdm_est'] = OUTPUT_DICT['batch_mdm_est'].type(self.dtype)
                OUTPUT_DICT['batch_cm_est'] = OUTPUT_DICT['batch_cm_est'].type(self.dtype)



            verts_red = torch.stack([verts[:, 1325, :],
                                     verts[:, 336, :],  # head
                                     verts[:, 1032, :],  # l knee
                                     verts[:, 4515, :],  # r knee
                                     verts[:, 1374, :],  # l ankle
                                     verts[:, 4848, :],  # r ankle
                                     verts[:, 1739, :],  # l elbow
                                     verts[:, 5209, :],  # r elbow
                                     verts[:, 1960, :],  # l wrist
                                     verts[:, 5423, :]]).permute(1, 0, 2)  # r wrist

            verts_offset = verts_red.clone().detach().cpu()
            verts_offset = torch.Tensor(verts_offset.numpy()).type(self.dtype)

        else:
            #print("got here")
            shapedirs = torch.bmm(gender_switch, self.meshDepthLib.shapedirs[0:current_batch_size, :, :])\
                             .view(current_batch_size, self.meshDepthLib.B, self.meshDepthLib.R*self.meshDepthLib.D)

            betas_shapedirs_mult = torch.bmm(OUTPUT_DICT['batch_betas_est_post_clip'].unsqueeze(1), shapedirs)\
                                        .squeeze(1)\
                                        .view(current_batch_size, self.meshDepthLib.R, self.meshDepthLib.D)

            v_template = torch.bmm(gender_switch, self.meshDepthLib.v_template[0:current_batch_size, :, :])\
                              .view(current_batch_size, self.meshDepthLib.R, self.meshDepthLib.D)

            v_shaped = betas_shapedirs_mult + v_template

            J_regressor_repeat = torch.bmm(gender_switch, self.meshDepthLib.J_regressor[0:current_batch_size, :, :])\
                                      .view(current_batch_size, self.meshDepthLib.R, 24)

            Jx = torch.bmm(v_shaped[:, :, 0].unsqueeze(1), J_regressor_repeat).squeeze(1)
            Jy = torch.bmm(v_shaped[:, :, 1].unsqueeze(1), J_regressor_repeat).squeeze(1)
            Jz = torch.bmm(v_shaped[:, :, 2].unsqueeze(1), J_regressor_repeat).squeeze(1)


            J_est = torch.stack([Jx, Jy, Jz], dim=2)  # these are the joint locations with home pose (pose is 0 degree on all angles)
            #J_est = J_est - J_est[:, 0:1, :] + root_shift_est.unsqueeze(1)


            targets_est, A_est = KinematicsLib().batch_global_rigid_transformation(Rs_est, J_est, self.meshDepthLib.parents,
                                                                                   self.GPU, rotate_base=False)

            targets_est = targets_est - J_est[:, 0:1, :] + root_shift_est.unsqueeze(1)

            # assemble a reduced form of the transformed mesh
            v_shaped_red = torch.stack([v_shaped[:, self.verts_list[0], :],
                                        v_shaped[:, self.verts_list[1], :],  # head
                                        v_shaped[:, self.verts_list[2], :],  # l knee
                                        v_shaped[:, self.verts_list[3], :],  # r knee
                                        v_shaped[:, self.verts_list[4], :],  # l ankle
                                        v_shaped[:, self.verts_list[5], :],  # r ankle
                                        v_shaped[:, self.verts_list[6], :],  # l elbow
                                        v_shaped[:, self.verts_list[7], :],  # r elbow
                                        v_shaped[:, self.verts_list[8], :],  # l wrist
                                        v_shaped[:, self.verts_list[9], :]]).permute(1, 0, 2)  # r wrist
            pose_feature = (Rs_est[:, 1:, :, :]).sub(1.0, torch.eye(3).type(self.dtype)).view(-1, 207)
            posedirs_repeat = torch.bmm(gender_switch, self.meshDepthLib.posedirs[0:current_batch_size, :, :]) \
                .view(current_batch_size, 10 * self.meshDepthLib.D, 207) \
                .permute(0, 2, 1)
            v_posed = torch.bmm(pose_feature.unsqueeze(1), posedirs_repeat).view(-1, 10, self.meshDepthLib.D)
            v_posed = v_posed.clone() + v_shaped_red
            weights_repeat = torch.bmm(gender_switch, self.meshDepthLib.weights_repeat[0:current_batch_size, :, :]) \
                .squeeze(1) \
                .view(current_batch_size, 10, 24)
            T = torch.bmm(weights_repeat, A_est.view(current_batch_size, 24, 16)).view(current_batch_size, -1, 4, 4)
            v_posed_homo = torch.cat([v_posed, torch.ones(current_batch_size, v_posed.shape[1], 1).type(self.dtype)], dim=2)
            v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))


            verts = v_homo[:, :, :3, 0] - J_est[:, 0:1, :] + root_shift_est.unsqueeze(1)

            verts_offset = torch.Tensor(verts.clone().detach().cpu().numpy()).type(self.dtype)

            OUTPUT_DICT['batch_mdm_est'] = None
            OUTPUT_DICT['batch_cm_est'] = None


        #print verts[0:10], 'VERTS EST INIT'

        if CTRL_PNL['v2v'] == True:
            INPUT_DICT['batch_verts_gt'] = self.forward_kinematic_angles_gt(betas, angles_gt, root_shift, gender_switch)
            OUTPUT_DICT['batch_verts_est'] = verts.clone()

        OUTPUT_DICT['verts'] = verts.clone().detach().cpu().numpy()


        targets_est_detached = torch.Tensor(targets_est.clone().detach().cpu().numpy()).type(self.dtype)
        synth_joint_addressed = [3, 15, 4, 5, 7, 8, 18, 19, 20, 21]
        for real_joint in range(10):
            verts_offset[:, real_joint, :] = verts_offset[:, real_joint, :] - targets_est_detached[:, synth_joint_addressed[real_joint], :]


        #here we need to the ground truth to make it a surface point for the mocap markers
        #if is_training == True:
        synth_real_switch_repeated = synth_real_switch.unsqueeze(1).repeat(1, 3)
        for real_joint in range(10):
            targets_est[:, synth_joint_addressed[real_joint], :] = synth_real_switch_repeated * targets_est[:, synth_joint_addressed[real_joint], :].clone() \
                                   + torch.add(-synth_real_switch_repeated, 1) * (targets_est[:, synth_joint_addressed[real_joint], :].clone() + verts_offset[:, real_joint, :])


        targets_est = targets_est.contiguous().view(-1, 72)

        OUTPUT_DICT['batch_targets_est'] = targets_est.data*1000. #after it comes out of the forward kinematics

        scores = scores.unsqueeze(0)
        scores = scores.unsqueeze(0)
        scores = F.pad(scores, (0, 100 + add_idx, 0, 0))
        scores = scores.squeeze(0)
        scores = scores.squeeze(0)


        #tweak this to change the lengths vector
        scores[:, 40+add_idx:112+add_idx] = torch.mul(targets_est[:, 0:72], 1.)

        scores[:, 0:10] = torch.mul(synth_real_switch.unsqueeze(1), torch.sub(scores[:, 0:10], betas))#*.2

        scores[:, 10:16] = scores[:, 13:19].clone()
        if self.loss_vector_type == 'anglesEU':
            scores[:, 10:13] = scores[:, 10:13].clone() - torch.cos(KinematicsLib().batch_euler_angles_from_dir_cos_angles(angles_gt[:, 0:3].view(-1, 1, 3).clone()).contiguous().view(-1, 3))
            scores[:, 13:16] = scores[:, 13:16].clone() - torch.sin(KinematicsLib().batch_euler_angles_from_dir_cos_angles(angles_gt[:, 0:3].view(-1, 1, 3).clone()).contiguous().view(-1, 3))
        elif self.loss_vector_type == 'anglesDC':
            scores[:, 10:13] = scores[:, 10:13].clone() - torch.cos(angles_gt[:, 0:3].clone())
            scores[:, 13:16] = scores[:, 13:16].clone() - torch.sin(angles_gt[:, 0:3].clone())

            #print euler_root_rot_gt[0, :], 'body rot angles gt'

        #compare the output angles to the target values
        if reg_angles == True:
            if self.loss_vector_type == 'anglesDC':

                #print scores[0, 19:91].view(24, 3), '13 to 85'
                #print angles_gt[0, :].view(24, 3), 'angles GT'
                #print scores[0, 40:112].view(24, 3), '34 to 106'

                scores[:, 40:112] = angles_gt.clone().view(-1, 72) - scores[:, 19:91]
                #print scores[0, 40:112].view(24, 3), '34 to 106 post'
                #print torch.sum( scores[:, 40:112]), 'sum'

                #scores[:, 40:112] = torch.mul(synth_real_switch.unsqueeze(1), torch.sub(scores[:, 40:112], angles_gt.clone().view(-1, 72)))
                #print scores[0, 40:112].view(24, 3), '34 to 106 post2'

            elif self.loss_vector_type == 'anglesEU':
                scores[:, 40:112] = KinematicsLib().batch_euler_angles_from_dir_cos_angles(angles_gt.view(-1, 24, 3).clone()).contiguous().view(-1, 72) - scores[:, 19:91]

            scores[:, 40:112] = torch.mul(synth_real_switch.unsqueeze(1), scores[:, 40:112].clone())


        #compare the output joints to the target values

        scores[:, 40+add_idx:112+add_idx] = targets[:, 0:72]/1000 - scores[:, 40+add_idx:112+add_idx]
        scores[:, 112+add_idx:184+add_idx] = ((scores[:, 40+add_idx:112+add_idx].clone())+0.0000001).pow(2)


        for joint_num in range(24):
            if joint_num in [0, 1, 2, 6, 9, 10, 11, 12, 13, 14, 16, 17, 22, 23]: #torso is 3 but forget training it
                scores[:, 16+joint_num] = torch.mul(synth_real_switch,
                                                        (scores[:, 112+add_idx+joint_num*3] +
                                                         scores[:, 113+add_idx+joint_num*3] +
                                                         scores[:, 114+add_idx+joint_num*3]))
                scores[:, 16+joint_num] = scores[:, 16+joint_num].sqrt()


            else:
                scores[:, 16+joint_num] = (scores[:, 112+add_idx+joint_num*3] +
                                           scores[:, 113+add_idx+joint_num*3] +
                                           scores[:, 114+add_idx+joint_num*3]).sqrt()


        scores = scores.unsqueeze(0)
        scores = scores.unsqueeze(0)
        scores = F.pad(scores, (0, -151, 0, 0))
        scores = scores.squeeze(0)
        scores = scores.squeeze(0)


        scores[:, 0:10] = torch.mul(scores[:, 0:10].clone(), (1/1.728158146914805))#1.7312621950698526)) #weight the betas by std
        scores[:, 10:16] = torch.mul(scores[:, 10:16].clone(), (1/0.3684988513298487))#0.2130542427733348)*np.pi) #weight the body rotation by the std
        scores[:, 16:40] = torch.mul(scores[:, 16:40].clone(), (1/0.1752780723422608))#0.1282715100608753)) #weight the 24 joints by std
        if reg_angles == True:
            #print scores[0, 34 + OSA:106 + OSA].view(24, 3), '34 to 106 post3'
            #scores[:, 40:112] = torch.mul(scores[:, 40:112].clone(), (1/0.29641429463719227))#0.2130542427733348)) #weight the angles by how many there are
            scores[:, 43:112] = torch.mul(scores[:, 43:112].clone(), (1/0.29641429463719227))#0.2130542427733348)) #weight the angles by how many there are

        #scores[:, 0:10] = torch.mul(scores[:, 0:10].clone(), (1./10)) #weight the betas by how many betas there are
        #scores[:, 10:34] = torch.mul(scores[:, 10:34].clone(), (1./24)) #weight the joints by how many there are
        #if reg_angles == True: scores[:, 34:106] = torch.mul(scores[:, 34:106].clone(), (1./72)) #weight the angles by how many there are


        return scores, OUTPUT_DICT




    def forward_kinematic_angles_gt(self, betas, angles_gt, root_shift, gender_switch):

        Rs_est = KinematicsLib().batch_rodrigues(angles_gt.view(-1, 24, 3).clone()).view(-1, 24, 3, 3)
        current_batch_size = gender_switch.size()[0]

        num_normal_sub_batches = int(current_batch_size / self.meshDepthLib.N)
        if current_batch_size % self.meshDepthLib.N != 0:
            sub_batch_incr_list = num_normal_sub_batches * [int(self.meshDepthLib.N)] + [
                current_batch_size % self.meshDepthLib.N]
        else:
            sub_batch_incr_list = num_normal_sub_batches * [int(self.meshDepthLib.N)]
        start_incr, end_incr = 0, 0

        for sub_batch_incr in sub_batch_incr_list:
            end_incr += sub_batch_incr
            verts_sub, J_est_sub, targets_est_sub = self.meshDepthLib.HMR(gender_switch, betas,
                                                                          Rs_est, root_shift,
                                                                          start_incr, end_incr, self.GPU)
            if start_incr == 0:
                verts = verts_sub.clone()
                J_est = J_est_sub.clone()
                targets_est = targets_est_sub.clone()
            else:
                verts = torch.cat((verts, verts_sub), dim=0)
                J_est = torch.cat((J_est, J_est_sub), dim=0)
                targets_est = torch.cat((targets_est, targets_est_sub), dim=0)
            start_incr += sub_batch_incr

        return verts