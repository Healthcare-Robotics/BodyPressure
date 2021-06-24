# BodyPressure - v1.0
## Inferring Body Pose and Contact Pressure from a Depth Image

<p align="center">
  <img width="98%" src="https://github.com/Healthcare-Robotics/BodyPressure/blob/master/docs/figures/front_page_fig_v3.JPG?raw=true" alt="None"/>
</p>    

### Paper:
Clever, Henry M., Patrick Grady, Greg Turk, and Charles C. Kemp. "BodyPressure: Inferring Body Pose and Contact Pressure from a Depth Image." ArXiv preprint: https://arxiv.org/pdf/2105.09936.pdf

### Code version (v1.0) note:
This is the "initial submission" version of the code.


## What code is in here?

This repository: 
* Describes how to get started and download the data.
* Has a tool to visualize the SLP-3Dfits SMPL fits to the SLP dataset. 
* Has a tool to visualize the BodyPressureSD dataset. 
* Has instructions for training BodyPressureWnet and BodyPressureBnet. 
* Has instructions for testing BodyPressureWnet and BodyPressureBnet on real SLP data.
* Has code for generating resting bodies, synthetic pressure images, and synthetic depth images with FleX, DartFleX, and pyRender.


## License
Please read the licenses carefully. Currently, we have research only and non-commercial licenses on this repository and on the BodyPressureSD dataset because we filed a provisional patent on the method to infer pressure underneath a person using a camera. We are actively looking for someone who is interested in commercializing this, so if you are, please contact `henryclever@gatech.edu`. If nothing comes of the provisional patent by the time it expires, we intend to make this repository as open-source as possible. The SLP-3Dfits data has an MIT license, and the code for it is in a different repo. Additionally, please note and be aware of all third party software licenses that apply.


## Getting started

### Setup code:
Clone this repository to get started with inspecting the DepthPress data and training the deep network variants.\
`git clone https://github.com/Healthcare-Robotics/BodyPressure.git`\
`cd BodyPressure`\
`pip install -r requirements.txt`

If it's missing any requirements please create an issue and I will fix it.

Change `FILEPATH.txt` to reference the location of this folder on your computer.

Download SMPL human model, you must create a free account here https://smpl.is.tue.mpg.de/en. Copy smpl directory to `BodyPressure/smpl`. 

This repository uses Python 3.6, with the exception of the BodyPressureSD synthetic data generation code (which is 2.7).

### Download data:
* `cd data_BP`

Simultaneously-collected multimodal Lying Pose dataset: Follow the instructions on the site below.
* https://web.northeastern.edu/ostadabbas/2019/06/27/multimodal-in-bed-pose-estimation/
* Then, put it into `data_BP/SLP/`

Cleaned up and calibrated real data addendum: SLP depth images with in-fill for where hair causes noise and where viewframe drops off side of the bed, spatial transforms from the camera to the bed, and ground truth reconstructed pressure maps P+. This is 484 MB.
* Run `./download_cleaned_SLP_data` to download 18,180 images, 9,090 maps, and associated transforms.
* Link to these cleaned up real images: https://doi.org/10.7910/DVN/ZS7TQS

SLP-3Dfits data: 4,545 SMPL bodies fit to 101 participants in the SLP dataset using mesh vertex to point cloud point optimization. This is 18 MB.
* Follow the instructions in the following repo: https://github.com/pgrady3/SLP-3Dfits
* Then, put it in the `data_BP` folder.

BodyPressureSD synthetic dataset: 97,495 SMPL body shapes + poses with synthetic depth images and pressure images. This is 8.5 GB.
* Run `./download_BodyPressureSD` to download this data. 
* Link to BodyPressureSD: https://doi.org/10.7910/DVN/C6J1SP

Trained models: the best performing networks presented in the paper.
* Run `./download_BodyPressureWnet` to download Mod1 and Mod2 for the best performing white-box reconstruction network (177 MB).
* To use this with the training and evaluation code, you'll have to specify the correct flags. See sections below to understand what flags to use.
* Link to the models: https://doi.org/10.7910/DVN/8DJRNX

BodyPressureSD addendum: 3D environment meshes for human, blanket, deformed mattress, and deformed pressure sensing mat.
* You don't need this to get started.
* Link to this dataset addendum: https://doi.org/10.7910/DVN/DJWQPB (WARNING!! 148 GB).

Your file structure should look like this:

```
BodyPressure
├── data_BP
│   ├── convnets
│   │   ├── CAL_10665ct_128b_500e_0.0001lr.pt
│   │   ├── betanet_108160ct_128b_volfrac_500e_0.0001lr.pt
│   │   ├── resnet34_1_anglesDC_108160ct_128b_x1pm_rgangs_lb_slpb_dpns_rt_100e_0.0001lr.pt
│   │   └── resnet34_2_anglesDC_108160ct_128b_x1pm_0.5rtojtdpth_depthestin_angleadj_rgangs_lb_lv2v_slpb_dpns_rt_40e_0.0001lr.pt
│   │
│   ├── mod1est_real
│   ├── mod1est_synth
│   ├── results
│   ├── SLP
│   │   └── danaLab
│   │       ├── 00001
│   │       .
│   │       └── 00102
│   │   
│   ├── slp_real_cleaned
│   │   ├── depth_uncover_cleaned_0to102.npy
│   │   ├── depth_cover1_cleaned_0to102.npy
│   │   ├── depth_cover2_cleaned_0to102.npy
│   │   ├── depth_onlyhuman_0to102.npy
│   │   ├── O_T_slp_0to102.npy
│   │   ├── slp_T_cam_0to102.npy
│   │   ├── pressure_recon_Pplus_gt_0to102.npy
│   │   └── pressure_recon_C_Pplus_gt_0to102.npy
│   │   
│   ├── SLP_SMPL_fits
│   │   └── fits
│   │       ├── p001
│   │       .
│   │       └── p102
│   │   
│   ├── synth
│   │   ├── train_slp_lay_f_1to40_8549.p
│   │   .
│   │   └── train_slp_rside_m_71to80_1939.p
│   │   
│   ├── synth_depth
│   │   ├── train_slp_lay_f_1to40_8549_depthims.p
│   │   .
│   │   └── train_slp_rside_m_71to80_1939_depthims.p
│   │   
│   └── synth_meshes
│
├── docs
.
.
└── smpl
    ├── models
    ├── smpl_webuser
    └── smpl_webuser3
```

## SLP-3Dfits dataset visualization

To visualize a particular subject in the SLP-3Dfits dataset with pressure projection in 3D, run the following:
* `cd viz_data_only`
* `python viz_SLP3Dfits.py --p_idx 77 --pose_num 13 --ctype 'uncover' --viz '3D'`

This will do a 3D rendering of subject 77, pose number 13, with an uncovered point cloud. You can choose any subject from 1 to 102 or any pose from 1 to 45. All poses below are ground truth SLP-3Dfits (not deep model inferences). It shows them separately to better inspect correspondence with the pressure mat and point cloud. The green also has 3D joints on the SMPL model visualized.

<p align="center">
  <img width="27%" src="https://github.com/Healthcare-Robotics/BodyPressure/blob/master/docs/figures/00077_13_1.png?raw=true" alt="None"/>
  <img width="29%" src="https://github.com/Healthcare-Robotics/BodyPressure/blob/master/docs/figures/00077_13_2.png?raw=true" alt="None"/>
  <img width="34%" src="https://github.com/Healthcare-Robotics/BodyPressure/blob/master/docs/figures/00077_13_3.png?raw=true" alt="None"/>
</p>

To visualize a particular subject in the SLP-3Dfits dataset in a 2D rendering, run the following:
* `python3.6 viz_SLP3Dfits.py --p_idx 77 --pose_num 12 --ctype 'cover1' --viz '2D'`

<p align="center">
  <img width="30%" src="https://github.com/Healthcare-Robotics/BodyPressure/blob/master/docs/figures/00077_13_4.png?raw=true" alt="None"/>
</p>

Both pressure mats are ground truth.


## BodyPressureSD dataset visualization

To visualize a particular subject in the BodyPressureSD dataset with pressure projection in 3D, run the following:
* `cd viz_data_only`
* `python viz_BodyPressureSD.py --filenum 2 --pose_num 492 --viz '3D'`

This will do a 3D rendering of filenum 2, which corresponds to `train_slp_lside_f_1to40_8136.p`, on pose number 492. You can choose any file from 1 to 18 or any pose from 1 to 10000ish, however many poses are in the file. All poses below are ground truth BodyPressureSD data samples (not deep model inferences). It shows them separately to better inspect correspondence with the pressure mat. The green also has 3D joints on the SMPL model visualized.

<p align="center">
  <img width="34%" src="https://github.com/Healthcare-Robotics/BodyPressure/blob/master/docs/figures/02_492_1.png?raw=true" alt="None"/>
  <img width="32%" src="https://github.com/Healthcare-Robotics/BodyPressure/blob/master/docs/figures/02_492_2.png?raw=true" alt="None"/>
  <img width="32%" src="https://github.com/Healthcare-Robotics/BodyPressure/blob/master/docs/figures/02_492_3.png?raw=true" alt="None"/>
</p>

To visualize a particular subject in the SLP-3Dfits dataset in a 2D rendering, run the following:
* `python3.6 viz_BodyPressureSD.py --filenum 2 --pose_num 492 --viz '2D'`

<p align="center">
  <img width="30%" src="https://github.com/Healthcare-Robotics/BodyPressure/blob/master/docs/figures/02_492_4.png?raw=true" alt="None"/>
</p>

Pressure mat is ground truth.

## Training BodyPressure deep networks

There are 4 steps to train BodyPressureWnet or BodyPressureBnet as implemented in the paper.
* Step 1: Train the BetaNet and CAL components by running the following: `python3.6 train_BPXnet.py --X_is 'W'  --slp 'mixedreal' --train_only_betanet` and `python3.6 train_BPXnet.py --X_is 'W'  --slp 'real' --train_only_CAL`. These shouldn't take more than an hour or so. You don't need CAL for BPBnet.

* Step 2: Train Mod1 for 100 epochs using loss function 1 (about 12 hrs on my machine). Run the following for BPWnet: `python train_BPXnet.py --X_is 'W' --mod 1 --slp 'mixedreal'`. Run the following for BPBnet: `python train_BPXnet.py --X_is 'B' --mod 1 --slp 'mixedreal'`. This will train with a mixed real and synthetic dataset of 108160 images. If you change the `--slp` flag to `real` or `synth` it will train with 10665 real or 97495 synthetic images, respectively. There are various other flags in the `lib_py/optparse_lib.py` file that you can use to alter the loss function or do other things. If you don't have enough CPU RAM to load it all, then comment out some of the synthetic data files in the `get_slpsynth_pressurepose` function in `filename_input_lib_bp.py`. It's important to visualize things to make sure your network is training OK. So if you use the `--viz` flag a set of pressure maps pops up with joint markers projected into 2D - there are 24 of them. Green - ground truth, yellow - estimated. Note the ground truth pressure and contact images at the bottom right. This is just here to show you that there are in fact ground truth images available, but these are not used when training mod1.


<p align="center">
  <img width="80%" src="https://github.com/Healthcare-Robotics/BodyPressure/blob/master/docs/figures/train_BPWnet_mod1.jpg?raw=true" alt="None"/>
</p>



* Step 3: Compute a new dataset that has spatial map reconstructions from the PMR output of Mod1. Run the following: `python compute_depthmod1_spatialmaps.py --X_is 'W' --slp 'mixedreal'`. Make sure the flags on this match the flags you trained Mod1 on, but omit `--mod 1`. This will create a copy of the existing dataset plus estimated residual depth maps in separate files with longer filename tags. It will put these files in `data_BP/mod1est_real` and `data_BP/mod1est_synth`.  Make sure you have at least 10GB free.

* Step 4: Train Mod2 for 40 epochs using loss function 2. Run the following:  `python train_BPXnet.py --X_is 'W' --mod 2 --pmr --slp 'mixedreal' --v2v`, or alternatively `python train_BPXnet.py --X_is 'B' --mod 2 --slp 'mixedreal' --v2v`. If you do visualize, expect a box like the one below to pop up (for BPWnet; BPBnet is a bit different). This shows a lot more images because Mod2 inputs reconstructed depth maps from Mod1 and it computes a loss on output maps. See the paper to better understand these maps and their corresponding variables. Note the black rectangle on the input depth image- this is a part of the synthetic occlusion that is being used to add noise to the input data, which was from the SLP dataset code.


<p align="center">
  <img width="80%" src="https://github.com/Healthcare-Robotics/BodyPressure/blob/master/docs/figures/train_BPWnet_mod2_pmr.jpg?raw=true" alt="None"/>
</p>

* NOTE: These were trained on a RTX-3090 GPU, which has 24 GB ram. BPBnet won't fit on anything much smaller unless you cut the overall batch size. BPWnet will though. Change the `batch_sub_divider` variable in line 231 of `lib_py/mesh_depth_lib_bp.py` to some multiple of 2, e.g. 2 or 4 or 8 and observe an improved memory footprint, at some cost to training speed, but no cost to the overall batch size.

## Testing BodyPressure deep networks on real SLP data

To test, you can visualize its output in different ways.
* Run `python evaluate_depthreal_slp.py --X_is 'W' --slp 'mixedreal'  --pmr --mod 2 --v2v --p_idx 83 --ctype 'cover2' --pose_num 25 --viz '3D'` to do a 3D rendering of the results. Choose a participant between 81 and 102, a pose between 1 and 45, and a cover type. For each of the four renderings below, the ground truth is shown on the right side - with a green mesh and point cloud, as well as the ground truth distributed pressure just below or above it. The estimated pose, contact pressure image, and estimated distribution of pressure are shown on the left side of each rendering.

<p align="center">
  <img width="23%" src="https://github.com/Healthcare-Robotics/BodyPressure/blob/master/docs/figures/eval_model_3D_1.png?raw=true" alt="None"/>
  <img width="23%" src="https://github.com/Healthcare-Robotics/BodyPressure/blob/master/docs/figures/eval_model_3D_2.png?raw=true" alt="None"/>
  <img width="24%" src="https://github.com/Healthcare-Robotics/BodyPressure/blob/master/docs/figures/eval_model_3D_3.png?raw=true" alt="None"/>
  <img width="20%" src="https://github.com/Healthcare-Robotics/BodyPressure/blob/master/docs/figures/eval_model_3D_4.png?raw=true" alt="None"/>
</p>


* Run `python evaluate_depthreal_slp.py --X_is 'W' --slp 'mixedreal'  --pmr --mod 2 --v2v --p_idx 83 --ctype 'cover2' --pose_num 25 --viz '2D'` to do a 2D rendering of the results.

<p align="center">
  <img width="70%" src="https://github.com/Healthcare-Robotics/BodyPressure/blob/master/docs/figures/eval_model_2D.png?raw=true" alt="None"/>
</p>

* Run `python evaluate_depthreal_slp.py --X_is 'W' --slp 'mixedreal'  --pmr --mod 2 --v2v --p_idx 83 --ctype 'cover2' --pose_num 25 --savefig` to save a picture with the results.


<p align="center">
  <img width="40%" src="https://github.com/Healthcare-Robotics/BodyPressure/blob/master/docs/figures/eval_model_savefig.png?raw=true" alt="None"/>
</p>

## Code for generating BodyPressureSD

* There is reference code for doing the random initial sampling from the real data in `slp_sampling/generate_pose_slp_prox.py`. I have not yet included code for the DART and FleX simulations. If you want this code, make a request and I will bundle it up in a zip file here to add it, but it is complicated enough that I won't be able to help every step of the way to get it going.

* There is also the code for rendering the deformed meshes (body, blanket, pressure mat, mattress) and generating depth imagery from them in `process data` folder. Some of this might be useful if you decide to work with the 3D mesh data I provided as an addendum. However, this code is not needed for training the models or using any of the BodyPressureSD dataset. I have not attempted to fix file references in it to make it work out of the box, with the exception of `process_data/create_depth_ims_slp.py`. If you download the meshes and run it, the following image should pop up, which renders basic RGB images in addition to depth images:


<p align="center">
  <img width="70%" src="https://github.com/Healthcare-Robotics/BodyPressure/blob/master/docs/figures/depth_save_example.png?raw=true" alt="None"/>
</p>
