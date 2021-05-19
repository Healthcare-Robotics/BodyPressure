#!/bin/bash
mkdir -p ./slp_real_cleaned

wget -O ./slp_real_cleaned/depth_uncover_cleaned_0to102.npy slp_real_cleaned https://dataverse.harvard.edu/api/access/datafile/4641889
wget -O ./slp_real_cleaned/depth_cover1_cleaned_0to102.npy slp_real_cleaned https://dataverse.harvard.edu/api/access/datafile/4641886
wget -O ./slp_real_cleaned/depth_cover2_cleaned_0to102.npy slp_real_cleaned https://dataverse.harvard.edu/api/access/datafile/4641890
wget -O ./slp_real_cleaned/depth_onlyhuman_0to102.npy slp_real_cleaned https://dataverse.harvard.edu/api/access/datafile/1888
wget -O ./slp_real_cleaned/O_T_slp_0to102.npy slp_real_cleaned https://dataverse.harvard.edu/api/access/datafile/4641893
wget -O ./slp_real_cleaned/slp_T_cam_0to102.npy slp_real_cleaned https://dataverse.harvard.edu/api/access/datafile/4641887
wget -O ./slp_real_cleaned/pressure_recon_Pplus_gt_0to102.npy slp_real_cleaned https://dataverse.harvard.edu/api/access/datafile/4641892
wget -O ./slp_real_cleaned/pressure_recon_C_Pplus_gt_0to102.npy slp_real_cleaned https://dataverse.harvard.edu/api/access/datafile/4641891


