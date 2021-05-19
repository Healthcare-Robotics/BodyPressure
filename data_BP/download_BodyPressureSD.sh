#!/bin/bash
mkdir -p ./synth

wget -O ./synth/train_slp_lay_f_1to40_8549.p synth https://dataverse.harvard.edu/api/access/datafile/4642064
wget -O ./synth/train_slp_lay_f_41to70_6608.p synth https://dataverse.harvard.edu/api/access/datafile/4642082
wget -O ./synth/train_slp_lay_f_71to80_2184.p synth https://dataverse.harvard.edu/api/access/datafile/4642090
wget -O ./synth/train_slp_lay_m_1to40_8493.p synth https://dataverse.harvard.edu/api/access/datafile/4642067
wget -O ./synth/train_slp_lay_m_41to70_6597.p synth https://dataverse.harvard.edu/api/access/datafile/4642073
wget -O ./synth/train_slp_lay_m_71to80_2188.p synth https://dataverse.harvard.edu/api/access/datafile/4642069

wget -O ./synth/train_slp_lside_f_1to40_8136.p synth https://dataverse.harvard.edu/api/access/datafile/4642070
wget -O ./synth/train_slp_lside_f_41to70_6158.p synth https://dataverse.harvard.edu/api/access/datafile/4642076
wget -O ./synth/train_slp_lside_f_71to80_2058.p synth https://dataverse.harvard.edu/api/access/datafile/4642072
wget -O ./synth/train_slp_lside_m_1to40_7761.p synth https://dataverse.harvard.edu/api/access/datafile/4642063
wget -O ./synth/train_slp_lside_m_41to70_5935.p synth https://dataverse.harvard.edu/api/access/datafile/4642083
wget -O ./synth/train_slp_lside_m_71to80_2002.p synth https://dataverse.harvard.edu/api/access/datafile/4642071

wget -O ./synth/train_slp_rside_f_1to40_7677.p synth https://dataverse.harvard.edu/api/access/datafile/4642085
wget -O ./synth/train_slp_rside_f_41to70_6006.p synth https://dataverse.harvard.edu/api/access/datafile/4642079
wget -O ./synth/train_slp_rside_f_71to80_2010.p synth https://dataverse.harvard.edu/api/access/datafile/4642080
wget -O ./synth/train_slp_rside_m_1to40_7377.p synth https://dataverse.harvard.edu/api/access/datafile/4642068
wget -O ./synth/train_slp_rside_m_41to70_5817.p synth https://dataverse.harvard.edu/api/access/datafile/4642092
wget -O ./synth/train_slp_rside_m_71to80_1939.p synth https://dataverse.harvard.edu/api/access/datafile/4642081


mkdir -p ./synth_depth

wget -O ./synth_depth/train_slp_lay_f_1to40_8549_depthims.p synth_depth https://dataverse.harvard.edu/api/access/datafile/4642074
wget -O ./synth_depth/train_slp_lay_f_41to70_6608_depthims.p synth_depth https://dataverse.harvard.edu/api/access/datafile/4542077
wget -O ./synth_depth/train_slp_lay_f_71to80_2184_depthims.p synth_depth https://dataverse.harvard.edu/api/access/datafile/4642065
wget -O ./synth_depth/train_slp_lay_m_1to40_8493_depthims.p synth_depth https://dataverse.harvard.edu/api/access/datafile/4642087
wget -O ./synth_depth/train_slp_lay_m_41to70_6597_depthims.p synth_depth https://dataverse.harvard.edu/api/access/datafile/4642094
wget -O ./synth_depth/train_slp_lay_m_71to80_2188_depthims.p synth_depth https://dataverse.harvard.edu/api/access/datafile/4642084

wget -O ./synth_depth/train_slp_lside_f_1to40_8136_depthims.p synth_depth https://dataverse.harvard.edu/api/access/datafile/4642075
wget -O ./synth_depth/train_slp_lside_f_41to70_6158_depthims.p synth_depth https://dataverse.harvard.edu/api/access/datafile/4642089
wget -O ./synth_depth/train_slp_lside_f_71to80_2058_depthims.p synth_depth https://dataverse.harvard.edu/api/access/datafile/4642062
wget -O ./synth_depth/train_slp_lside_m_1to40_7761_depthims.p synth_depth https://dataverse.harvard.edu/api/access/datafile/4642097
wget -O ./synth_depth/train_slp_lside_m_41to70_5935_depthims.p synth_depth https://dataverse.harvard.edu/api/access/datafile/4642066
wget -O ./synth_depth/train_slp_lside_m_71to80_2002_depthims.p synth_depth https://dataverse.harvard.edu/api/access/datafile/4642091

wget -O ./synth_depth/train_slp_rside_f_1to40_7677_depthims.p synth_depth https://dataverse.harvard.edu/api/access/datafile/4642093
wget -O ./synth_depth/train_slp_rside_f_41to70_6006_depthims.p synth_depth https://dataverse.harvard.edu/api/access/datafile/4642088
wget -O ./synth_depth/train_slp_rside_f_71to80_2010_depthims.p synth_depth https://dataverse.harvard.edu/api/access/datafile/4642078
wget -O ./synth_depth/train_slp_rside_m_1to40_7377_depthims.p synth_depth https://dataverse.harvard.edu/api/access/datafile/4642095
wget -O ./synth_depth/train_slp_rside_m_41to70_5817_depthims.p synth_depth https://dataverse.harvard.edu/api/access/datafile/4642096
wget -O ./synth_depth/train_slp_rside_m_71to80_1939_depthims.p synth_depth https://dataverse.harvard.edu/api/access/datafile/4642086

