# Cougher_Identification
 Coughing person identification Pytorch. 
 
This codes is for the paper "Cougher-TDNN: Modified ECAPA-TDNN for Coughing Person Identification".

# Folders
The folder "aug_data" contains the dictionary of labels for the experiment, because the cougher dataset is not open source. So that's why there aren't wav files.

The folder "Ecapa" contains the codes experimental results and best model for the Ecapa-TDNN model. 

The folder "x-vector" contains the codes experimental results and best model for the x-vector model.

The folder "Resnet" contains the codes experimental results and best model for the Resnet50 model.

The folder "c-tdnn" contains the experimental results and best model for the Cougher-TDNN model.

# Codes:
_0_2_3_2_data_prepare_Cougher_pick166_noagu.py,_0_2_3_3_data_prepare_Cougher_pick166_noagu_80.py and _0_2_3_2_data_prepare_Cougher_pick166_noagu_86.py are the codes for creating label for experiment.

# Version of  AI tools
numpy                     1.20.1      
python                    3.8.8       
torch                     2.0.1+cu118 
torchaudio                2.0.2+cu118 
torchvision               0.15.2+cu118

