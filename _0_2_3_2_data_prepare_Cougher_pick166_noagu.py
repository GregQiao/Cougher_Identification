"""
Creates data manifest files for cough data (childs & adults ).
For people, different kinds of the cough must appear in train,
validation, and test sets. In this case, these sets are thus derived from
splitting the original training set intothree chunks.

"""

import os
import json
import shutil
import random
import logging
from speechbrain.utils.data_utils import get_all_files, download_file
from speechbrain.dataio.dataio import read_audio

import pandas as pd  
from sklearn.model_selection import train_test_split  

from glob import glob
import math

logger = logging.getLogger(__name__)
SAMPLERATE = 16000

picknum= 166 
typ = 'noagu'

def prepare_cough(
    data_folder,
    save_json_train,
    save_json_valid,
    save_json_test,
    split_ratio=[80, 10, 10],
):
    """
    Prepares the json files for the cough dataset.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the Mini Librispeech dataset is stored.
    save_json_train : str
        Path where the train data specification file will be saved.
    save_json_valid : str
        Path where the validation data specification file will be saved.
    save_json_test : str
        Path where the test data specification file will be saved.
    split_ratio: list
        List composed of three integers that sets split ratios for train, valid,
        and test sets, respectively. For instance split_ratio=[80, 10, 10] will
        assign 80% of the sentences to training, 10% for validation, and 10%
        for test.

    Example
    -------
    >>> data_folder = '/path/to/mini_librispeech'
    >>> prepare_mini_librispeech(data_folder, 'train.json', 'valid.json', 'test.json')
    """

    # Check if this phase is already done (if so, skip it)
    #if skip(save_json_train, save_json_valid, save_json_test):
    #    logger.info("Preparation completed in previous run, skipping.")
    #    return

    # If the dataset doesn't exist yet, download it
    #train_folder = os.path.join(data_folder, "LibriSpeech", "train-clean-5")
    #if not check_folders(train_folder):
    #    download_mini_librispeech(data_folder)
    train_folder=data_folder
    # List files and create manifest from list
    logger.info(
        f"Creating {save_json_train}, {save_json_valid}, and {save_json_test}"
    )
    #extension = [".wav"]
    #wav_list = get_all_files(train_folder, match_and=extension)
    #lst80p = glob(data_folder + "/80_pure/*.wav")
    #lst42p = glob(data_folder + "/42_pure/*.wav")
    lst = glob(data_folder + "/pick_" + str(picknum) + "/*.wav")
    lst = [x for x in lst if not '_t' in x]  
    
    #lst80n = glob(data_folder + "/80_nos/*.wav")
    #lst42n = glob(data_folder + "/42_nos/*.wav")
 
    #lst80p.extend(lst80n)
    random.shuffle(lst)  
    #lst = lst80p[:46984]  
    #lst = lst80p[:len(lst42p)]
    #lst.extend(lst42n)
    
    random.shuffle(lst)
    #lists = lst80[:23495] 
    #lists.extend(lst42) 
    
    #random.shuffle(lstp)
    #f=open("random_20221010.txt","w")
    #f.write(str(lst))
    #f.close()
    # Random split the signal list into train, valid, and test sets.
    data_split = split_sets(lst, split_ratio)

    # Creating json files
    create_json(data_split["train"], save_json_train)
    create_json(data_split["valid"], save_json_valid)
    create_json(data_split["test"], save_json_test)

    
def create_json(wav_list, json_file):
    """
    Creates the json file given a list of wav files.

    Arguments
    ---------
    wav_list : list of str
        The list of wav files.
    json_file : str
        The path of the output json file
    """
    # Processing all the wav files in the list
    json_dict = {}    
    random.shuffle(wav_list)
    for wav_file in wav_list:
        #print(wav_file)
        # Reading the signal (to retrieve duration in seconds)
        signal = read_audio(wav_file)
        duration = signal.shape[0] / SAMPLERATE

        # Manipulate path to get relative path and uttid
        path_parts = wav_file.split(os.path.sep)
        uttid, _ = os.path.splitext(path_parts[-1])
        #relative_path = os.path.join("{data_root}", *path_parts[-5:])
        relative_path = os.path.join("", *path_parts[-5:])

        # Getting speaker-id from utterance-id
       # spk_id = uttid.split("-")[0]
        #Judging whether is a child
        cougher = wav_file.split ( "\\" )[-1].split ( "_" )[0]
#        if wav_file.count('_0.wav') > 0 :
#            child = "No"

        # Create entry for this utterance
        json_dict[uttid] = {
            "wav": relative_path,
            "length": duration,
            "cougher": cougher,
        }

    # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file} successfully created!")
    

    
def check_folders(*folders):
    """Returns False if any passed folder does not exist."""
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True


def update_dict(key1, key2, new_value):  

    key_tuple = (key1, key2)  
      
 
    if key_tuple in my_dict:  

        my_dict[key_tuple] = new_value  
    else:  

        my_dict[key_tuple] = new_value  
      
    return my_dict  
    
def split_sets(wav_list, split_ratio):
    """Randomly splits the wav list into training, validation, and test lists.
    Note that a better approach is to make sure that all the classes have the
    same proportion of samples (e.g, spk01 should have 80% of samples in
    training, 10% validation, 10% test, the same for speaker2 etc.). This
    is the approach followed in some recipes such as the Voxceleb one. For
    simplicity, we here simply split the full list without necessarily respecting
    the split ratio within each class.

    Arguments
    ---------
    wav_lst : list
        list of all the signals in the dataset
    split_ratio: list
        List composed of three integers that sets split ratios for train, valid,
        and test sets, respectively. For instance split_ratio=[80, 10, 10] will
        assign 80% of the sentences to training, 10% for validation, and 10%
        for test.

    Returns
    ------
    dictionary containing train, valid, and test splits.
    """
    data_dic = {}
    #print(len(wav_list))
      
    for wav_file in wav_list:

        cougher = wav_file.split ( "\\" )[-1].split ( "_" )[0]
        data_dic[cougher] = []
        #print(cougher)

    #items = list(data_dic.items())

    #items.sort() 

    #print(items[0])
    #print(items[0][0])
    
    keys = list(data_dic.keys())
    for key in keys:  
        #print(type(key))
        pth = []
        for wav_file in wav_list:

            #print(pth)
            cougher = wav_file.split ( "\\" )[-1].split ( "_" )[0]
            
            if key == cougher :
                pth.extend([wav_file])
        data_dic[key] = pth
        #print(type(cougher))
        #print(data_dic[cougher])
    
    #print(items[0][0],data_dic[items[0][0]])  #''.join(items[0])
    #print(data_dic)
    #print(len(data_dic))
  
    #df = pd.DataFrame(data_dic).T 

    #df['label'] = df.index  
 
    
    tot_split = sum(split_ratio)
    splits = ["train", "valid"]
    data_s = {}

    for key in keys:  
        tot_snts = len(data_dic[key])
        print(key,'this ppl nums:',tot_snts)
        n_snts = int(tot_snts * split_ratio[0] / tot_split)
        if (tot_snts - n_snts) % 2 ==0:
            data_s[('train',key)] = data_dic[key][0:n_snts]
            del data_dic[key][0:n_snts]
            print('train','nums:',len(data_s[('train',key)]))
            
            data_s[('valid',key)] = data_dic[key][0:int(len(data_dic[key])/2)]#
            del data_dic[key][0:int(len(data_dic[key])/2)]
            print('valid','nums:',len(data_s[('valid',key)]))
            
            data_s[("test",key)] = data_dic[key]
            print('test','nums:',len(data_s[('test',key)]))
        else :
            data_s[('train',key)] = data_dic[key][0:n_snts+1]
            del data_dic[key][0:n_snts+1]
            print('train','nums:',len(data_s[('train',key)]))
            
            data_s[('valid',key)] = data_dic[key][0:int(len(data_dic[key])/2)]#
            del data_dic[key][0:int(len(data_dic[key])/2)]
            print('valid','nums:',len(data_s[('valid',key)]))
            
            data_s[("test",key)] = data_dic[key]
            print('test','nums:',len(data_s[('test',key)]))
    
    data_split = {'train': [], 'valid': [], 'test': []}
    for i, split in enumerate(["train", "valid", "test"]):
        for key in keys:  
            data_split[split].extend(data_s[(split,key)])
    #print(data_split["test"])  

    return data_split




data_folder='../aug_data'
save_json_train=data_folder + '/train_Cougher_pick' + str(picknum) + '_'+ typ +'.json'
save_json_valid=data_folder +'/valid_Cougher_pick' + str(picknum) + '_'+ typ +'.json'
save_json_test=data_folder +'/test_Cougher_pick' + str(picknum) + '_'+ typ +'.json'
prepare_cough(
    data_folder,
    save_json_train,
    save_json_valid,
    save_json_test,
    split_ratio=[80, 10, 10])


