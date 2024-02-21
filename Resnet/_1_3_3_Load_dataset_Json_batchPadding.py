import json  
import torch  
import torchaudio  
from torch.utils.data import Dataset
import numpy as np  
import librosa  
from sklearn.preprocessing import scale

from speechbrain.lobes.augment import EnvCorrupt

#with open('../aug_data/test_Cougher_1_1.json', 'r') as f:  
#    data = json.load(f)  
  

#print(data)
#print(type(data))
#items_list = list(data.items()) 
#print(items_list[1])
#print(items_list[1][1]['cougher'])
corrupter = EnvCorrupt(openrir_folder='F:/AI/IsCough_X_vector/aug_data',
babble_prob= 0.0,
reverb_prob= 0.0,
noise_prob= 1.0 ,
noise_snr_low= 0,
noise_snr_high= 15)


class CougherDataset(Dataset):
    def __init__(self, file_j,lb_dic):
        self.file = file_j 
        #self.dictlist = os.listdir(data_dir_json)
        with open(file_j, 'r') as f:  
            self.dictlist = json.load(f) 
        with open(lb_dic, 'r') as d:  
            self.lb_dic = json.load(d) 
   
        self.sampling_rate = 16000

    def __len__(self):
        return len(self.dictlist)
    def __getitem__(self, index):
        items_list = list(self.dictlist.items()) 
        filename = items_list[index][1]['wav']
        label = items_list[index][1]['cougher']
        #print(index)
        #print(label)        
        sr=self.sampling_rate
        audio_array, sample_rate = librosa.load(filename, sr=sr, mono=True)
        audio_array = torch.from_numpy(audio_array).unsqueeze(0)
        #print(audio_array.shape)
        audio_array = corrupter(audio_array, torch.ones(audio_array.shape[0]))
        

        frame_length = int(sr*0.025)  # 25ms  
        hop_length = int(sr*0.01)  # 10ms  
        

        melspec = librosa.feature.melspectrogram(y=audio_array.squeeze().numpy(), sr=sr, n_mels=23, hop_length=hop_length, n_fft=frame_length)
        

        fbank = librosa.power_to_db(melspec, ref=np.max)  
        
        fbank = torch.from_numpy(fbank) 
        
        fbank = (fbank - fbank.mean()) / fbank.std()
        
      
        label_indices = torch.tensor([self.lb_dic[label]])
        return fbank,  label_indices

def collate_fn(batch):
    
    sizes = [len(x[0][0]) for x in batch]

   
    max_size = max(sizes)
   
    padded_batch = [(torch.nn.functional.pad(x[0], (0, max_size - len(x[0][1]))), x[1]) for x in batch]

    data_bt = [x[0].unsqueeze(0) for x in padded_batch]
    data_batch = torch.stack(data_bt,dim=0)
    label_batch = torch.stack([x[1] for x in padded_batch],dim=0)
    #torch.stack([x[0] for x in padded_batch])
    #for x in data_bt :
        #print(x.shape)
    #print(data_bt[0])
    return data_batch, label_batch.squeeze(1)


