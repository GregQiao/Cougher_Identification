import json  
import torch  
import torchaudio  
from torch.utils.data import Dataset
import numpy as np  
import librosa  
from sklearn.preprocessing import scale

from speechbrain.lobes.augment import EnvCorrupt

corrupter = EnvCorrupt(openrir_folder='F:/AI/IsCough_X_vector/aug_data',  #RIR folder
babble_prob= 0.0,
reverb_prob= 0.0,
noise_prob= 1.0 ,
noise_snr_low= 0,
noise_snr_high= 15)


class CougherDataset(Dataset):
    def __init__(self, file_j,lb_dic,win,hops):
        self.file = file_j 
        #self.dictlist = os.listdir(data_dir_json)
        with open(file_j, 'r') as f:  
            self.dictlist = json.load(f) 
        with open(lb_dic, 'r') as d:  
            self.lb_dic = json.load(d) 
     
        self.sampling_rate = 16000
        self.win = win
        self.hops = hops

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
        
        frame_length = int(sr*self.win)  # 25ms  int(sr*0.025) 
        hop_length = int(sr*self.hops)  # 10ms   int(sr*0.01)
        
       
        melspec = librosa.feature.melspectrogram(y=audio_array.squeeze().numpy(), sr=sr, n_mels=23, hop_length=hop_length, n_fft=frame_length)
        
        
        fbank = librosa.power_to_db(melspec, ref=np.max)  
        
        fbank = torch.from_numpy(fbank)
        #print(fbank.shape)
        
        fbank = (fbank - fbank.mean()) / fbank.std()
        
        label_indices = torch.tensor([self.lb_dic[label]])
        return fbank,  label_indices

def collate_fn(batch):
    #print(batch[0][0])
    #print(batch[0][0])
    #print(batch[0][1])
    #if len(batch)==1:
    #    #print(batch)
    #    data_batch=torch.permute(torch.Tensor(batch[0][0]),(1,0)).unsqueeze(0)
    #    label_batch=torch.Tensor(batch[0][1]).unsqueeze(0)
    #    print(data_batch.shape,label_batch)
    #    return data_batch,label_batch
    #else:    
        #print(batch.shape)
        #print(len(batch))
        
        sizes = [len(x[0][0]) for x in batch]
        max_size = max(sizes)
        padded_batch = [(torch.nn.functional.pad(x[0], (0, max_size - len(x[0][1]))), x[1]) for x in batch]
        
      
        data_bt = [x[0].unsqueeze(0) for x in padded_batch]
        data_batch = torch.stack(data_bt,dim=0)
        label_batch = torch.stack([x[1] for x in padded_batch],dim=0)
        #torch.stack([x[0] for x in padded_batch])
        #for x in data_bt :
            #print(x.shape)
        #print(data_bt[0])#([16, 1, 23, 49]) ([16, 1, 49, 23])
        data_batch = torch.permute(data_batch, (0, 1,3,2)).squeeze(1) 
        #print(data_batch.shape) #
        #print(label_batch.shape)
        #print(label_batch.squeeze(1))
        return data_batch, label_batch  #label_batch.squeeze(1)

