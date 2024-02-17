import json  
import torch  
import torchaudio  
from torch.utils.data import Dataset
import numpy as np  
import librosa  
from sklearn.preprocessing import scale

from speechbrain.lobes.augment import EnvCorrupt
# 打开文件并读取JSON数据  
#with open('../aug_data/test_Cougher_1_1.json', 'r') as f:  
#    data = json.load(f)  
  
# 输出JSON数据  
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
        self.file = file_j #json文件路径
        #self.dictlist = os.listdir(data_dir_json)
        with open(file_j, 'r') as f:  
            self.dictlist = json.load(f) #音频文件信息字典
        with open(lb_dic, 'r') as d:  
            self.lb_dic = json.load(d) #音频文件信息字典
        #音频文件采样率
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
        
        #print(filename)
        #print(audio_array.shape)
        #audio_array = scale(audio_array) #去除均值和方差缩放：通过(X-X_mean)/std计算每个属性(每列)，进而使所有数据聚集在0附近，方差为1.
        # 使用np.pad函数填充每个通道的音频数据
        #padded_audio = np.pad(audio_array.squeeze().numpy(), (0, self.max_length - len(audio_array.squeeze().numpy())), 'constant')
        #print(padded_audio.shape)
        #+++1+++ Fbank        
            # 计算帧大小（窗口大小）和帧移  
        frame_length = int(sr*0.025)  # 25ms  
        hop_length = int(sr*0.01)  # 10ms  
        
        # 使用librosa.feature.melspectrogram()函数计算Fbank特征  
        melspec = librosa.feature.melspectrogram(y=audio_array.squeeze().numpy(), sr=sr, n_mels=23, hop_length=hop_length, n_fft=frame_length)
        
        # 将功率转换为dB比例  
        fbank = librosa.power_to_db(melspec, ref=np.max)  
        
        fbank = torch.from_numpy(fbank) # 转换为PyTorch张量
        #print(fbank.shape)
        # Normalization sentence 归一化操作 语句级别
        fbank = (fbank - fbank.mean()) / fbank.std()
        #将label转换为数值
        label_indices = torch.tensor([self.lb_dic[label]])
        return fbank,  label_indices

#对每一批数据做补丁，是所有的数据的尺寸保持一致
def collate_fn(batch):
    # 获取每个样本的尺寸
    #print(len(batch))
    sizes = [len(x[0][0]) for x in batch]

    #for x in batch :
        #print(dir(x))
        #print(x[0].shape)
        #print(x[0][0].shape)
        #print(x[0][1].shape)
        #print(x[1])
    # 计算批中最大的尺寸
    max_size = max(sizes)
    #print(max_size)
    # 对每个样本进行补丁操作，使其尺寸一致
    padded_batch = [(torch.nn.functional.pad(x[0], (0, max_size - len(x[0][1]))), x[1]) for x in batch]

    # 拆分为数据和标签
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


'''
# 创建数据集对象 test_Cougher_pick8_agu.json
#train_dt = CougherDataset('../aug_data/train_Cougher_pick8_agu.json')
#valid_dt = CougherDataset('../aug_data/valid_Cougher_pick8_agu.json')
file_j = '../aug_data/test_Cougher_pick16_noagu.json'
lb_dic = '../aug_data/Label_index16_noagu.json'
test_dt = CougherDataset(file_j,lb_dic)

#可以使用torch.utils.data.DataLoader来创建一个数据加载器，用于批量加载和处理数据。示例代码如下：
from torch.utils.data import DataLoader
test_dt_ld = DataLoader(test_dt, batch_size=4, collate_fn=collate_fn, shuffle=True)

#可以使用数据加载器来迭代加载数据。示例代码如下：

for data_batch, label_batch in test_dt_ld:
    print('data_batch',type(data_batch),data_batch.shape)
    #for tens in data_batch:
    #    print(tens)
    print('label_batch',type(label_batch),label_batch.shape) #,fbank.shape
    print(label_batch)
'''