import torch  
import torchvision.models as models
from torch.utils.data import DataLoader
import json 
import os,time

#loss function
from speechbrain.nnet.losses import LogSoftmaxWrapper, AdditiveAngularMargin 
#scheduler 
from speechbrain.nnet.schedulers import CyclicLRScheduler 
from torch.optim.lr_scheduler import CyclicLR
#dataloader
from _1_3_0_Load_dataset_Json_batchPadding_EcapaSp_w import CougherDataset,collate_fn

#设置实验参数
#设置实验随机种子
import random
import numpy as np
# 设置随机种子
seed = 20986
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
   #cudnn.benchmark = True       
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

picknum = 166  #咳嗽人数量
pplnum = 86
typ = 'noagu_86'
lllr ='lra[122331][73][512]A_M_eca_fixsed_128_scl_c16'
win_s=18
hop_s=8
win = 'w'+str(win_s)+'[' + str(hop_s) +']'

n_mels =23  # F-bank的维度
lin_neur = 192 #ecapa输出的神经元的数量
'''
1. 导入Ecapa-tdnn模型
'''
from _0_3_3_ECAPA_TDNN_A_M_ECA_scl_128_model import ECAPA_TDNN  #,Classifier
'''
ECAPA_TDNN参数
    input_size,
    device : str  Device used, e.g., "cpu" or "cuda".
    activation : torch class
        A class for constructing the activation layers.
    channels : list of ints
        Output channels for TDNN/SERes2Net layer.
    kernel_sizes : list of ints
        List of kernel sizes for each layer.
    dilations : list of ints
        List of dilations for kernels in each layer.
    lin_neurons : int
        Number of neurons in linear layers.
    groups : list of ints
        List of groups for kernels in each layer.

classifier 参数
        input_size,
        device="cpu",
    lin_blocks : int
        Number of linear layers. default 0
    lin_neurons : int
        Number of neurons in linear layers. default 192
    out_neurons : int
        Number of classes.
'''
#选择CPU还是GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Initial ecapa model  #Tensor of shape (batch, time, channel).
ecapa = ECAPA_TDNN(input_size= n_mels
,channels          = [512, 512, 512, 512,512,2048] #[128, 128, 128, 128, 128, 512]
,kernel_sizes      = [7, 3, 3, 3,3, 1]
,dilations         = [1,2, 2, 3,3, 1]
,groups            = [1,1, 1,  1,1, 1]
,attention_channels= 128
,lin_neurons       = lin_neur
,out_neurons       = pplnum
,res2net_scale=16)
#Initial Classifier for ecapa model
#classify = Classifier( lin_neurons=lin_neur, input_size=lin_neur,out_neurons=picknum,device="cuda") #lin_blocks=0,


'''
1.1 导入标签
'''
lb_dic = '../aug_data/Label_index'+str(picknum)+'_'+typ +'.json'
with open(lb_dic, 'r') as d:  
    labels_dic = json.load(d) #音频文件信息字典
 
'''
2.更新全连接层。ResNet50模型原本是用于图像分类的，最后一层全连接层需要根据你的任务进行调整。这里假设你的任务是分类，你需要将最后一层的输出特征数更新为你的分类数量：
'''
batch_siz = 16  
learning_rate = 0.001  
num_epochs = 300  # 假设训练300个epoch


'''
3.定义损失函数和优化器。根据你的任务，选择相应的损失函数来计算损失值，并选择相应的优化器来更新模型的参数。这里假设你的任务是分类，使用交叉熵损失函数和随机梯度下降（SGD）优化器：
'''
#criterion = torch.nn.CrossEntropyLoss()
log_prob = LogSoftmaxWrapper(AdditiveAngularMargin(margin=0.2, scale=30))
#loss = log_prob(outputs, targets)


optimizer = torch.optim.Adam(ecapa.parameters(), lr=learning_rate, weight_decay = 2e-5)

## 现在我们来定义学习率调度器。
## 我们需要解决的问题是找到一个适当的gamma值，使得0.01乘以gamma的300次方等于0.001。
#gamma = 0.001**(1/300)
#scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma)
scheduler = CyclicLRScheduler(base_lr=0.00000001, max_lr=0.001, step_size=65000)


#torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up=2000, step_size_down=None, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=- 1, verbose=False)
#scheduler = CyclicLR(optimizer, base_lr=0.00000001, max_lr=0.001, step_size_up=65000, step_size_down=None, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=- 1, verbose=False)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)
# 创建数据集对象 test_Cougher_pick8_agu.json

train_dt = CougherDataset('../aug_data/train_Cougher_pick'+str(picknum)+'_'+ typ +'.json',lb_dic,win=win_s*0.001,hops=hop_s*0.001)
valid_dt = CougherDataset('../aug_data/valid_Cougher_pick'+str(picknum)+'_'+ typ +'.json',lb_dic,win=win_s*0.001,hops=hop_s*0.001) 
test_dt = CougherDataset('../aug_data/test_Cougher_pick'+str(picknum)+'_'+ typ +'.json',lb_dic,win=win_s*0.001,hops=hop_s*0.001)

#可以使用torch.utils.data.DataLoader来创建一个数据加载器，用于批量加载和处理数据。示例代码如下：
#train_ld = DataLoader(train_dt, batch_size=16, collate_fn=collate_fn, shuffle=True,drop_last=True)
#valid_ld = DataLoader(valid_dt, batch_size=16, collate_fn=collate_fn, shuffle=True,drop_last=True)
#test_ld = DataLoader(test_dt, batch_size=16, collate_fn=collate_fn, shuffle=True,drop_last=True)
train_ld = DataLoader(train_dt, batch_size=batch_siz, collate_fn=collate_fn, shuffle=True)
valid_ld = DataLoader(valid_dt, batch_size=batch_siz, collate_fn=collate_fn, shuffle=True)
test_ld = DataLoader(test_dt, batch_size=batch_siz, collate_fn=collate_fn, shuffle=True)

'''
7. 训练模型。利用数据加载器、损失函数和优化器进行模型训练。循环遍历训练集中的每个批次并执行以下操作：
  - 将输入和标签加载到GPU（如果可用）上；
  - 前向传播计算模型的输出；
  - 计算损失值；
  - 反向传播和梯度更新；
'''
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ecapa.to(device)  # 将模型移动到GPU（如果可用）
 
# 训练模型  
best_acc = 0.0 #用于保存最高的准确率


# 定义保存训练日志的文件路径  
#    os.remove(file_path) 
log_file = "./results/"+str(picknum)+typ+lllr+win+"/train_log.txt"   
#删除旧的日志文件
if os.path.isfile(log_file):  
    # 删除文件  
    os.remove(log_file) 

if os.path.exists('results') is False :
    os.mkdir('results')
   
if os.path.exists('./results/'+str(picknum)+typ+lllr+win) is False :
    os.mkdir('./results/'+str(picknum)+typ+lllr+win)
    

    
for epoch in range(num_epochs):
    ecapa.train()
    for inputs, labels in train_ld:
        inputs = inputs.to(device)
        labels = labels.to(device)
        #print(labels)
        #print(inputs)

        outputs = ecapa(inputs)
        #outputs = classify(outputs)
        loss = log_prob(outputs, labels) #criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    ## 在每个epoch结束后，你可以这样来更新你的learning rate
    scheduler.on_batch_end(optimizer) #scheduler.step()    
    with open(log_file, "a") as f:
        f.write(f'lr: {optimizer.param_groups[0]["lr"]}\n')  #scheduler.get_last_lr()
    print(optimizer.param_groups[0]["lr"]) #print(scheduler.get_last_lr()) # 打印上一次的学习率will print last learning rate.
    # 在验证集上评估模型性能  
    ecapa.eval()
    with torch.no_grad():  
        correct = 0  
        total = 0  
        for inputs, labels in valid_ld:  
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = ecapa(inputs)  
            #outputs = classify(outputs)
            #print(outputs.shape,labels.shape)
            val_loss = log_prob(outputs, labels) # log_prob => criterion
            max_values, max_indices = outputs.max(dim=2)  #_, predicted = torch.max(outputs.data, 1)  
            predicted = max_indices
            #print(predicted.shape,labels.shape)
            #print(labels)
            #print(outputs)
            total += labels.size(0)  
            correct += (predicted == labels).sum().item()  
        accuracy = 100 * correct / total  
        # 打开文件以写入训练日志  
        with open(log_file, "a") as f:  
         # 将损失值写入文件  f.write(f"Epoch {epoch+1}, Loss: {loss.item():.4f}\n") 
            f.write(f'Epoch {epoch+1}, Train Loss:{loss.item():.2f}, Valid_Loss:{val_loss.item():.2f}, Valid_Accuracy:{accuracy:.2f}% \n')
        print(f'Epoch: {epoch+1}, Train Loss: {loss.item():.2f}, Valid_Loss: {val_loss.item():.2f}, Valid_Accuracy: {accuracy:.2f}%')  
         # 保存最优模型
        if accuracy >= best_acc:
            best_acc = accuracy
            best_model_wts = ecapa.state_dict()
            last_epoch = epoch +1
        
# 保存最优模型的权重参数
torch.save(best_model_wts, "./results/"+str(picknum)+typ+lllr+win+'/best_model_{}.pth'.format(time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime())))
torch.save(best_model_wts, "./results/"+str(picknum)+typ+lllr+win+'/best_model.pth')
'''        
8.在测试集上评估模型。使用验证集对模型进行评估，计算准确率、精确率、召回率等指标。循环遍历验证集中的每个批次并执行以下操作：
  - 将输入和标签加载到GPU（如果可用）上；
  - 前向传播计算模型的输出；
  - 计算评估指标；
'''
ecapa.eval()  # 将模型设置为评估模式

# 加载最优模型
best_model_path = './results/'+str(picknum)+typ+lllr+win+'/best_model.pth'
ecapa.load_state_dict(torch.load(best_model_path))

with torch.no_grad():
    correct = 0
    total = 0

    for inputs, labels in test_ld:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = ecapa(inputs)
        #outputs = classify(outputs)
        tes_loss = log_prob(outputs, labels)  # log_prob => criterion
        max_values, max_indices = outputs.max(dim=2)#_, predicted = torch.max(outputs.data, 1)
        predicted = max_indices
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

T_accuracy = 100 * correct / total
print(f'Load Epoch: {last_epoch},', 'Test_Loss: {:.2f}'.format(tes_loss.item()),"Test_accuracy: {:.2f}%".format(T_accuracy))
# 打开文件以写入训练日志  
with open(log_file, "a") as f:  
# 将损失值写入文件  f.write(f"Epoch {epoch+1}, Loss: {loss.item():.4f}\n") 
    f.write(f'Load Epoch: {last_epoch}, Test_Loss: {tes_loss.item():.2f}, Test_accuracy: {T_accuracy:.2f}%\n')