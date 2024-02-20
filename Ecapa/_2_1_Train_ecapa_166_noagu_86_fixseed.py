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
from _1_3_3_Load_dataset_Json_batchPadding_EcapaSp import CougherDataset,collate_fn


import random
import numpy as np

seed = 20986 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed) 
torch.cuda.manual_seed_all(seed)


picknum = 166  #total number of coughing person
pplnum = 86  #pick  80 coughing persons
typ = 'noagu_86'
lllr ='lraw25[10]_fixsed_'

n_mels =23  # F-bank dim
lin_neur = 192 #ecapa output neur number
'''
1. import Ecapa-tdnn
'''
from _0_3_ECAPA_TDNN_model import ECAPA_TDNN  #,Classifier
'''
ECAPA_TDNN
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

classifier 
        input_size,
        device="cpu",
    lin_blocks : int
        Number of linear layers. default 0
    lin_neurons : int
        Number of neurons in linear layers. default 192
    out_neurons : int
        Number of classes.
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Initial ecapa model  #Tensor of shape (batch, time, channel).
ecapa = ECAPA_TDNN(input_size= n_mels
,channels          = [1024, 1024, 1024, 1024, 3072]
,kernel_sizes      = [5, 3, 3, 3, 1]
,dilations         = [1, 2, 3, 4, 1]
,groups            = [1, 1, 1, 1, 1]
,attention_channels= 128
,lin_neurons       = lin_neur
,out_neurons       = pplnum)
#Initial Classifier for ecapa model
#classify = Classifier( lin_neurons=lin_neur, input_size=lin_neur,out_neurons=picknum,device="cuda") #lin_blocks=0,


'''
1.1 Import label
'''
lb_dic = '../aug_data/Label_index'+str(picknum)+'_'+typ+'.json'
with open(lb_dic, 'r') as d:  
    labels_dic = json.load(d) 
 
'''
2.
'''
batch_siz = 16  
learning_rate = 0.001  
num_epochs = 300  


'''
3.
'''
#criterion = torch.nn.CrossEntropyLoss()
log_prob = LogSoftmaxWrapper(AdditiveAngularMargin(margin=0.2, scale=30))
#loss = log_prob(outputs, targets)


optimizer = torch.optim.Adam(ecapa.parameters(), lr=learning_rate, weight_decay = 2e-5)

#gamma = 0.001**(1/300)
#scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma)
scheduler = CyclicLRScheduler(base_lr=0.00000001, max_lr=0.001, step_size=65000)


#torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up=2000, step_size_down=None, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=- 1, verbose=False)
#scheduler = CyclicLR(optimizer, base_lr=0.00000001, max_lr=0.001, step_size_up=65000, step_size_down=None, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=- 1, verbose=False)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)

train_dt = CougherDataset('../aug_data/train_Cougher_pick'+str(picknum)+'_'+ typ +'.json',lb_dic)
valid_dt = CougherDataset('../aug_data/valid_Cougher_pick'+str(picknum)+'_'+ typ +'.json',lb_dic) 
test_dt = CougherDataset('../aug_data/test_Cougher_pick'+str(picknum)+'_'+ typ +'.json',lb_dic)

#train_ld = DataLoader(train_dt, batch_size=16, collate_fn=collate_fn, shuffle=True,drop_last=True)
#valid_ld = DataLoader(valid_dt, batch_size=16, collate_fn=collate_fn, shuffle=True,drop_last=True)
#test_ld = DataLoader(test_dt, batch_size=16, collate_fn=collate_fn, shuffle=True,drop_last=True)
train_ld = DataLoader(train_dt, batch_size=batch_siz, collate_fn=collate_fn, shuffle=True)
valid_ld = DataLoader(valid_dt, batch_size=batch_siz, collate_fn=collate_fn, shuffle=True)
test_ld = DataLoader(test_dt, batch_size=batch_siz, collate_fn=collate_fn, shuffle=True)

'''
7.
'''
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ecapa.to(device)  
 
# train 
best_acc = 0.0 #initial
smaller_distance = 10

 
#    os.remove(file_path) 
log_file = "./results/"+str(picknum)+typ+lllr+"/train_log.txt"   

if os.path.isfile(log_file):  
    os.remove(log_file) 
    
if os.path.exists('./results/') is False :
    os.mkdir('./results/')
    
if os.path.exists('./results/'+str(picknum)+typ+lllr) is False :
    os.mkdir('./results/'+str(picknum)+typ+lllr)
    

    
for epoch in range(num_epochs):
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
    scheduler.on_batch_end(optimizer) #scheduler.step()    
    with open(log_file, "a") as f:
        f.write(f'lr: {optimizer.param_groups[0]["lr"]}')  #scheduler.get_last_lr()
    print(optimizer.param_groups[0]["lr"]) #print(scheduler.get_last_lr()) #  print last learning rate.

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
        distance = abs(loss.item()-val_loss.item())
        accuracy = 100 * correct / total  

        with open(log_file, "a") as f:  
            f.write(f'Epoch {epoch+1}, Train Loss:{loss.item():.2f}, Valid_Loss:{val_loss.item():.2f}, Valid_Accuracy:{accuracy:.2f}% \n')
        print(f'Epoch: {epoch+1}, Train Loss: {loss.item():.2f}, Valid_Loss: {val_loss.item():.2f}, Valid_Accuracy: {accuracy:.2f}%')  

        if accuracy == best_acc and distance < smaller_distance:
            smaller_distance = distance
            best_model_wts = ecapa.state_dict()
            last_epoch = epoch +1
        if accuracy > best_acc:
            best_acc = accuracy
            best_model_wts = ecapa.state_dict()
            smaller_distance = distance            
            last_epoch = epoch +1
        
# save the best weights
torch.save(best_model_wts, "./results/"+str(picknum)+typ+lllr+'/best_model_{}.pth'.format(time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime())))
torch.save(best_model_wts, "./results/"+str(picknum)+typ+lllr+'/best_model.pth')
'''        
8.
'''
ecapa.eval()  

best_model_path = './results/'+str(picknum)+typ+lllr+'/best_model.pth'
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
