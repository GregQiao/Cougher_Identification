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

#seed
import random
import numpy as np

seed = 20986
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
   #cudnn.benchmark = True       
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

picknum = 166  #total number of coughing persons
pplnum = 166  # pick 166 coughing persons
typ = 'noagu'
lllr ='lra[122331][73][512]A_M_eca_fixsed_128_scl_c16'
win_s=18
hop_s=8
win = 'w'+str(win_s)+'[' + str(hop_s) +']'

n_mels =23  # F-bank dim
lin_neur = 192 #ecapa lin_neur
'''
1. import Ecapa-tdnn
'''
from _0_3_3_ECAPA_TDNN_A_M_ECA_scl_128_model import ECAPA_TDNN  #,Classifier
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
1.1 import label
'''
lb_dic = '../aug_data/Label_index'+str(picknum)+'_'+typ +'.json'
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

scheduler = CyclicLRScheduler(base_lr=0.00000001, max_lr=0.001, step_size=65000)

train_dt = CougherDataset('../aug_data/train_Cougher_pick'+str(picknum)+'_'+ typ +'.json',lb_dic,win=win_s*0.001,hops=hop_s*0.001)
valid_dt = CougherDataset('../aug_data/valid_Cougher_pick'+str(picknum)+'_'+ typ +'.json',lb_dic,win=win_s*0.001,hops=hop_s*0.001) 
test_dt = CougherDataset('../aug_data/test_Cougher_pick'+str(picknum)+'_'+ typ +'.json',lb_dic,win=win_s*0.001,hops=hop_s*0.001)

#train_ld = DataLoader(train_dt, batch_size=16, collate_fn=collate_fn, shuffle=True,drop_last=True)
#valid_ld = DataLoader(valid_dt, batch_size=16, collate_fn=collate_fn, shuffle=True,drop_last=True)
#test_ld = DataLoader(test_dt, batch_size=16, collate_fn=collate_fn, shuffle=True,drop_last=True)
train_ld = DataLoader(train_dt, batch_size=batch_siz, collate_fn=collate_fn, shuffle=True)
valid_ld = DataLoader(valid_dt, batch_size=batch_siz, collate_fn=collate_fn, shuffle=True)
test_ld = DataLoader(test_dt, batch_size=batch_siz, collate_fn=collate_fn, shuffle=True)

'''
7. 
'''
ecapa.to(device)  
 
# train
best_acc = 0.0 

log_file = "./results/"+str(picknum)+typ+lllr+win+"/train_log.txt" 

if os.path.isfile(log_file):  
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
   
    scheduler.on_batch_end(optimizer) #scheduler.step()    
    with open(log_file, "a") as f:
        f.write(f'lr: {optimizer.param_groups[0]["lr"]}\n')  #scheduler.get_last_lr()
    print(optimizer.param_groups[0]["lr"]) #print(scheduler.get_last_lr()) #  print last learning rate.
 
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

        with open(log_file, "a") as f:  
        
            f.write(f'Epoch {epoch+1}, Train Loss:{loss.item():.2f}, Valid_Loss:{val_loss.item():.2f}, Valid_Accuracy:{accuracy:.2f}% \n')
        print(f'Epoch: {epoch+1}, Train Loss: {loss.item():.2f}, Valid_Loss: {val_loss.item():.2f}, Valid_Accuracy: {accuracy:.2f}%')  
    
        if accuracy >= best_acc:
            best_acc = accuracy
            best_model_wts = ecapa.state_dict()
            last_epoch = epoch +1
        

torch.save(best_model_wts, "./results/"+str(picknum)+typ+lllr+win+'/best_model_{}.pth'.format(time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime())))
torch.save(best_model_wts, "./results/"+str(picknum)+typ+lllr+win+'/best_model.pth')
'''        
8.
'''
ecapa.eval() 

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

with open(log_file, "a") as f:  

    f.write(f'Load Epoch: {last_epoch}, Test_Loss: {tes_loss.item():.2f}, Test_accuracy: {T_accuracy:.2f}%\n')
