import torch  
import torchvision.models as models
from torch.utils.data import DataLoader
import json 
import os,time

from _1_3_0_Load_dataset_Json_batchPadding_EcapaSp_w import CougherDataset,collate_fn


#seed
import random
import numpy as np

seed = 20986
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
picknum = 166
pplnum =86
typ = 'noagu_'+str(pplnum)
lllr ='lra0001'
win_s=18
hop_s=8
win = 'w'+str(win_s)+'[' + str(hop_s) +']'

'''
1. import Resnet50
'''
#resnet = models.resnet50(pretrained=False)
resnet = models.resnet50(weights=None)

resnet.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)

#print(resnet)
#first_conv_layer = resnet.conv1  
#print(first_conv_layer.weight)
'''
1.1 import label
'''
lb_dic = '../aug_data/Label_index'+str(picknum)+'_'+typ+'.json'
with open(lb_dic, 'r') as d:  
    labels_dic = json.load(d) 
 
'''
2.
'''
batch_s = 16  
lr_start = 0.001  
#lr_end = 0.001
num_epochs = 300  

num_classes = pplnum  
resnet.fc = torch.nn.Linear(resnet.fc.in_features, num_classes)
'''
3.
'''
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet.parameters(), lr=lr_start)

#=====3.1===== #

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9772372209558107)  

train_dt = CougherDataset('../aug_data/train_Cougher_pick'+str(picknum)+'_'+ typ +'.json',lb_dic,win=win_s*0.001,hops=hop_s*0.001)
valid_dt = CougherDataset('../aug_data/valid_Cougher_pick'+str(picknum)+'_'+ typ +'.json',lb_dic,win=win_s*0.001,hops=hop_s*0.001) 
test_dt = CougherDataset('../aug_data/test_Cougher_pick'+str(picknum)+'_'+ typ +'.json',lb_dic,win=win_s*0.001,hops=hop_s*0.001)


#train_ld = DataLoader(train_dt, batch_size=16, collate_fn=collate_fn, shuffle=True,drop_last=True)
#valid_ld = DataLoader(valid_dt, batch_size=16, collate_fn=collate_fn, shuffle=True,drop_last=True)
#test_ld = DataLoader(test_dt, batch_size=16, collate_fn=collate_fn, shuffle=True,drop_last=True)
train_ld = DataLoader(train_dt, batch_size=batch_s, collate_fn=collate_fn, shuffle=True)
valid_ld = DataLoader(valid_dt, batch_size=batch_s, collate_fn=collate_fn, shuffle=True)
test_ld = DataLoader(test_dt, batch_size=batch_s, collate_fn=collate_fn, shuffle=True) #drop_last=True

'''
7.
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet.to(device)  
 
# train
best_acc = 0.0 

results_fd ="./results/"+str(picknum)+typ+'_'+win+'_'+lllr
log_file = results_fd+"/train_log.txt"   

if os.path.isfile(log_file):  
    os.remove(log_file) 
    
if os.path.exists(results_fd) is False :
    os.mkdir(results_fd)
    
    
for epoch in range(num_epochs):
    for inputs, labels in train_ld:
        inputs = inputs.to(device)
        labels = labels.to(device)
        #print(labels)
        #print(inputs)

        outputs = resnet(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()    
    with open(log_file, "a") as f:
        f.write(f'lr: {scheduler.get_last_lr()}')
    print(scheduler.get_last_lr()) #print last learning rate.

    with torch.no_grad():  
        correct = 0  
        total = 0  
        for inputs, labels in valid_ld:  
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = resnet(inputs)  
            val_loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)  
            total += labels.size(0)  
            correct += (predicted == labels).sum().item()  
        accuracy = 100 * correct / total  
 
        with open(log_file, "a") as f:  
         
            f.write(f'Epoch {epoch+1}, Train Loss:{loss.item():.2f}, Valid_Loss:{val_loss.item():.2f}, Valid_Accuracy:{accuracy:.2f}% \n')
        print(f'Epoch: {epoch+1}, Train Loss: {loss.item():.2f}, Valid_Loss: {val_loss.item():.2f}, Valid_Accuracy: {accuracy:.2f}%')  

        if accuracy >= best_acc:
            best_acc = accuracy
            best_model_wts = resnet.state_dict()
            last_epoch = epoch +1
        

torch.save(best_model_wts, results_fd+'/best_model_{}.pth'.format(time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime())))
torch.save(best_model_wts, results_fd+'/best_model.pth')
'''        
8.
'''
resnet.eval() 

best_model_path = results_fd+'/best_model.pth'
resnet.load_state_dict(torch.load(best_model_path))

with torch.no_grad():
    correct = 0
    total = 0

    for inputs, labels in test_ld:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = resnet(inputs)
        tes_loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

T_accuracy = 100 * correct / total
print(f'Load Epoch: {last_epoch},', 'Test_Loss: {:.2f}'.format(tes_loss.item()),"Test_accuracy: {:.2f}%".format(T_accuracy))

with open(log_file, "a") as f:  
    f.write(f'Load Epoch: {last_epoch}, Test_Loss: {tes_loss.item():.2f}, Test_accuracy: {T_accuracy:.2f}%\n')
