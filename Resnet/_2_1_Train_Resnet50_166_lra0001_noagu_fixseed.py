import torch  
import torchvision.models as models
from torch.utils.data import DataLoader
import json 
import os,time

from _1_3_3_Load_dataset_Json_batchPadding import CougherDataset,collate_fn

#设置实验随机种子
import random
import numpy as np
# 设置随机种子
seed = 20986
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

picknum = 166
typ = 'noagu'
lllr ='lra0001'
'''
1. 导出Resnet50模型
'''
#resnet = models.resnet50(pretrained=False)
resnet = models.resnet50(weights=None)
#输入数据确实是1通道，修改ResNet50模型的第一层，使其接受1通道的输入。通过修改第一层的卷积核大小
resnet.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)

#print(resnet)
#first_conv_layer = resnet.conv1  
#print(first_conv_layer.weight)
'''
1.1 导入标签
'''
lb_dic = '../aug_data/Label_index'+str(picknum)+'_'+typ+'.json'
with open(lb_dic, 'r') as d:  
    labels_dic = json.load(d) #音频文件信息字典
 
'''
2.更新全连接层。ResNet50模型原本是用于图像分类的，最后一层全连接层需要根据你的任务进行调整。这里假设你的任务是分类，你需要将最后一层的输出特征数更新为你的分类数量：
'''
batch_size = 32  
lr_start = 0.001  
#lr_end = 0.001
num_epochs = 300  # 假设训练300个epoch

num_classes = picknum  # 假设有10个类别
resnet.fc = torch.nn.Linear(resnet.fc.in_features, num_classes)
'''
3.定义损失函数和优化器。根据你的任务，选择相应的损失函数来计算损失值，并选择相应的优化器来更新模型的参数。这里假设你的任务是分类，使用交叉熵损失函数和随机梯度下降（SGD）优化器：
'''
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet.parameters(), lr=lr_start)

#=====3.1===== ExponentialLR#
# 现在我们来定义学习率调度器。
# 我们需要解决的问题是找到一个适当的gamma值，使得0.01乘以gamma的300次方等于0.001。
#gamma = 0.01**(1/300)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

##=====3.2===== LambdaLR#
## 定义一个lambda函数来计算学习率
#lr_lambda = lambda epoch: round(lr_start - (lr_start - lr_end) * min(epoch / num_epochs, 1.0),5)
## 创建一个LambdaLR调度器，将lr_lambda函数作为参数传递
#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
#=====3.3===== #
# 定义学习率调度器  
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9772372209558107)  


# 创建数据集对象 test_Cougher_pick8_agu.json
#lb = '../aug_data/Label_index8.json'

train_dt = CougherDataset('../aug_data/train_Cougher_pick'+str(picknum)+'_'+ typ +'.json',lb_dic)
valid_dt = CougherDataset('../aug_data/valid_Cougher_pick'+str(picknum)+'_'+ typ +'.json',lb_dic) 
test_dt = CougherDataset('../aug_data/test_Cougher_pick'+str(picknum)+'_'+ typ +'.json',lb_dic)

#可以使用torch.utils.data.DataLoader来创建一个数据加载器，用于批量加载和处理数据。示例代码如下：
#train_ld = DataLoader(train_dt, batch_size=16, collate_fn=collate_fn, shuffle=True,drop_last=True)
#valid_ld = DataLoader(valid_dt, batch_size=16, collate_fn=collate_fn, shuffle=True,drop_last=True)
#test_ld = DataLoader(test_dt, batch_size=16, collate_fn=collate_fn, shuffle=True,drop_last=True)
train_ld = DataLoader(train_dt, batch_size=16, collate_fn=collate_fn, shuffle=True)
valid_ld = DataLoader(valid_dt, batch_size=16, collate_fn=collate_fn, shuffle=True)
test_ld = DataLoader(test_dt, batch_size=16, collate_fn=collate_fn, shuffle=True)

'''
7. 训练模型。利用数据加载器、损失函数和优化器进行模型训练。循环遍历训练集中的每个批次并执行以下操作：
  - 将输入和标签加载到GPU（如果可用）上；
  - 前向传播计算模型的输出；
  - 计算损失值；
  - 反向传播和梯度更新；
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet.to(device)  # 将模型移动到GPU（如果可用）
 
# 训练模型  
best_acc = 0.0 #用于保存最高的准确率

# 定义保存训练日志的文件路径  
#    os.remove(file_path) 
log_file = "./results/"+str(picknum)+typ+lllr+"/train_log.txt"   
if os.path.exists('./results/') is False :
    os.mkdir('./results/')
if os.path.exists('./results/'+str(picknum)+typ+lllr) is False :
    os.mkdir('./results/'+str(picknum)+typ+lllr)
    
#删除就的日志文件
if os.path.isfile(log_file):  
    # 删除文件  
    os.remove(log_file) 
    
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
    # 在每个epoch结束后，你可以这样来更新你的learning rate
    scheduler.step()    
    with open(log_file, "a") as f:
        f.write(f'lr: {scheduler.get_last_lr()}')
    print(scheduler.get_last_lr()) # 打印上一次的学习率will print last learning rate.
    # 在验证集上评估模型性能  
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
        # 打开文件以写入训练日志  
        with open(log_file, "a") as f:  
         # 将损失值写入文件  f.write(f"Epoch {epoch+1}, Loss: {loss.item():.4f}\n") 
            f.write(f'Epoch {epoch+1}, Train Loss:{loss.item():.2f}, Valid_Loss:{val_loss.item():.2f}, Valid_Accuracy:{accuracy:.2f}% \n')
        print(f'Epoch: {epoch+1}, Train Loss: {loss.item():.2f}, Valid_Loss: {val_loss.item():.2f}, Valid_Accuracy: {accuracy:.2f}%')  
         # 保存最优模型
        if accuracy >= best_acc:
            best_acc = accuracy
            best_model_wts = resnet.state_dict()
            last_epoch = epoch +1
        
# 保存最优模型的权重参数
torch.save(best_model_wts, "./results/"+str(picknum)+typ+lllr+'/best_model_{}.pth'.format(time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime())))
torch.save(best_model_wts, "./results/"+str(picknum)+typ+lllr+'/best_model.pth')
'''        
8.在测试集上评估模型。使用验证集对模型进行评估，计算准确率、精确率、召回率等指标。循环遍历验证集中的每个批次并执行以下操作：
  - 将输入和标签加载到GPU（如果可用）上；
  - 前向传播计算模型的输出；
  - 计算评估指标；
'''
resnet.eval()  # 将模型设置为评估模式

# 加载最优模型
best_model_path = "./results/"+str(picknum)+typ+lllr+'/best_model.pth'
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
# 打开文件以写入训练日志  
with open(log_file, "a") as f:  
# 将损失值写入文件  f.write(f"Epoch {epoch+1}, Loss: {loss.item():.4f}\n") 
    f.write(f'Load Epoch: {last_epoch}, Test_Loss: {tes_loss.item():.2f}, Test_accuracy: {T_accuracy:.2f}%\n')