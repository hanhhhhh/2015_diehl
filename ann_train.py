import torch
import numpy as np
import random
from torch.backends import cudnn
from ann_network import ANN
from torch import nn
from __init__ import *
from collections import OrderedDict
from copy import deepcopy
from utils import accuracy

seed=0
torch.manual_seed(seed) # 为CPU设置随机种子
torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
cudnn.benchmark = False
cudnn.deterministic = True

learning_rate=1e-3

epochs=10

ann=ANN()
print(ann)
ann=torch.nn.DataParallel(ann,device_ids)  #指定要用到的卡
ann=ann.cuda()  #将模型加载到主卡

optimizer=torch.optim.Adam(ann.parameters(),lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=epochs)
loss_fn=nn.CrossEntropyLoss()
Train=True

params_dict=OrderedDict(learning_rate=learning_rate,batch_size=batch_size,epochs=epochs,loss_fn=loss_fn,optimizer=optimizer)
if Train:
    best_acc=0
    ann.train()
    loss_list=[]
    train_acc_list=[]
    test_acc_list=[]
    for epoch in range(epochs):
        running_loss=0
        running_data=0
        for i,(batch_x,batch_y) in enumerate(train_dataloader):
            batch_x=batch_x.cuda()
            batch_y=batch_y.cuda()
            output=ann(batch_x)
            loss=loss_fn(output,batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()   #Tensor.item() to get a Python number from a tensor containing a single value:
            running_data+=batch_x.shape[0]
        loss_item=running_loss/running_data
        loss_list.append(loss_item)
        train_acc=accuracy(ann,train_dataloader)
        test_acc=accuracy(ann,test_dataloader)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        scheduler.step()
        print("Epoch: %3d | loss: %.4f | train accuracy: %.4f | test accuracy: %.4f" % (
        epoch, loss_item, train_acc,test_acc))
        if best_acc<test_acc:
            best_acc=test_acc
            params_dict.update(model=deepcopy(ann.module.state_dict()), accuracy=best_acc, loss=loss_list, train_acc=train_acc_list,
                              test_acc=test_acc_list)
        else:
            params_dict.update(loss=loss_list, train_acc=train_acc_list,
                              test_acc=test_acc_list)
        torch.save(params_dict,params_file)
if not Train:
    # 单GPU
    ann=ANN().cuda()
    ann.load_state_dict(torch.load(params_file)["model"])
    test_acc=accuracy(ann,test_dataloader)
    print("test accuracy: %.4f" %(test_acc))


