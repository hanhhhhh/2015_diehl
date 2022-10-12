from collections import OrderedDict,defaultdict
from copy import deepcopy
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
def data_based_normalization(model,data_based,dataloader):
    # register hook
    features=defaultdict(list) #dict中value的类型是list
    def hook(module,inp,out):
        features[model.module2name[module]].append(F.relu(out).cpu())
    handles=[]
    for name, module in model.named_modules():
        if isinstance(module,(nn.Conv2d,nn.Linear)):
            handles.append(module.register_forward_hook(hook=hook))

    # input
    with torch.no_grad():
        model.eval()
        with tqdm(dataloader) as loader_bar:
            for (batch_x,batch_y) in loader_bar:
                batch_x=batch_x.cuda()
                _=model(batch_x)

    # clear hook
    for handle in handles:
        handle.remove()

    #data based normalization
    pre_factor=1
    for name,module in model.named_modules():
        if isinstance(module,(nn.Conv2d,nn.Linear)):
            max_weight=torch.max(module.weight.data)
            max_act=np.max(np.vstack(features[model.module2name[module]]))
            post_factor=max(max_weight,max_act)
            scale_factor=post_factor/pre_factor
            module.weight.data=module.weight.data/scale_factor
            print(name,scale_factor)
            pre_factor=post_factor
    return model


def model_based_normalization_error(model):
    """
    进行 model_based_normalization，将一个全为1的样本输入，记录 每层Conv 和 Linear 层的输出最大值，来缩放权重
    但是这种方法有问题：随着网络的加深，最大值会越来越大，导致缩放后的权重越来越小
    :param model:要缩放的
    :return:
    """
    temp_model = deepcopy(model)
    # register the hook to save max positive activation
    features=OrderedDict()
    def hook(module,inp,out):
        features[temp_model.module2name[module]]=out

    handles=[]
    for name,module in temp_model.named_modules():
        if isinstance(module,(nn.Conv2d,nn.Linear)):
            handles.append(module.register_forward_hook(hook=hook))
            module.weight.data=torch.where(module.weight.data>0,module.weight.data,torch.tensor(0).float().cuda())
    #input x
    x = torch.ones(1, 1, 28, 28).cuda()

    with torch.no_grad():
        temp_model.eval()
        _=temp_model(x)

    #remove handle
    for handle in handles:
        handle.remove()

    #calculate
    scale_factors=OrderedDict()
    for name,module in temp_model.named_modules():
        if isinstance(module,(nn.Conv2d,nn.Linear)):
            scale_factors[name]=torch.max(features[name]).item()
            print(scale_factors[name])
    #scale weight
    for name,module in model.named_modules():
        if isinstance(module,(nn.Conv2d,nn.Linear)):
            module.weight.data/=scale_factors[name]
    return model


def model_based_normalization(model):
    scale_factors = OrderedDict()
    for name,layer in model.named_modules():
        if isinstance(layer,(nn.Conv2d,nn.Linear)):
            weight = torch.maximum(layer.weight.detach(), torch.zeros(1, device=layer.weight.device))
            scale_factors[name] = weight.sum([i for i in range(1, weight.dim())]).max().item()
            # print(scale_factors[name])
            weight.data=weight.data/scale_factors[name]
            #12个1*5*5的feature map，每个feature map权重求和，再求所有feature map中最大的值，也就得到了最大正激活值之和
    return model

