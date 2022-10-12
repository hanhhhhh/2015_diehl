import torch
from __init__ import *
from ann_network import ANN
from max_norm import data_based_normalization,model_based_normalization,model_based_normalization
from utils import accuracy
ann=ANN().cuda()
ann.load_state_dict(torch.load(params_file)["model"])
data_based=True
max_norm=True
print("accuracy before max_normalization %.8f" % accuracy(ann,test_dataloader))
if max_norm:
    ann=model_based_normalization(ann)
    # ann=data_based_normalization(ann,data_based,train_dataloader)
print("accuracy after max_normalization %.8f"  % accuracy(ann,test_dataloader))
