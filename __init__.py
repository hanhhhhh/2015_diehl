import os
import platform
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

is_linux=True if platform.system()=='Linux' else False
num_workers=20 if is_linux else 0
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2'  #当前可以被python环境程序检测到的显卡,第一个为主卡
device_ids=[0,1,2]
dataset_root=r'/data3/jnhan/data/'  # r:保持字符原始值
dataset_name='MNIST'
saved_models_root=r'/data3/jnhan/saved_models/2015_diehl'
if not os.path.isdir(saved_models_root):
    os.mkdir(saved_models_root)
params_file=os.path.join(saved_models_root,'_'.join([dataset_name,'params.pt']))
batch_size=128

transform_MNIST = transforms.Compose(
        [transforms.ToTensor()]
    )
train_data = datasets.MNIST(root=dataset_root,train=True,download=True,transform=transform_MNIST)
test_data = datasets.MNIST(root=dataset_root,train=False,download=True,transform=transform_MNIST)
train_dataloader = DataLoader(train_data,
                              batch_size=batch_size*len(device_ids),  ## 单卡batch size * 卡数,因为由主卡将数据读取并分配到device_ids张卡上
                              shuffle=True,
                              num_workers=num_workers)
test_dataloader = DataLoader(test_data,
                             batch_size=batch_size*len(device_ids),
                             shuffle=False,
                             num_workers=num_workers)
