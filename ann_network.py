from torch import nn
from collections import OrderedDict
class ANN(nn.Module):
 def __init__(self,bias=False):
  super(ANN,self).__init__()
  self.feature=nn.Sequential(
   nn.Conv2d(1,12,kernel_size=5,stride=1,padding=2,bias=bias),
   nn.ReLU(),
   nn.AvgPool2d(kernel_size=2,stride=2),
   nn.Conv2d(12,64,kernel_size=5,stride=1,padding=2,bias=bias),
   nn.ReLU(),
   nn.AvgPool2d(kernel_size=2, stride=2),
  )
  self.classifier=nn.Linear(7*7*64,10,bias=bias)
  self.module2name=self.get_information()

 def forward(self,input):
  output=self.feature(input)
  output=output.view(input.shape[0],-1)
  output=self.classifier(output)
  return output
 def get_information(self):
  module2name=OrderedDict()
  for name,module in self.named_modules():
   if isinstance(module,(nn.Conv2d,nn.Linear)):
    module2name[module]=name
  return module2name
