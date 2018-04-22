import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn

m = nn.Softmax()
input = Variable(torch.randn(2)).int()
print(input.data[0])
