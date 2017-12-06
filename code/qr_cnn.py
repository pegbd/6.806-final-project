import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import preprocessing

# cnn model


# class Conv_Net(nn.Module):
# 	"""
# 	"""
# 	def __init__(self, in_channels, output_dim):
# 		super(Net, self).__init__()
# 		Lin1 = input_dim
# 		kernel_size1 = 5
# 		padding1 = 4
# 		dialation1 = 1
# 		Lout1=floor((Lin1+2*padding1−dilation1*(kernel_size_1−1)−1)/stride+1)
# 		self.conv1 = nn.Conv1d(1, 1)


class CNNTrainer:
	"""
	"""
	def __init__(self, n_emb_channels, l_embs):
		self.n_emb_channels = n_emb_channels
		self.l_embs = l_embs

	def setup(self):
		

