from __future__ import print_function
import torch
from torch import nn
from torch.nn import functional as F


class Discriminator(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(Discriminator, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size

		self.fc1 = nn.Linear(input_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, 1)
		self.softmax = nn.Softmax()

	def forward(self, x):
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = self.softmax(x)
		return x