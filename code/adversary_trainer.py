from __future__ import print_function
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from pre_conv import PreConv
import random

from eval_cnn import Evaluation
import time

from qr_cnn import CNN_Net, CNN_Model, CNN_Evaluator