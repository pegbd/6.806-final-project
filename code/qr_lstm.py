import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import preprocessing

# implementation 1, built in torch LSTM

# lstm = torch.nn.LSTM()

# implementation 2, custom model

class LSTM_Cell(nn.Module):
    """
    An individual LSTM cell
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()

        self.use_bias = use_bias # for later, perhaps?

        # input weights
        self.W_i = nn.Linear(input_dim, hidden_dim)
        self.W_f = nn.Linear(input_dim, hidden_dim)
        self.W_o = nn.Linear(input_dim, hidden_dim)
        self.W_z = nn.Linear(input_dim, hidden_dim)

        # hidden weights
        self.U_i = nn.Linear(hidden_dim, hidden_dim)
        self.U_f = nn.Linear(hidden_dim, hidden_dim)
        self.U_o = nn.Linear(hidden_dim, hidden_dim)
        self.U_z = nn.Linear(hidden_dim, hidden_dim)

        # output/decoder
        self.h2o = nn.Linear(hidden_dim, output_dim)

        def forward(self, x_t, h_t_1, c_t_1):

            """
            inputs:

            x_t = word t in the sentence
            h_t_1 = the hidden state at step t - 1
            c_t_1 = the visible state at step t - 1

            outputs:

            h_t = the current hidden state
            c_t = the current visible state

            """

            # input gate
            i = F.sigmoid(self.W_i(x_t) + self.U_i(h_t_1))

            # forget gate
            f = F.sigmoid(self.W_f(x_t) + self.U_f(h_t_1))

            # output gate
            o = F.sigmoid(self.W_o(x_t) + self.U_o(h_t_1))

            # activation
            z = F.tanh(self.W_z(x_t) + self.U_z(h_t_1))

            # calculate the next visible asnd hidden states, then return
            c_t = (i * z) + (f * c_t_1)
            h_t = o * F.tanh(c_t)

            return h_t, c_t

# Work Space
words = preprocessing.word_to_vector
questions = preprocessing.id_to_question

# Network Setup
torch.manual_seed(1)
INPUT_SIZE = 300
HIDDEN_SIZES = [INPUT_SIZE/4, INPUT_SIZE/2, INPUT_SIZE, 2*INPUT_SIZE]
OUTPUT_SIZE = 2
LEARNING_RATES = [.00001, .001, .1, 10]
L2_NORMS = [.00001, .001, 10]
NUM_ITERATIONS = 50

# Initialize Network and Optimizer
lstm = LSTM_Cell(INPUT_SIZE, HIDDEN_SIZES[0], OUTPUT_SIZE)
optimizer = optim.Adam(params=lstm.parameters(), lr=LEARNING_RATES[0], weight_decay=L2_NORMS[0])
