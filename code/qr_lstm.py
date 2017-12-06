import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import preprocessing

####################################################################
########################## Custom Classes ##########################
####################################################################

class QR_Maximum_Margin_Loss(torch.autograd.Function):

    def forward(self, q, positive, negatives):
        del_value = .01
        s_positive = F.cosine_similarity(q, positive, 0)
        return torch.max(torch.stack([F.cosine_similarity(q, n, 0) - s_positive + del_value for n in negatives]))


class LSTM_Cell(nn.Module):
    """
    An individual LSTM cell
    """
    def __init__(self, input_dim, hidden_dim, output_dim, use_bias=False):
        super(LSTM_Cell, self).__init__()

        self.use_bias = use_bias # for later, perhaps?
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

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

    def forward_step(self, x_t, h_t_1, c_t_1):

        """
        Computes one forward step (within a larger sequence)

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

    def forward(self, X):

        """
        Computes forward_step over the entire sequence X, and returns the averages of
        the hidden and visible states

        inputs:

        X = entire matrix sequence of word vectors

        outputs:

        h_avg = the average over the entire sequence of hiddens states
        c_avg = the average over the entire sequence of visible states
        """

        # initialize hidden and visible states at t = 0
        h_0 = preprocessing.to_float_variable([0.0 for i in range(self.hidden_dim)])
        c_0 = preprocessing.to_float_variable([0.0 for i in range(self.hidden_dim)])

        h = h_0
        c = c_0

        h_prev = h_0
        c_prev = c_0

        for i in range(len(X)):
            h_t, c_t = self.forward_step(X[i], h_prev, c_prev)

            h += h_t
            c += c_t
            h_prev = h_t
            c_prev = c_t

        h_avg = h/len(X)
        c_avg = c/len(X)

        return (h_avg, c_avg)

####################################################################
############################# TRAINING #############################
####################################################################

# Network Setup
torch.manual_seed(1)
INPUT_SIZE = 200
HIDDEN_SIZES = [INPUT_SIZE/4, INPUT_SIZE/2, INPUT_SIZE, 2*INPUT_SIZE]
OUTPUT_SIZE = 2
LEARNING_RATES = [.00001, .001, .1, 10]
L2_NORMS = [.00001, .001, 10]
NUM_ITERATIONS = 1
BATCH_SIZE = 64

# Work Space
words = preprocessing.word_to_vector
questions = preprocessing.id_to_question
candidates = preprocessing.question_to_candidates

training_batches = preprocessing.split_into_batches(candidates.keys(), BATCH_SIZE)

# Initialize Network and Optimizer
lstm = LSTM_Cell(input_dim=INPUT_SIZE, hidden_dim=HIDDEN_SIZES[1], output_dim=OUTPUT_SIZE)
optimizer = optim.Adam(params=lstm.parameters(), lr=LEARNING_RATES[0], weight_decay=L2_NORMS[0])

for batch in training_batches:
    for i in range(NUM_ITERATIONS):
        batch_loss = 0
        for qr in batch:

            # get the title and body matrices for the question, the positive and negative candidates
            q = questions[qr]
            pos = [questions[p] for p in candidates[qr][0]]
            neg = [questions[n] for n in candidates[qr][1]]

            # model representation of question q
            q_title_avg, _ = lstm(q[0])
            q_body_avg, _ = lstm(q[1])
            q_avg = (q_title_avg + q_body_avg) / 2

            # model representation of positive candidates
            pos_avgs = []
            for pos_cand in pos:
                p_title_avg, _ = lstm(pos_cand[0])
                p_body_avg, _ = lstm(pos_cand[1])
                p_avg = (p_title_avg + p_body_avg) / 2

                pos_avgs.append(p_avg)

            # model representation of negative candidates
            neg_avgs = []
            for neg_cand in neg:
                n_title_avg, _ = lstm(neg_cand[0])
                n_body_avg, _ = lstm(neg_cand[1])
                n_avg = (n_title_avg + n_body_avg) / 2

                neg_avgs.append(n_avg)

            mml = QR_Maximum_Margin_Loss()

            negatives = torch.stack(neg_avgs)
            loss = mml(q_avg, pos_avgs[0], negatives)
            print("loss ", loss)
            batch_loss += loss

            # back propogate with gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print ("batch loss = ", batch_loss)
        print ("------------------------")

lstm.save_state_dict('mytraininglstm.pt')
