import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import preprocessing
import time

####################################################################
########################## Custom Classes ##########################
####################################################################

# 1: Custom Maximum Margin Loss class
class QR_Maximum_Margin_Loss(nn.Module):
    def __init__(self):
        super(QR_Maximum_Margin_Loss, self).__init__()

    def forward(self, q, positive, negatives, del_value=.01):
        # ctx.save_for_backward(q, positive, negatives, del_value)
        return torch.max(torch.stack([F.cosine_similarity(q, n, 0) - F.cosine_similarity(q, positive, 0) + del_value for n in negatives]))


# 2: Custom LSTM Cell class
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


# 3. Model wrapper for built-in Bidirectional LSTM_Cell
class BiDiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(BiDiLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # self.decoder = nn.Linear(2*hidden_dim, output_dim)

        # initialize the wrapped bidirectional network
        self.net = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)

        # initialize and store parameter weights
        self.params = {}
        for name, param in self.net.named_parameters():
              nn.init.normal(param)
              self.params[name] = param

    def forward(self, X):

        data = self.net(X)

        return data

####################################################################
############################# TRAINING #############################
####################################################################


###############################
######## Network Set Up #######
###############################

torch.manual_seed(1)
INPUT_SIZE = 200
HIDDEN_SIZES = [INPUT_SIZE/4, INPUT_SIZE/2, INPUT_SIZE, 2*INPUT_SIZE]
OUTPUT_SIZE = 2
LEARNING_RATES = [.00001, .001, .1, 10]
L2_NORMS = [.00000000000001, .001, 10]
NUM_ITERATIONS = 1
BATCH_SIZE = 16
NUM_LAYERS = 1
DEL = .0001
EPOCHS = 4

# Work Space
words = preprocessing.word_to_vector
questions = preprocessing.id_to_question
candidates = preprocessing.question_to_candidates

training_batches = preprocessing.split_into_batches(candidates.keys(), BATCH_SIZE)

# Initialize Network and Optimizer
lstm = BiDiLSTM(input_dim=INPUT_SIZE, hidden_dim=HIDDEN_SIZES[1], num_layers=NUM_LAYERS, output_dim=OUTPUT_SIZE)
optimizer = optim.SGD(params=lstm.parameters(), lr=LEARNING_RATES[1], weight_decay=L2_NORMS[1])

# function wrapper for training
def train_model_epoch():

    # initialize epoch loss
    epoch_loss = 0

    print("starting training")
    for i in range(len(training_batches)):

        print("\n")
        print(" TRAINING BATCH:")
        print(str(i + 1) + '/' + str(len(training_batches)))
        print("\n")

        batch = training_batches[i]

        # intialize batch timer
        batch_time_start = time.time()

        ###############################
        ## Initial Batch Data Set Up ##
        ###############################

        # the word matrices for each sequence in the entire batch
        batch_title_data = []
        batch_body_data = []

        # the ordered, true lengths of each sequence before padding
        batch_title_lengths = []
        batch_body_lengths = []

        format_start = time.time()
        for qr in batch:

            # question of interest
            q = questions[qr]

            batch_title_data.append(preprocessing.sentence_to_embeddings(q[0]))
            batch_body_data.append(preprocessing.sentence_to_embeddings(q[1]))
            batch_title_lengths.append(1.0 / min(preprocessing.MAX_SEQUENCE_LENGTH, len(q[0])))
            batch_body_lengths.append(1.0 / min(preprocessing.MAX_SEQUENCE_LENGTH, len(q[1])))

            # positive example
            pos = questions[candidates[qr][0][0]]

            batch_title_data.append(preprocessing.sentence_to_embeddings(pos[0]))
            batch_body_data.append(preprocessing.sentence_to_embeddings(pos[1]))
            batch_title_lengths.append(1.0 / min(preprocessing.MAX_SEQUENCE_LENGTH, len(pos[0])))
            batch_body_lengths.append(1.0 / min(preprocessing.MAX_SEQUENCE_LENGTH, len(pos[1])))

            # negative examples
            for n in candidates[qr][1]:
                neg = questions[n]

                batch_title_data.append(preprocessing.sentence_to_embeddings(neg[0]))
                batch_body_data.append(preprocessing.sentence_to_embeddings(neg[1]))
                batch_title_lengths.append(1.0 / min(preprocessing.MAX_SEQUENCE_LENGTH, len(pos[0])))
                batch_body_lengths.append(1.0 / min(preprocessing.MAX_SEQUENCE_LENGTH, len(pos[1])))

        print("formatting took ", time.time() - format_start)

        # convert batch data and lengths to Variables
        batch_title_data = preprocessing.to_float_variable(batch_title_data)
        batch_body_data = preprocessing.to_float_variable(batch_body_data)
        batch_title_lengths = preprocessing.to_float_variable(batch_title_lengths)
        batch_body_lengths = preprocessing.to_float_variable(batch_body_lengths)


        #################################
        ## Run data through LSTM Model ##
        #################################

        forward_start = time.time()
        title_states, title_out = lstm(batch_title_data)
        print ("the title forward lstm took ", time.time() - forward_start)

        forward_start = time.time()
        body_states, body_out = lstm(batch_body_data)
        print ("the body forward lstm took ", time.time() - forward_start)

        ##########################################
        ## Re-arrange Data For Loss Calculation ##
        ##########################################

        # mean pooling of the hidden states of each question's title and body sequences
        title_states = torch.sum(title_states, dim=1, keepdim=False)
        averaged_title_states = title_states * batch_title_lengths.repeat(title_states.size(dim=1), 1).t()

        body_states = torch.sum(body_states, dim=1, keepdim = False)
        averaged_body_states = body_states * batch_body_lengths.repeat(body_states.size(dim=1), 1).t()

        # take the average between the title and body representations for the final representation
        final_question_reps = (averaged_title_states + averaged_body_states).div(2)

        # separate out each training instance in the batch
        training_instances = torch.chunk(final_question_reps, len(batch))


        ###############################################
        ## Calculate Loss for Each Training Instance ##
        ###############################################

        print(" ###### Calculating the loss ######")

        loss_data = [F.cosine_similarity(instance[1:], instance[0] , -1) for instance in training_instances]
        loss_data = torch.stack(loss_data, 1).t()

        target_data = [0 for inst in range(len(training_instances))]
        target_data = preprocessing.to_long_variable(target_data)

        # pass in loss data into loss function
        mml = nn.MultiMarginLoss()
        loss = mml(loss_data, target_data)

        print("batch loss is", loss)

        # back prop and SGD
        back_time_start = time.time()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("backpropogation took ", time.time() - back_time_start)

        # add batch loss to total epoch loss
        epoch_loss += loss.data[0]

        print("this batch took ", time.time() - batch_time_start)

    return epoch_loss

def train_model():
    for epoch in range(EPOCHS):
        loss_this_epoch = train_model_epoch()
        print("Epoch " + str(epoch) + " loss:" + str(loss_this_epoch))


if __name__ == "__main__":
    train_model()
    torch.save(lstm, 'qr_lstm_model.pt')
