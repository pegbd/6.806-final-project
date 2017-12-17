import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import preprocessing
import time
import random
from pre_android import PreAndroid

pre_and = PreAndroid()

####################################################################
########################## Custom Classes ##########################
####################################################################

# 1. Model wrapper for built-in Bidirectional LSTM
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
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal(param)

    def forward(self, X):
        return self.net(X)

####################################################################
############################# TRAINING #############################
####################################################################

def get_mask(sequence, hidden_size):
    n = min(100, len(sequence))
    return list(np.transpose(np.array([[1 for i in range(n)] + [0 for i in range(preprocessing.MAX_SEQUENCE_LENGTH - n)] for j in range(2*hidden_size)])))

def mirror_sequence(sequence):
    return sequence + sequence.reverse()

###############################
######## Network Set Up #######
###############################

torch.manual_seed(1)
INPUT_SIZE = 300 # FOR ANDROID
HIDDEN_SIZES = [INPUT_SIZE/4, INPUT_SIZE/2, INPUT_SIZE, 2*INPUT_SIZE]
OUTPUT_SIZE = 2
LEARNING_RATES = [.00001, .001, .1, 1, 10]
L2_NORMS = [0, .00000000000001, .001, .1, 1, 10]
NUM_ITERATIONS = 1
BATCH_SIZE = 16
NUM_LAYERS = 1
DEL = .0001
EPOCHS = 4

# Work Space

# words = preprocessing.word_to_vector # UBUNTU
words = pre_and.word_to_vector # ANDROID

# Training on Ubuntu
questions = preprocessing.id_to_question
candidates = preprocessing.question_to_candidates

training_batches = preprocessing.split_into_batches(candidates.keys(), BATCH_SIZE)

# Initialize Network and Optimizer
# lstm = BiDiLSTM(input_dim=INPUT_SIZE, hidden_dim=HIDDEN_SIZES[2], num_layers=NUM_LAYERS, output_dim=OUTPUT_SIZE)
# mml = nn.MultiMarginLoss()
# optimizer = optim.Adam(params=lstm.parameters(), lr=LEARNING_RATES[1], weight_decay=L2_NORMS[2])

# Initialize Network and Optimizer
lstm = None
mml = None
optimizer = None

# function wrapper for training
def train_model_epoch(lstm, loss_func, optimizer):

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

        # masks for title and body
        batch_title_mask = []
        batch_body_mask = []

        format_start = time.time()

        total_batch_size = 0

        for qr in batch:

            total_batch_size += len(candidates[qr][0])

            for pos_cand in candidates[qr][0]:

                # question of interest
                q = questions[qr]

                # batch_title_data.append(preprocessing.sentence_to_embeddings(q[0])) # UBUNTU
                print q[0]
                batch_title_data.append(pre_and.list_to_vec(q[0])) # ANDROID

                batch_title_lengths.append(1.0 / min(preprocessing.MAX_SEQUENCE_LENGTH, len(q[0])))
                batch_title_mask.append(get_mask(q[0], HIDDEN_SIZES[2]))


                # batch_body_data.append(preprocessing.sentence_to_embeddings(q[1])) # UBUNTU
                batch_body_data.append(pre_and.list_to_vec(q[1]))

                batch_body_lengths.append(1.0 / min(preprocessing.MAX_SEQUENCE_LENGTH, len(q[1])))
                batch_body_mask.append(get_mask(q[1], HIDDEN_SIZES[2]))

                # positive example
                pos = questions[pos_cand]

                # batch_title_data.append(preprocessing.sentence_to_embeddings(pos[0])) # UBUNTU
                batch_title_data.append(pre_and.list_to_vec(pos[0])) # ANDROID

                batch_title_lengths.append(1.0 / min(preprocessing.MAX_SEQUENCE_LENGTH, len(pos[0])))
                batch_title_mask.append(get_mask(pos[0], HIDDEN_SIZES[2]))

                # batch_body_data.append(preprocessing.sentence_to_embeddings(pos[1])) # UBUNTU
                batch_body_data.append(pre_and.list_to_vec(pos[1])) # ANDROID

                batch_body_lengths.append(1.0 / min(preprocessing.MAX_SEQUENCE_LENGTH, len(pos[1])))
                batch_body_mask.append(get_mask(pos[1], HIDDEN_SIZES[2]))

                # negative examples
                neg_cands = random.sample(candidates[qr][1], preprocessing.NEGATIVE_CANDIDATE_SIZE)
                for n in neg_cands:
                    neg = questions[n]

                    # batch_title_data.append(preprocessing.sentence_to_embeddings(neg[0])) # UBUNTU
                    batch_title_data.append(pre_and.list_to_vec(neg[0])) #ANDROID

                    batch_title_lengths.append(1.0 / min(preprocessing.MAX_SEQUENCE_LENGTH, len(neg[0])))
                    batch_title_mask.append(get_mask(neg[0], HIDDEN_SIZES[2]))

                    # batch_body_data.append(preprocessing.sentence_to_embeddings(neg[1])) # UBUNTU
                    batch_body_data.append(pre_and.list_to_vec(neg[1]))

                    batch_body_lengths.append(1.0 / min(preprocessing.MAX_SEQUENCE_LENGTH, len(neg[1])))
                    batch_body_mask.append(get_mask(neg[1], HIDDEN_SIZES[2]))

        print("formatting took ", time.time() - format_start)

        # convert batch data and lengths to Variables
        batch_title_data = preprocessing.to_float_variable(batch_title_data)
        batch_body_data = preprocessing.to_float_variable(batch_body_data)
        batch_title_lengths = preprocessing.to_float_variable(batch_title_lengths)
        batch_body_lengths = preprocessing.to_float_variable(batch_body_lengths)
        batch_title_mask = preprocessing.to_float_variable(batch_title_mask)
        batch_body_mask = preprocessing.to_float_variable(batch_body_mask)


        #################################
        ## Run data through LSTM Model ##
        #################################

        # first! set zero grad... just in case.
        optimizer.zero_grad()

        forward_start = time.time()
        title_states, title_out = lstm(batch_title_data)
        print ("the title forward lstm took ", time.time() - forward_start)

        forward_start = time.time()
        body_states, body_out = lstm(batch_body_data)
        print ("the body forward lstm took ", time.time() - forward_start)

        ##########################################
        ## Re-arrange Data For Loss Calculation ##
        ##########################################

        title_states = title_states * batch_title_mask
        body_states = body_states * batch_body_mask

        # mean pooling of the hidden states of each question's title and body sequences
        title_states = torch.sum(title_states, dim=1, keepdim=False)
        averaged_title_states = title_states * batch_title_lengths.repeat(title_states.size(dim=1), 1).t()

        body_states = torch.sum(body_states, dim=1, keepdim = False)
        averaged_body_states = body_states * batch_body_lengths.repeat(body_states.size(dim=1), 1).t()

        # take the average between the title and body representations for the final representation
        final_question_reps = (averaged_title_states + averaged_body_states).div(2)

        # separate out each training instance in the batch
        training_instances = torch.chunk(final_question_reps, total_batch_size)

        ###############################################
        ## Calculate Loss for Each Training Instance ##
        ###############################################

        print(" ###### Calculating the loss ######")

        cosine_scores = [F.cosine_similarity(instance[1:], instance[0] , -1) for instance in training_instances]
        cosine_scores = torch.stack(cosine_scores, 1).t()

        print cosine_scores.t()

        target_data = [0 for inst in range(len(training_instances))]
        target_data = preprocessing.to_long_variable(target_data)

        # pass in loss data into loss function
        loss = loss_func(cosine_scores, target_data)

        print("batch loss is", loss)

        # back prop and SGD
        back_time_start = time.time()

        # print_params()

        loss.backward()
        print batch_title_data.grad
        print batch_body_data.grad
        optimizer.step()

        # print_params()

        print("backpropogation took ", time.time() - back_time_start)

        # add batch loss to total epoch loss
        epoch_loss += loss.data[0]

        print("this batch took ", time.time() - batch_time_start)

        if i > 0 and i % 50 == 0:
            torch.save(lstm, 'qr_lstm_model_android.pt')


    # save model after each epoch
    torch.save(lstm, 'qr_lstm_model_android.pt')
    return epoch_loss

def print_params():
    for k, v in lstm.params.items():
        print k
        print v

def load_model():
    return torch.load('qr_lstm_model.pt')

def initialize(new_model=False):

    if new_model:
        model = BiDiLSTM(input_dim=INPUT_SIZE, hidden_dim=HIDDEN_SIZES[2], num_layers=NUM_LAYERS, output_dim=OUTPUT_SIZE)
    else:
        model = load_model()

    mml = nn.MultiMarginLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATES[1], weight_decay=L2_NORMS[2])

    return model, mml, optimizer
