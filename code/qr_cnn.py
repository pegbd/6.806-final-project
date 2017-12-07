import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from pre_conv import PreConv
import random.choice
import random.seed

# cnn model


class ConvNetCell(nn.Module):
	"""
	"""
	def __init__(self, output_dim):
		super(Net, self).__init__()

		# conv layer

		in_channels = 1 # TODO
		kernel_size1 = 5
		padding1 = 4
		out_channels1 = 20
		self.conv1 = nn.Conv1d(in_channels, out_channels1)


		# mean pooling layer
		kernel_size_mp1 = 4
		stride_mp1 = 2
		self.mean_pool1 = nn.AvgPool1d(kernel_size2, stride_mp1)

	def forward(self, X):
		# TODO




class CNNTrainer:
	"""
	"""
	def __init__(self, batch_size):
		self.batch_size = batch_size
		self.id_batches = self.init_id_batches()
		

	def init_id_batches(self):
		

	def train(self):
		preprocessor = PreConv()
		words = preprocessor.get_word_to_vector_dict()
		questions = preprocessor.get_questions_dict()
		candidate_ids = preprocessor.get_candidate_ids()

		#gettings the batches as ids (not yet the actual data)
		id_batches = preprocessor.split_into_batches(candidate_ids.keys(), batch_size)

		cnn = nn.Conv1d(1, 1)
		optimizer = optim.Adam(
			params=cnn.parameters(), 
			lr=LEARNING_RATES[0], 
			weight_decay=L2_NORMS[0])

		for id_batch in self.id_batches:
			batch_loss = 0
			for q_id in batch:
				question = questions[q_id]
				positives_q = random.choice(candidates[q_id]p[0])
				########################
				######################
				###################
				# todo: get one random positive sample





####################################################################
############################# TRAINING #############################
####################################################################

# Network Setup
torch.manual_seed(1)
random.seed(1)
INPUT_SIZE = 200
OUTPUT_SIZE = 2
LEARNING_RATES = [.00001, .001, .1, 10]
L2_NORMS = [.00001, .001, 10]
NUM_ITERATIONS = 1
BATCH_SIZE = 64

# getting data before batches
trainer = CNNTrainer()
trainer.train()


# Initialize Network and Optimizer
### in the trainer


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

