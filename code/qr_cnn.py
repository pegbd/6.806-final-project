from __future__ import print_function
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from pre_conv import PreConv
import random


# 1: Custom Maximum Margin Loss class
class QR_Maximum_Margin_Loss(nn.Module):
    def __init__(self):
        super(QR_Maximum_Margin_Loss, self).__init__()

    def forward(self, y_pred, delta=.01):
    	'''

    	loss = max_p(score(q, p) - score(q, p_i) + delta)
    	     = max_p(score1 - score2 + delta)

    	'''
        # ctx.save_for_backward(q, positive, negatives, del_value)

        q, positive, negatives = y_pred[0], y_pred[1], y_pred[2:]
        
        score2 = F.cosine_similarity(q, positive, 0)
        score1s_pos = score2 - delta # + delta to account for adding delta later 
        score1s = [F.cosine_similarity(q, n, 0) for n in negatives]
        score1s.append(score1s_pos)
        
        values = [score1 - score2 + delta for score1 in score1s]
        return torch.max(torch.stack(values))


# cnn model
class Net(nn.Module):
	"""
	"""
	def __init__(self):
		super(Net, self).__init__()
		# conv layer
		self.conv1 = nn.Conv1d(
			in_channels = 100, 
			out_channels = 64, 
			kernel_size = 5,
			padding = 3
		)

	def forward(self, X, sentence_lengths):
		"""
		Args:
			X: shape of (n_examples, in_channels, n_features)
			sentence_lengths = shape of (n_examples, 1, 1)

		Returns:
			x: Pytorch variable of shape 
				(n_examples, 1, n_features)
		"""
		# TODO: apply batch normalization ???

		x = self.conv1(X)
		x = torch.sum(x, dim=1) / sentence_lengths
		x = F.tanh(x)
		return x



class CNNTrainer:
	"""
	"""
	def __init__(self, batch_size, lr, l2_norm, debug=False):
		self.batch_size = batch_size
		self.lr = lr
		self.l2_norm = l2_norm
		self.debug = debug
		SENTENCE_LENGTHS = 100
		

	def train(self, conv_net_cell):
		print('pre-processing . . .')
		preprocessor = PreConv(self.debug)
		questions = preprocessor.get_question_dict()
		candidate_ids = preprocessor.get_candidate_ids()

		#gettings the batches as ids (not yet the actual data)
		id_batches = preprocessor.split_into_batches(candidate_ids.keys(), self.batch_size)


		print('setting up model . . .')
		
		mm_loss = QR_Maximum_Margin_Loss()
		optimizer = optim.Adam(
			params=conv_net_cell.parameters(),
			lr=self.lr, 
			weight_decay=self.l2_norm)

		i_batch = -1
		for id_batch in id_batches:
			i_batch += 1
			batch_loss = 0
			for q_id in id_batch:
				optimizer.zero_grad()
				question_title, question_body, q_len_title, q_len_body = questions[q_id]
				pos_id = random.choice(candidate_ids[q_id][0])

				pos_title, pos_body, pos_len_title, pos_len_body = questions[pos_id]
				neg_titles = [questions[n][0] for n in candidate_ids[q_id][1]]
				neg_bodies = [questions[n][1] for n in candidate_ids[q_id][1]]
				neg_len_titles = [questions[n][2] for n in candidate_ids[q_id][1]]
				neg_len_bodies = [questions[n][3] for n in candidate_ids[q_id][1]]

				# run the model on the titles
				x_titles = [question_title, pos_title]
				x_titles.extend(neg_titles)
				# need the length of each title to average later
				x_lens_titles = [q_len_title, pos_len_title]
				x_lens_titles.extend(neg_len_titles)
				output_titles = conv_net_cell(
					torch.stack(x_titles), 
					torch.stack(x_lens_titles))

				# run the model on the bodies
				x_bodies = [question_body, pos_body]
				x_bodies.extend(neg_bodies)
				# need the length of each body to average later
				x_lens_bodies = [q_len_body, pos_len_body]
				x_lens_bodies.extend(neg_len_bodies)
				output_bodies = conv_net_cell(
					torch.stack(x_bodies),
					torch.stack(neg_len_bodies))

				# average the two for each corresponding question
				out_avg = torch.sum(out_titles + out_bodies) / 2.0				
				
				# run the output through the loss
				loss = mm_loss(out_avg)
				loss.backward()
				print(loss)
				batch_loss += loss

				# back propagate the errors
				optimizer.step()

			print('batch %s loss = %s'%(str(i_batch), str(batch_loss)))

		conv_net_cell.save_state_dict('model_cnn_v1.pt')




debug=True
torch.manual_seed(1)
random.seed(1)
BATCH_SIZE = 1
LEARNING_RATE = .00001
L2_NORM = .00001
# BATCH_SIZE = 64

# getting data before batches
trainer = CNNTrainer(BATCH_SIZE, LEARNING_RATE, L2_NORM, debug)
conv_net_cell = Net()
trainer.train(conv_net_cell)



