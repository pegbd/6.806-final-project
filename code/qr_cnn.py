from __future__ import print_function
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from pre_conv import PreConv
import random


# 1: Custom Maximum Margin Loss class
# class QR_Maximum_Margin_Loss(nn.Module):
# 	def __init__(self):
# 		super(QR_Maximum_Margin_Loss, self).__init__()

# 	def forward(self, y_pred, delta=.01):
# 		'''

# 		loss = max_p(score(q, p) - score(q, p_i) + delta)
# 			 = max_p(score1 - score2 + delta)

# 		'''
# 		# ctx.save_for_backward(q, positive, negatives, del_value)

# 		q, positive, negatives = y_pred[0], y_pred[1], y_pred[2:]
		
# 		score2 = F.cosine_similarity(q, positive, 0)
# 		score1s_pos = score2 - delta # + delta to account for adding delta later 
# 		score1s = [F.cosine_similarity(q, n, 0) for n in negatives]
# 		score1s.append(score1s_pos)
		
# 		values = [score1 - score2 + delta for score1 in score1s]
# 		return torch.max(torch.stack(values))


# cnn model
class Net(nn.Module):
	"""
	"""
	def __init__(self):
		super(Net, self).__init__()
		# conv layer
		self.conv1 = nn.Conv1d(
			in_channels = 200,
			out_channels = 64,
			kernel_size = 5
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
		print('______-_____')
		print(X)
		print(X.size())
		x = self.conv1(X)
		print(x)
		print(x.size())
		sum1 = torch.sum(x, dim=1)
		print(sum1)
		print(sum1.size())
		lengths = sentence_lengths.repeat(1, sum1.size()[1])
		print(lengths)
		print(lengths.size())
		
		x = sum1.div(lengths)
		print(x)
		x = F.tanh(x)
		print(x)
		print('______^______')
		return x



class CNNTrainer:
	"""
	"""
	def __init__(self, batch_size, lr, l2_norm, delta, debug=False):
		self.batch_size = batch_size
		self.lr = lr
		self.l2_norm = l2_norm
		self.delta = delta
		self.debug = debug
		SENTENCE_LENGTHS = 100

	def get_loss_data(self, train_instances):
		loss_data = []
		for instance in train_instances:

			# rep of question of interest and positive candidate
			h_q = instance[0]
			h_p = instance[1]

			# score of positive candidate
			s_p = F.cosine_similarity(h_q, h_p, 0)
			scores = [s_p - s_p]

			# scores of negatives
			for i in range(2, len(instance)):
				h_n = instance[i]

				score = F.cosine_similarity(h_q, h_n, 0) - s_p + self.delta
				scores.append(score)

			loss_data.append(torch.cat(scores, 0))
		loss_data = torch.stack(loss_data, 1)
		return loss_data

	def run_through_model(self, x, lens):
		# run the model on the title/body
		x = torch.stack(x)
		x = torch.transpose(x, 1, 2)
		# need the length of each text to average later
		lens = Variable(torch.FloatTensor(lens), requires_grad=False)
		lens.resize(lens.size()[0], 1)
		return conv_net_cell(x, lens)
		

	def train(self, conv_net_cell):
		print('pre-processing . . .')
		preprocessor = PreConv(self.debug)
		questions = preprocessor.get_question_dict()
		candidate_ids = preprocessor.get_candidate_ids()

		#gettings the batches as ids (not yet the actual data)
		id_batches = preprocessor.split_into_batches(candidate_ids.keys(), self.batch_size)


		print('setting up model . . .')
		
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


				x_titles = [question_title, pos_title]
				x_titles.extend(neg_titles)
				x_lens_titles = [q_len_title, pos_len_title]
				x_lens_titles.extend(neg_len_titles)

				x_bodies = [question_body, pos_body]
				x_bodies.extend(neg_bodies)
				x_lens_bodies = [q_len_body, pos_len_body]
				x_lens_bodies.extend(neg_len_bodies)

				# run the model on the bodies
				output_titles = self.run_through_model(
					x_titles, x_lens_titles)
				output_bodies = self.run_through_model(
					x_bodies, x_lens_bodies)
				

				# average the two for each corresponding question
				out_avg = (output_titles + output_bodies).div(2)				
				
				# run the output through the loss
				# loss = mm_loss(out_avg)
				train_instances = torch.chunk(out_avg, self.batch_size)
				loss_data = self.get_loss_data(train_instances)
				targets = Variable(torch.LongTensor(
					[0 for i in range(len(loss_data))]), 
					requires_grad=True)

				mml = nn.MultiMarginLoss()
				loss = mml(loss_data, targets)

				# back propagate the errors
				loss.backward()
				print(loss)
				batch_loss += loss
				optimizer.step()
			assert False

			print('batch %s loss = %s'%(str(i_batch), str(batch_loss)))

		torch.save(conv_net_cell.state_dict(), 'model_cnn_v1.pt')




debug=True
torch.manual_seed(1)
random.seed(1)
BATCH_SIZE = 1
LEARNING_RATE = .00001
L2_NORM = .00001
DELTA = 1.0 # is was 0.0001 earlier
# BATCH_SIZE = 64

# getting data before batches
trainer = CNNTrainer(BATCH_SIZE, LEARNING_RATE, L2_NORM, DELTA, debug)
conv_net_cell = Net()
trainer.train(conv_net_cell)


## nn.utils.clip_grad_norm(net.parameters(), 10)



