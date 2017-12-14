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
		p_drop = 0.3
		self.conv1 = nn.Conv1d(
			in_channels = 200,
			out_channels = 64,
			kernel_size = 5, 
			padding = 2
		)
		self.dropout1 = nn.Dropout(p_drop)


	def forward(self, x, conv_len_mask):
		# print('______-_____')
		# print(x)
		# print(conv_len_mask)

		x = self.conv1(x)
		# print(x)

		x = torch.mul(x, conv_len_mask)
		# print(x)

		x = torch.sum(x, dim=2)
		# print(x)

		x = self.dropout1(x)

		x = F.tanh(x)
		# print(x)

		# print('______^______')
		return x



class CNNTrainer:
	"""
	"""
	def __init__(self, batch_size, lr, delta, debug=False):
		self.batch_size = batch_size
		self.lr = lr
		self.delta = delta
		self.debug = debug
		SENTENCE_LENGTHS = 100

		self.conv_net_cell = Net()
		self.preprocessor = PreConv(self.debug)

	def get_cosine_scores_target_data(self, train_instances):
		cosine_scores = [F.cosine_similarity(instance[1:], instance[0] , -1) for instance in train_instances]
		cosine_scores = torch.stack(cosine_scores, 1).t()

		print(cosine_scores.t())

		target_data = [0 for inst in range(len(train_instances))]
		target_data = Variable(torch.LongTensor(target_data))
		return cosine_scores, target_data

	def get_mini_mask(self, seq_len, out_channels=64, max_seq_len=100):
		n = min(max_seq_len, seq_len)
		mask = torch.cat( (torch.ones(n) * 1.0/n, torch.zeros(max_seq_len - n)) ).repeat(out_channels, 1)
		mask = Variable(torch.FloatTensor(mask), requires_grad = False)
		return mask

	def get_mask(self, lens, out_channels=64, max_seq_len=100):
		mask = [self.get_mini_mask(len) for len in lens]
		mask = torch.stack(mask)
		return mask

	def run_through_model(self, x, lens):
		# run the model on the title/body
		x = torch.stack(x)
		x = torch.transpose(x, 1, 2)
		# need the length of each text to average later
		conv_mask = self.get_mask(lens)
		return self.conv_net_cell(x, conv_mask)

	def sequences_to_input_vecs(self, sequences):
		return [self.preprocessor.sequence_to_vec(seq) for seq in sequences]

	def sequences_to_len_masks(self, sequences):
		return [self.preprocessor.get_seq_len(seq) for seq in sequences]
		

	def train(self):
		print('pre-processing . . .')
		vector_dict = self.preprocessor.get_word_to_vector_dict()
		questions = self.preprocessor.get_question_dict()
		candidate_ids = self.preprocessor.get_candidate_ids()

		#gettings the batches as ids (not yet the actual data)
		id_batches = self.preprocessor.split_into_batches(candidate_ids.keys(), self.batch_size)


		print('setting up model . . .')
		mml = nn.MultiMarginLoss()
		optimizer = optim.Adam(
			params=self.conv_net_cell.parameters(),
			lr=self.lr)

		i_batch = -1
		for id_batch in id_batches:
			i_batch += 1
			batch_loss = 0

			title_seqs, body_seqs = [], []
			for q_id in id_batch:
				
				question_title_seq, question_body_seq = questions[q_id]

				pos_id = random.choice(candidate_ids[q_id][0])

				pos_title_seq, pos_body_seq = questions[pos_id]
				neg_title_seqs = [questions[n][0] for n in candidate_ids[q_id][1]]
				neg_body_seqs = [questions[n][1] for n in candidate_ids[q_id][1]]
				
				# put all sequences together
				title_seqs.extend([question_title_seq] + [pos_title_seq] + neg_title_seqs)
				body_seqs.extend([question_body_seq] + [pos_body_seq] + neg_body_seqs)


			# get all the word embedding vectors
			x_titles, x_bodies = self.sequences_to_input_vecs(title_seqs), self.sequences_to_input_vecs(body_seqs)
			
			# get the lengths of all the sequences
			lens_titles, lens_bodies = self.sequences_to_len_masks(title_seqs), self.sequences_to_len_masks(body_seqs)
			
			# run the model on the bodies
			output_titles = self.run_through_model(x_titles, lens_titles)
			output_bodies = self.run_through_model(x_bodies, lens_bodies)
			
			# average the two for each corresponding question
			out_avg = (output_titles + output_bodies).div(2)				
			
			# run the output through the loss
			train_instances = torch.chunk(out_avg, self.batch_size)
			cos_scores, targets = self.get_cosine_scores_target_data(train_instances)

			loss = mml(cos_scores, targets)

			# back propagate the errors
			optimizer.zero_grad()
			loss.backward()
			nn.utils.clip_grad_norm(self.conv_net_cell.parameters(), 5)
			optimizer.step()

			print('batch %s loss = %s'%(str(i_batch), str(loss)))

		torch.save(conv_net_cell.state_dict(), 'model_cnn_v1.pt')




debug=True
torch.manual_seed(1)
random.seed(1)
BATCH_SIZE = 16
LEARNING_RATE = .0000001
DELTA = 0.01 # is was 0.0001 earlier
# BATCH_SIZE = 64

# getting data before batches
trainer = CNNTrainer(BATCH_SIZE, LEARNING_RATE, DELTA, debug)
trainer.train()


## nn.utils.clip_grad_norm(net.parameters(), 10)



