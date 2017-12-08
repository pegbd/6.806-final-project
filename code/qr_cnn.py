from __future__ import print_function
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from pre_conv import PreConv
import random.choice
import random.seed


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
        score1s = [F.cosine_similarity(q, n, 0) for n in negatives].extend(score1s_pos)
        
        values = [score1 - score2 + delta for score1 in score1s]
        return torch.max(torch.stack(values))


# cnn model
class ConvNetCell(nn.Module):
	"""
	"""
	def __init__(self, sentence_length):
		super(Net, self).__init__()
		# conv layer
		self.conv1 = nn.Conv1d(
			in_channels = sentence_length, 
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
	def __init__(self, lr, l2_norm):
		self.lr = lr
		self.l2_norm = l2_norm
		SENTENCE_LENGTHS = 100
		

	def train(self):
		preprocessor = PreConv()
		questions = preprocessor.get_questions_dict()
		candidate_ids = preprocessor.get_candidate_ids()

		#gettings the batches as ids (not yet the actual data)
		id_batches = preprocessor.split_into_batches(candidate_ids.keys(), batch_size)


		conv_net_cell = ConvNetCell(SENTENCE_LENGTHS)
		mm_loss = QR_Maximum_Margin_Loss()
		optimizer = optim.Adam(
			params=conv_net_cell.parameters(), 
			lr=lr, 
			weight_decay=l2_norm)

		i_batch = -1
		for id_batch in self.id_batches:
			i_batch += 1
			batch_loss = 0
			for q_id in batch:
				question_title, question_body = questions[q_id]
				pos_id = random.choice(candidates[q_id][0])

				pos_title, pos_body = questions[pos_id]
				neg_titles = [questions[n][0] for n in candidates[qr][1]]
				neg_bodies = [questions[n][1] for n in candidates[qr][1]]

				# run the model on the titles
				x_titles = [question_title, pos_title].extend(neg_titles)
				output_titles = conv_net_cell(x_titles)

				# run the model on the bodies
				x_bodies = [question_body, pos_body].extend(neg_bodies)
				output_bodies = conv_net_cell(x_bodies)

				# average the two for each corresponding question
				out_avg = torch.sum(x_titles + x_bodies) / 2				
				
				# run the output through the loss
				loss = mm_loss(out_avg)
				print(loss)
				batch_loss += loss

				# back propagate the errors
				optimizer.zero_grad()
				optimizer.backward()
				optimizer.step()

			print('batch %s loss = %s'%(str(i_batch), str(batch_loss)))

		conv_net_cell.save_state_dict('model_cnn_v1.pt')



if __name__ == '__main__':

	torch.manual_seed(1)
	random.seed(1)
	LEARNING_RATE = .00001
	L2_NORM = .00001
	# BATCH_SIZE = 64

	# getting data before batches
	trainer = CNNTrainer()
	trainer.train()



