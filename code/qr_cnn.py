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
class CNN_Net(nn.Module):
	"""
	"""
	def __init__(self, out_channels, p_drop=0.3):
		super(CNN_Net, self).__init__()
		self.out_channels = out_channels
		# conv layer
		self.p_drop = p_drop
		self.conv1 = nn.Conv1d(
			in_channels = 200,
			out_channels = self.out_channels,
			kernel_size = 5, 
			padding = 2
		)
		self.dropout1 = nn.Dropout(self.p_drop)


	def forward(self, x, conv_len_mask):
		x = self.conv1(x)
		x = torch.mul(x, conv_len_mask)
		x = torch.sum(x, dim=2)
		x = self.dropout1(x)
		x = F.tanh(x)
		return x


class CNNModel:
	def __init__(self, batch_size, out_channels, debug=False):
		self.batch_size = batch_size
		self.out_channels = out_channels
		self.debug = debug
		SENTENCE_LENGTHS = 100

		self.conv_net_cell = CNN_Net(out_channels=self.out_channels, p_drop=0.3)
		self.preprocessor = PreConv(debug=self.debug)

	def get_mini_mask(self, seq_len, out_channels=64, max_seq_len=100):
		n = min(max_seq_len, seq_len)
		mask = torch.cat( (torch.ones(n) * 1.0/n, torch.zeros(max_seq_len - n)) ).repeat(out_channels, 1)
		mask = Variable(torch.FloatTensor(mask), requires_grad = False)
		return mask

	def get_mask(self, lens, out_channels=64, max_seq_len=100):
		mask = [self.get_mini_mask(len, out_channels) for len in lens]
		mask = torch.stack(mask)
		return mask

	def run_through_model(self, x, lens):
		# run the model on the title/body
		x = torch.stack(x)
		x = torch.transpose(x, 1, 2)
		# need the length of each text to average later
		conv_mask = self.get_mask(lens, self.out_channels)
		return self.conv_net_cell(x, conv_mask)

	def get_cosine_scores_target_data(self, train_instances):
		cosine_scores = [F.cosine_similarity(instance[1:], instance[0] , -1) for instance in train_instances]
		cosine_scores = torch.stack(cosine_scores, 1).t()

		# print('cosine scores size:')
		# print(cosine_scores.t().size())

		target_data = [0 for inst in range(len(train_instances))]
		target_data = Variable(torch.LongTensor(target_data))
		return cosine_scores, target_data

	def sequences_to_input_vecs(self, sequences):
		return [self.preprocessor.sequence_to_vec(seq) for seq in sequences]

	def sequences_to_len_masks(self, sequences):
		return [self.preprocessor.get_seq_len(seq) for seq in sequences]

class CNNEvaluator(CNNModel):
	def __init__(self, out_channels, debug=False):
		# self.batch_size = batch_size
		self.out_channels = out_channels
		self.debug = debug
		SENTENCE_LENGTHS = 100

		self.conv_net_cell = CNN_Net(self.out_channels)
		self.preprocessor = None

	def init_preprocessor(self, data_type):
		self.preprocessor = PreConv(data_type=data_type, debug=self.debug)

	def get_eval_data(self, data_type):
		self.init_preprocessor(data_type)

		eval_data = []

		if data_type == 'dev': preprocessor = PreConv(data_type='dev')
		
		if data_type == 'test': preprocessor = PreConv(data_type='test')

		# vector_dict = self.preprocessor.get_word_to_vector_dict()
		questions = self.preprocessor.get_question_dict()
		candidate_ids = self.preprocessor.get_candidate_ids()
		pos_indicies_all = self.preprocessor.get_pos_indicies_all()

		#gettings the batches as ids (not yet the actual data)
		id_batches = self.preprocessor.split_into_batches(candidate_ids.keys(), batch_size=1)

		for ids_batch in id_batches:
			# print('one eval batch')

			title_seqs, body_seqs = [], []

			# for q_id in ids_batch:
			q_id = ids_batch[0] # this line replaces the 'for' line right above

			question_title_seq, question_body_seq = questions[q_id]

			####### _ _ _ don't do positive example for eval _ _ _ #######
			# pos_id = random.choice(candidate_ids[q_id][0])

			# pos_title_seq, pos_body_seq = questions[pos_id]
			####### ^^ don't do pos example for eval ^^ ########

			neg_title_seqs = [questions[n][0] for n in candidate_ids[q_id][1]]
			neg_body_seqs = [questions[n][1] for n in candidate_ids[q_id][1]]

			# put all sequences together
			# title_seqs.extend([question_title_seq] + [pos_title_seq] + neg_title_seqs)
			title_seqs.extend([question_title_seq] + neg_title_seqs)
			# body_seqs.extend([question_body_seq] + [pos_body_seq] + neg_body_seqs)
			body_seqs.extend([question_body_seq] + neg_body_seqs)

			################ commented out for loop ends ####################

			# get all the word embedding vectors
			x_titles, x_bodies = self.sequences_to_input_vecs(title_seqs), self.sequences_to_input_vecs(body_seqs)
			
			# get the lengths of all the sequences
			lens_titles, lens_bodies = self.sequences_to_len_masks(title_seqs), self.sequences_to_len_masks(body_seqs)
			
			# run the model on the bodies
			output_titles, output_bodies = self.run_through_model(x_titles, lens_titles), self.run_through_model(x_bodies, lens_bodies)
			
			# average the two for each corresponding question
			out_avg = (output_titles + output_bodies).div(2)          
			
			# run the output through the loss
			cosine_scores = F.cosine_similarity(out_avg[1:], out_avg[0] , -1)
			scores = list(cosine_scores.data)

			pos_indices = pos_indicies_all[q_id]
			sorted_eval = [x for _,x in sorted(zip(scores, pos_indices), reverse=True)]
			eval_data.append(sorted_eval)

		return eval_data

	def evaluate(self, model): 
		eval_dev_data = self.get_eval_data(data_type='dev')
		eval_test_data = self.get_eval_data(data_type='test')

		evaluation_of_dev = Evaluation(eval_dev_data)
		evaluation_of_test = Evaluation(eval_test_data)

		print("\nDEV\n")
		print("MAP", evaluation_of_dev.MAP())
		print("MRR", evaluation_of_dev.MRR())
		print("P@1", evaluation_of_dev.Precision(1))
		print("P@5", evaluation_of_dev.Precision(5))

		print("\nTEST\n")
		print("MAP", evaluation_of_test.MAP())
		print("MRR", evaluation_of_test.MRR())
		print("P@1", evaluation_of_test.Precision(1))
		print("P@5", evaluation_of_test.Precision(5))


class CNNTrainer(CNNModel):
	def __init__(self, batch_size, lr, delta, out_channels, save_model_path, load_model_path='', debug=False, p_drop=0.3):
		self.batch_size = batch_size
		self.lr = lr
		self.delta = delta
		self.out_channels = out_channels
		self.save_model_path = save_model_path
		self.load_model_path = load_model_path
		self.debug = debug
		self.p_drop = p_drop
		SENTENCE_LENGTHS = 100

		if self.load_model_path == '': 
			self.conv_net_cell = CNN_Net(out_channels=self.out_channels, p_drop=self.p_drop)
		else: 
			self.conv_net_cell = torch.load(self.load_model_path)

		self.preprocessor = PreConv(debug=self.debug)
		self.evaluator = CNNEvaluator(self.out_channels)

	def get_title_and_body_seqs(self, questions, candidate_ids, ids_batch):
		title_seqs, body_seqs = [], []
		for q_id in ids_batch:
			
			question_title_seq, question_body_seq = questions[q_id]

			pos_id = random.choice(candidate_ids[q_id][0])

			pos_title_seq, pos_body_seq = questions[pos_id]
			neg_title_seqs = [questions[n][0] for n in candidate_ids[q_id][1]]
			neg_body_seqs = [questions[n][1] for n in candidate_ids[q_id][1]]
			
			# put all sequences together
			title_seqs.extend([question_title_seq] + [pos_title_seq] + neg_title_seqs)
			body_seqs.extend([question_body_seq] + [pos_body_seq] + neg_body_seqs)
		return title_seqs, body_seqs

		

	def train(self):
		questions = self.preprocessor.get_question_dict()
		candidate_ids = self.preprocessor.get_candidate_ids()

		#gettings the batches as ids (not yet the actual data)
		id_batches = self.preprocessor.split_into_batches(candidate_ids.keys(), self.batch_size)

		mml = nn.MultiMarginLoss()
		optimizer = optim.Adam(
			params=self.conv_net_cell.parameters(),
			lr=self.lr)

		i_batch = 0
		last_time = time.time()
		start_time = time.time()
		total_time = 0.0
		for ids_batch in id_batches:
			i_batch += 1

			# get the input sequences
			title_seqs, body_seqs = self.get_title_and_body_seqs(questions, candidate_ids, ids_batch)

			# get all the word embedding vectors
			x_titles, x_bodies = self.sequences_to_input_vecs(title_seqs), self.sequences_to_input_vecs(body_seqs)
			
			# get the lengths of all the sequences
			lens_titles, lens_bodies = self.sequences_to_len_masks(title_seqs), self.sequences_to_len_masks(body_seqs)
			
			# run the data forward through the model 
			output_titles, output_bodies = self.run_through_model(x_titles, lens_titles), self.run_through_model(x_bodies, lens_bodies)
			
			# average the two for each corresponding question
			out_avg = (output_titles + output_bodies).div(2)	
			
			# run the output through the loss
			train_instances = torch.chunk(out_avg, len(ids_batch))
			cos_scores, targets = self.get_cosine_scores_target_data(train_instances)

			loss = mml(cos_scores, targets)

			# back propagate the errors
			optimizer.zero_grad()
			loss.backward()
			nn.utils.clip_grad_norm(self.conv_net_cell.parameters(), 20)
			optimizer.step()

			
			mod_size = 500.0
			if (i_batch % mod_size) == 0:
				print('---------------------------------------------|------------------|')
				print('batch %d out of %d . . . loss per batch  =|  %s  |'
				%(i_batch, len(id_batches), list(loss.data)[0]))
				print('---------------------------------------------|------------------|')


				# delta_time = time.time() - last_time 
				# time_per_question = delta_time / (self.batch_size * mod_size)
				# print('training is taking %f seconds per question'
				# 	%( time_per_question ))
				
				total_time = time.time() - start_time
				print('training for %f minutes so far'
					%(total_time / 60.0))
				pred_time = (total_time / i_batch) * len(id_batches)  / 60.0
				print('training on track to take %f minutes'
					%(pred_time))


				last_time = time.time()
	

		# torch.save(self.conv_net_cell.state_dict(), 'model_cnn_v1.pt')
		self.evaluator.evaluate(self.conv_net_cell)
		torch.save(self.conv_net_cell, self.save_model_path)




if __name__ == '__main__':
	models_dir = '../saved_models/'
	

	# v5 model 
	# --> this is the first model with 
	#     conv kernel size == 7 instead of 5 or 3
	version_num = 5
	n_epochs = 10
	for i_epoch in range(n_epochs):
		print('epoch %d out of %d epochs, for model version 4'
			%(i_epoch+1, n_epochs))
		save_model_path = models_dir + 'model_cnn_v%d.pt'%(version_num)
		if i_epoch == 0: load_model_path = ''
		else: load_model_path = models_dir + 'model_cnn_v%d.pt'%(version_num)
		
		DEBUG = False
		torch.manual_seed(7)
		random.seed(7)
		BATCH_SIZE = 5
		LEARNING_RATE = .00001
		DELTA = 0.1
		OUT_CHANNELS = 150
		DROPOUT = 0.3

		# getting data before batches
		trainer = CNNTrainer(BATCH_SIZE, LEARNING_RATE, DELTA, OUT_CHANNELS,
			save_model_path, load_model_path, DEBUG, DROPOUT)
		trainer.train()





	# # after epoch 4
	# training for 15.688509 minutes so far
	# training on track to take 15.964627 minutes

	# DEV

	# MAP 0.452442136527
	# MRR 0.578562564945
	# P@1 0.428571428571
	# P@5 0.359788359788

	# TEST

	# MAP 0.40801480879
	# MRR 0.496276377506
	# P@1 0.306451612903
	# P@5 0.309677419355






############################################################################

	# # v4 model 
	# # --> this is the first model with 
	# #     conv kernel size == 3 instead of 5
	# n_epochs = 10
	# for i_epoch in range(n_epochs):
	# 	print('epoch %d out of %d epochs, for model version 4'
	# 		%(i_epoch+1, n_epochs))
	# 	save_model_path = models_dir + 'model_cnn_v4.pt'
	# 	if i_epoch == 0: load_model_path = ''
	# 	else: load_model_path = models_dir + 'model_cnn_v4.pt'
		
	# 	DEBUG = False
	# 	torch.manual_seed(7)
	# 	random.seed(7)
	# 	BATCH_SIZE = 3
	# 	LEARNING_RATE = .0001
	# 	DELTA = 0.1
	# 	OUT_CHANNELS = 150
	# 	DROPOUT = 0.2

	# 	# getting data before batches
	# 	trainer = CNNTrainer(BATCH_SIZE, LEARNING_RATE, DELTA, OUT_CHANNELS,
	# 		save_model_path, load_model_path, DEBUG, DROPOUT)
	# 	trainer.train()



	# epoch 5 of version 4

	# training is taking 0.053204 seconds per question
	# training for 10.628970 minutes so far
	# training on track to take 11.269366 minutes

	# DEV

	# MAP 0.459988345899
	# MRR 0.571518809544
	# P@1 0.380952380952
	# P@5 0.356613756614

	# TEST

	# MAP 0.425791483675
	# MRR 0.528413060547
	# P@1 0.370967741935
	# P@5 0.330107526882












#######################################################################









	# # v3 model
	# n_epochs = 3
	# for i_epoch in range(n_epochs):
	# 	print('epoch %d out of %d epochs, for model version 3'
	# 		%(i_epoch+1, n_epochs))
	# 	save_model_path = models_dir + 'model_cnn_v3_post_epoch13.pt'
	# 	if i_epoch == 0: load_model_path = ''
	# 	else: load_model_path = models_dir + 'model_cnn_v3_post_epoch13.pt'
		
	# 	DEBUG = False
	# 	torch.manual_seed(1)
	# 	random.seed(1)
	# 	BATCH_SIZE = 3
	# 	LEARNING_RATE = .000001
	# 	DELTA = 1.0 # is was 0.0001 earlier
	# 	OUT_CHANNELS = 200
	# 	DROPOUT = 0.3

	# 	# getting data before batches
	# 	trainer = CNNTrainer(BATCH_SIZE, LEARNING_RATE, DELTA, OUT_CHANNELS,
	# 		save_model_path, load_model_path, DEBUG, DROPOUT)
	# 	trainer.train()


	# V3 on epoch 16 ish

	# training is taking 0.068564 seconds per question
	# training for 13.684439 minutes so far
	# training on track to take 14.508926 minutes

	# DEV

	# MAP 0.450649629339
	# MRR 0.559807960135
	# P@1 0.380952380952
	# P@5 0.353439153439

	# TEST

	# MAP 0.449834048749
	# MRR 0.569960228882
	# P@1 0.403225806452
	# P@5 0.34623655914

	# V3 on epoch 13 ish
	
	# training is taking 0.072794 seconds per question
	# training for 14.798071 minutes so far
	# training on track to take 15.689655 minutes

	# DEV

	# MAP 0.450649629339
	# MRR 0.559807960135
	# P@1 0.380952380952
	# P@5 0.353439153439

	# TEST

	# MAP 0.449834048749
	# MRR 0.569960228882
	# P@1 0.403225806452
	# P@5 0.34623655914




#######################################################################


	# # v2 model
	# n_epochs = 10
	# for i_epoch in range(n_epochs):
	# 	print('epoch %d out of %d epochs, for model version 2'
	# 		%(i_epoch+1, n_epochs))
	# 	save_model_path = models_dir + 'model_cnn_v2_epoch%d.pt'%(i_epoch+1)
	# 	if i_epoch == 0: load_model_path = ''
	# 	else: load_model_path = models_dir + 'model_cnn_v2_epoch%d.pt'%(i_epoch)
		
	# 	DEBUG = False
	# 	torch.manual_seed(1)
	# 	random.seed(1)
	# 	BATCH_SIZE = 5
	# 	LEARNING_RATE = .00001
	# 	DELTA = 1.0 # is was 0.0001 earlier
	# 	OUT_CHANNELS = 128
	# 	DROPOUT = 0.5

	# 	# getting data before batches
	# 	trainer = CNNTrainer(BATCH_SIZE, LEARNING_RATE, DELTA, OUT_CHANNELS,
	# 		save_model_path, load_model_path, DEBUG, DROPOUT)
	# 	trainer.train()

	# epoch 7 version2
	# 	training is taking 0.059478 seconds per question
	# training for 12.573277 minutes so far
	# training on track to take 12.794566 minutes
	# dev set
	# dev set


	# DEV


	# MAP 0.439141599593
	# MRR 0.526655488983
	# P@1 0.338624338624
	# P@5 0.359788359788


	# TEST


	# MAP 0.430796731464
	# MRR 0.531306387758
	# P@1 0.360215053763
	# P@5 0.31935483871





################################################################




	## V1 model
	# save_model_path = models_dir + 'model_cnn_v1_epoch4.pt'
	# load_model_path = models_dir + 'model_cnn_v1_epoch3.pt'
	
	# DEBUG = False
	# torch.manual_seed(1)
	# random.seed(1)
	# BATCH_SIZE = 5
	# LEARNING_RATE = .0001
	# DELTA = 0.01
	# OUT_CHANNELS = 64
	# DROPOUT = 0.3

	# # getting data before batches
	# trainer = CNNTrainer(BATCH_SIZE, LEARNING_RATE, DELTA, OUT_CHANNELS,
	# 	save_model_path, load_model_path, DEBUG, DROPOUT)
	# trainer.train()




