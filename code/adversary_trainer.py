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

from qr_cnn import CNN_Net
from discriminator import Discriminator

# from qr_lstm import ######################################### TODO FOR LSTM

import adversary_params as params
from pre_android import PreAndroid
from adversary_evaluator import AdversaryEvaluator



class AdversaryTrainer:
	def __init__(self):
		if params.load_encoder_path == '':
			self.encoder_net = self.get_new_encoder_net()
		else:
			self.encoder_net = torch.load(params.load_encoder_path)


		self.discr_input_size = params.cnn_out_channels

		if params.load_discr_path == '':
			self.discr_net = Discriminator(self.discr_input_size, params.discr_hidden_size)
		else:
			self.discr_net = torch.load(params.load_discr_path)

		self.ub_preprocessor = PreConv(
			debug=params.debug, 
			vectors_path=params.glove_vecs_path, 
			emb_channels=params.glove_vecs_n_channels)
		self.an_preprocessor = PreAndroid(debug=params.debug)

		self.evaluator = AdversaryEvaluator()


	def get_new_encoder_net(self):
		if params.encoder_type == 'cnn':
			return CNN_Net(params.cnn_out_channels, params.dropout, params.glove_vecs_n_channels)
		elif params.encoder_type == 'lstm':
			return None ###################################################################### TODO IMPLEMENT FOR LSTM

	def get_ub_title_and_body_seqs(self, questions, candidate_ids, ids_batch):
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

	def get_an_title_and_body_seqs(self, questions, ids_batch):
		title_seqs = [questions[q_id][0] for q_id in ids_batch]
		body_seqs = [questions[q_id][1] for q_id in ids_batch]
		return title_seqs, body_seqs

	def get_cosine_scores_target_data(self, train_instances):
		cosine_scores = [F.cosine_similarity(instance[1:], instance[0] , -1) for instance in train_instances]
		cosine_scores = torch.stack(cosine_scores, 1).t()

		target_data = [0 for _ in range(len(train_instances))]
		target_data = Variable(torch.LongTensor(target_data))
		return cosine_scores, target_data

	def get_total_questions_per_batch(self):
		ub_num_questions_per_batch = params.batch_size * 22
		return ub_num_questions_per_batch * 2 # android batch is the same size

	def get_mini_mask(self, seq_len, max_seq_len=100):
		n = min(max_seq_len, seq_len)
		mask = torch.cat( (torch.ones(n) * 1.0/n, torch.zeros(max_seq_len - n)) ).repeat(params.cnn_out_channels, 1)
		mask = Variable(torch.FloatTensor(mask), requires_grad = False)
		return mask

	def get_mask(self, lens):
		mask = [self.get_mini_mask(len) for len in lens]
		mask = torch.stack(mask)
		return mask

	def run_through_encoder(self, x, lens):
		if params.encoder_type == 'cnn':
			return self.run_through_cnn(x, lens)
		elif params.encoder_type == 'lstm':
			return self.run_through_lstm(x, lens)

	def run_through_cnn(self, x, lens):
		# run the model on the title/body
		x = torch.stack(x)
		x = torch.transpose(x, 1, 2)
		# need the length of each text to average later
		conv_mask = self.get_mask(lens)
		return self.encoder_net(x, conv_mask)

	def run_through_lstm(self, x, lens):
		##################################################################################### TODO: IMPLEMENT FOR LSTM!
		return

	def run_through_discr(self, both_out_avg): return torch.squeeze(self.discr_net(both_out_avg))

	def train(self):
		# get the ubuntu data (labeled ub)
		ub_questions = self.ub_preprocessor.get_question_dict()
		ub_candidate_ids = self.ub_preprocessor.get_candidate_ids()

		ub_ids_batches = self.ub_preprocessor.split_into_batches(ub_candidate_ids.keys(), params.batch_size)

		# get the android data (labeled an)
		an_questions = self.an_preprocessor.get_question_dict()
		an_id_pairs = self.an_preprocessor.get_all_id_pairs()
		
		# batch the ids
		an_ids_batches = self.an_preprocessor.split_into_batches(an_id_pairs)

		# discriminator labels (ubuntu --> 0, android --> 1)
		# when forwarding thru discriminator, first half if ubuntu, second half if android
		total_questions_per_batch = self.get_total_questions_per_batch()
		discr_targets = torch.cat([
			torch.ones(total_questions_per_batch / 2), 
			torch.zeros(total_questions_per_batch / 2)])
		discr_targets = Variable(torch.FloatTensor(discr_targets), requires_grad = False)

		# loss for classifier
		mml = nn.MultiMarginLoss()
		# loss for discriminator
		bcel = nn.BCELoss()

		# 2 different optimizers 
		optimizer1 = optim.Adam(
			params=self.encoder_net.parameters(),
			lr=params.forward_lr)

		optimizer2 = optim.Adam(
			[
				{'params':self.encoder_net.parameters(), 'lr': params.neg_lr}, 
				{'params':self.discr_net.parameters()}
			],
			lr=params.forward_lr)

		# start looping through batches
		last_time = time.time()
		start_time = time.time()
		total_time = 0.0

		n_batches = min(len(ub_ids_batches), len(an_ids_batches))
		for i_batch in range(n_batches):
			ub_ids_batch = ub_ids_batches[i_batch]
			an_ids_batch = an_ids_batches[i_batch]

			# get the input sequences
			ub_title_seqs, ub_body_seqs = self.get_ub_title_and_body_seqs(
				ub_questions, ub_candidate_ids, ub_ids_batch)
			an_title_seqs, an_body_seqs = self.get_an_title_and_body_seqs(
				an_questions, an_ids_batch)

			# get all the word embedding vectors
			ub_x_titles = [self.ub_preprocessor.sequence_to_vec(seq) for seq in ub_title_seqs]
			ub_x_bodies = [self.ub_preprocessor.sequence_to_vec(seq) for seq in ub_body_seqs]
			an_x_titles = [self.an_preprocessor.sequence_to_vec(seq) for seq in an_title_seqs]
			an_x_bodies = [self.an_preprocessor.sequence_to_vec(seq) for seq in an_body_seqs]

			# get the lengths of all the sequences
			ub_lens_titles = [self.ub_preprocessor.get_seq_len(seq) for seq in ub_title_seqs]
			ub_lens_bodies = [self.ub_preprocessor.get_seq_len(seq) for seq in ub_body_seqs]
			an_lens_titles = [self.an_preprocessor.get_seq_len(seq) for seq in an_title_seqs]
			an_lens_bodies = [self.an_preprocessor.get_seq_len(seq) for seq in an_body_seqs]

			# run the ubuntu data forward through the cnn model
			ub_output_titles  = self.run_through_encoder(ub_x_titles, ub_lens_titles)
			ub_output_bodies = self.run_through_encoder(ub_x_bodies, ub_lens_bodies)
			# run the android data forward through the cnn model
			an_output_titles = self.run_through_encoder(an_x_titles, an_lens_titles)
			an_output_bodies = self.run_through_encoder(an_x_bodies, an_lens_bodies)

			# average the representations
			ub_out_avg = (ub_output_titles + ub_output_bodies).div(2)
			an_out_avg = (an_output_titles + an_output_bodies).div(2)

			# now we have the internal feature representations
			# these features will go to the classifier (just cosine similarity and loss1)
			# and the features will go through the discriminator network (ending with loss2)

			# do the classification and loss1 for just the ubuntu data
			ub_train_instances = torch.chunk(ub_out_avg, len(ub_ids_batch))
			ub_cos_scores, ub_targets = self.get_cosine_scores_target_data(ub_train_instances)

			loss1 = mml(ub_cos_scores, ub_targets)

			# do discrimination and loss2 for both ubuntu and android
			
			# concatenate both ub and an
			both_out_avg = torch.cat([ub_out_avg, an_out_avg])

			# flatten for discriminator (has fc1 layer) ########### not sure if I have to do this actually...


			# run through discriminator
			out_discr = self.run_through_discr(both_out_avg)

			# calculate loss2
			# print(out_discr.size())
			# print(discr_targets.size())
			loss2 = bcel(out_discr, discr_targets)

			# create the total loss
			total_loss = loss1 - params.lambda_reg * loss2

			# now back propagate both optimizers
			optimizer1.zero_grad()
			optimizer2.zero_grad()

			total_loss.backward()
			# nn.utils.clip_grad_norm(self.conv_net_cell.parameters(), 20)
			
			optimizer1.step()
			optimizer2.step()


			mod_size = 100.0

			i_batch_print = i_batch + 1
			if (i_batch_print % mod_size) == 0:
				print('---------------------------------------------|------------------|')
				print('batch %d out of %d . . . loss per batch  =|  %s  |'
				%(i_batch_print, n_batches, list(total_loss.data)[0]))
				print('---------------------------------------------|------------------|')
				print('loss1 = %f'%(list(loss1.data)[0]))
				print('loss2 = %f'%(list(loss2.data)[0]))
				
				total_time = time.time() - start_time
				print('training for %f minutes so far'
					%(total_time / 60.0))
				pred_time = (total_time / i_batch_print) * n_batches  / 60.0
				print('training on track to take %f minutes'
					%(pred_time))

				last_time = time.time()

		self.evaluator.evaluate()
		torch.save(self.encoder_net, params.save_encoder_path)
		torch.save(self.discr_net, params.save_discr_path)




if __name__ == '__main__':
	torch.manual_seed(1)
	random.seed(1)

	trainer = AdversaryTrainer()
	trainer.train()





