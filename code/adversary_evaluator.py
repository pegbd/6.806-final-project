from __future__ import print_function
import torch
from torch.nn import functional as F
from torch.autograd import Variable
import random
from meter import AUCMeter
import numpy as np

from pre_android import PreAndroid
from adversary_trainer import AdversaryTrainer

import adversary_params as params
import time



class AdversaryEvaluator(AdversaryTrainer):

	def __init__(self):
		# do not want to load parameters until we actually call evaluate
		self.encoder_net = None
		self.pre = None
		self.questions = None

	def reset_params(self):
		self.encoder_net = torch.load(params.eval_encoder_path)
		self.pre = PreAndroid(debug=False)
		self.questions = self.pre.get_question_dict()


	def get_cosine_scores(self, feats_left, feats_right):
		scores = [F.cosine_similarity(feats_left[i], feats_right[i], -1) for i in xrange(len(feats_left))]
		scores = torch.squeeze(torch.stack(scores))
		return scores

	def get_output(self, ids_batch):
		# get the input sequences
		title_seqs, body_seqs = self.get_an_title_and_body_seqs(
			self.questions, ids_batch)

		# get the word embedding vectors
		x_titles = [self.pre.sequence_to_vec(seq) for seq in title_seqs]
		x_bodies = [self.pre.sequence_to_vec(seq) for seq in body_seqs]

		# get the lengths of all the sequences
		lens_titles = [self.pre.get_seq_len(seq) for seq in title_seqs]
		lens_bodies = [self.pre.get_seq_len(seq) for seq in body_seqs]

		# run the android data forward through the cnn model
		output_titles = self.run_through_encoder(x_titles, lens_titles)
		output_bodies = self.run_through_encoder(x_bodies, lens_bodies)

		# average the representations
		out_avg = (output_titles + output_bodies).div(2)
		return out_avg

	def evaluate(self, dev_or_test):
		''' dev_or_test must be one of 'dev' or 'test'
		'''
		print('lv0')
		self.reset_params()
		auc_meter = AUCMeter()

		# get the id batches
		pos_ids_batches_pair = self.pre.eval_split_into_batches(is_pos=True, dev_or_test=dev_or_test)
		neg_ids_batches_pair = self.pre.eval_split_into_batches(is_pos=False, dev_or_test=dev_or_test)

		# start looping thru the batches
		data_sets = [neg_ids_batches_pair, pos_ids_batches_pair]
		print('lv1')
		i_target = 0
		for ids_batches_pair in data_sets:
			assert i_target < 2
			print('dataset number %d'%(i_target))

			ids_batches_left = ids_batches_pair[0]
			ids_batches_right = ids_batches_pair[1]

			for i in xrange(len(ids_batches_left)):
				ids_batch_left = ids_batches_left[i]
				ids_batch_right = ids_batches_right[i]

				feats_left = self.get_output(ids_batch_left)
				feats_right = self.get_output(ids_batch_right)

				preds = self.get_cosine_scores(feats_left, feats_right).data.numpy()
				targets = np.ones(len(preds)) * i_target # 0s if neg, 1s if pos
				auc_meter.add(preds, targets)

			i_target += 1
			
		print('lv3')
		# all predictions are added
		# now get the AUC value
		auc_value = auc_meter.value(params.auc_max_fpr)
		print('AUC(%f) value for %s  =  %f'
			%(params.auc_max_fpr, dev_or_test, auc_value))










if __name__ == '__main__':
	start_time = time.time()
	torch.manual_seed(1)
	random.seed(1)

	evaluator = AdversaryEvaluator()
	
	evaluator.evaluate('test')
	# evaluator.evaluate('dev')
	total_minutes = (time.time()-start_time) / 60.0
	print('evaluation took %f minutes'%(total_minutes))


# for
# 	AUC(0.050000) value for dev  =  0.454898
# evaluation took 19.061163 minutes













######################################################################
	# AUC(0.050000) value for test  =  0.580768
	# AUC(0.050000) value for dev  =  0.581712
	# evaluation took 12.711603 minutes
	# save_encoder_path = '../saved_models/model_adversary_cnn_v2.pt'
	# save_discr_path = '../saved_models/model_adversary_discr_v1.pt'







# parameters: 


# from __future__ import print_function
# # parameters for the adversary network


# # cnn (encoder) network params
# cnn_out_channels = 128
# dropout = 0.3
# delta = 1.0
# forward_lr = 0.00001

# save_encoder_path = '../saved_models/model_adversary_cnn_v2.pt'
# load_encoder_path = ''
# eval_encoder_path = save_encoder_path


# # android data params
# andr_path = '../Android-master/'
# andr_dev_pos_path = andr_path + 'dev.pos.txt'
# andr_dev_neg_path = andr_path + 'dev.neg.txt'

# andr_test_pos_path = andr_path + 'test.pos.txt'
# andr_test_neg_path = andr_path + 'test.neg.txt'

# andr_tokens_path = andr_path + 'corpus.tsv'


# # embeddings params
# glove_vecs_path = '../glove_vectors/glove.840B.300d.txt'
# glove_vecs_n_channels = 300


# # discriminator network params
# neg_lr = -1.0 * forward_lr # do not regularize it here, the reg param is applied later
# discr_hidden_size = 200

# save_discr_path = '../saved_models/model_adversary_discr_v1.pt'
# load_discr_path = ''
# eval_discr_path = save_discr_path


# # general training parameters
# encoder_type = 'cnn'
# n_epochs = 1
# batch_size = 5
# lambda_reg = 0.0001 # the regularization parameter for loss2 when computing the total loss
# auc_max_fpr = 0.05


# debug = False
# if debug: print('debug is True')


######################################################################

