from __future__ import print_function
import torch
from torch.nn import functional as F
from torch.autograd import Variable
import random
from meter import AUCMeter

from pre_android import PreAndroid
from adversary_trainer import AdversaryTrainer

import adversary_params as params





class AdversaryEvaluator(AdversaryTrainer):

	def __init__(self):
		# do not want to load parameters until we actually call evaluate
		self.encoder_net = None
		self.discr_input_size = None
		self.discr_net = None
		self.an_preprocessor = None
		self.questions = None

	def reset_params(self):
		self.encoder_net = torch.load(params.eval_encoder_path)
		self.discr_input_size = params.cnn_out_channels
		self.discr_net = torch.load(params.eval_discr_path)
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

			ids_batches_left = ids_batches_pair[0]
			ids_batches_right = ids_batches_pair[1]

			for i in xrange(len(ids_batches_left)):
				ids_batch_left = ids_batches_left[i]
				ids_batch_right = ids_batches_right[i]

				feats_left = self.get_output(ids_batch_left)
				feats_right = self.get_output(ids_batch_right)

				preds = self.get_cosine_scores(feats_left, feats_right)
				# targets = Variable(torch.LongTensor(torch.ones(len(preds) * i_target))) # 0s if neg, 1s if pos
				targets = torch.ones(len(preds)) * i_target # 0s if neg, 1s if pos
				print(targets.size())
				print(len(targets))
				auc_meter.add(preds, targets)

			i_target += 1
			assert i_target < 2

		print('lv2')
		# #remove 3 lines below
		# # # get all the features organized into the corresponding pairs
		# # pos_feats_pair = out_features[0], out_features[1]
		# # neg_feats_pair = out_features[2], out_features[3]

		# # now 'classify' the features
		# # the positive pairs will have a target of 1
		# # the negative features will have a target of 0
		# pairs = (neg_feats_pair, pos_feats_pair) # purposely negative then positive
		# for i_feats_pair in range(len(pairs)):
		# 	feats_pair = pairs[i_feats_pair]
		# 	for i_feats in xrange(len(feats_pair[0])):
		# 		feats_left = feats_left[i_feats]
		# 		feats_right = feats_right[i_feats]

		# 		preds = self.get_cosine_scores(feats_left, feats_right)
		# 		targets = Variable(torch.LongTensor(Torch.ones(len(preds) * i_feats_pair))) # 0s if neg, 1s if pos

		# 		auc_meter.add(preds, targets)
		print('lv3')
		# all predictions are added
		# now get the AUC value
		auc_value = auc_meter.value(params.auc_max_fpr)
		print('AUC(%f) value for %s  =  %f'
			%(params.auc_max_fpr, dev_or_test, auc_value))










if __name__ == '__main__':
	torch.manual_seed(1)
	random.seed(1)

	evaluator = AdversaryEvaluator()
	evaluator.evaluate('dev')
	evaluator.evaluate('test')

