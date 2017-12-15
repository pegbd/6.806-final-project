from future import print_function
import torch
import numpy as np
from torch.autograd import Variable
import random

import adversary_params as params


class PreAndroid:
    def __init__(self, debug=False):
        self.debug = debug

        self.vectors_path = params.glove_vecs_path
        self.tokens_path = params.andr_tokens_path

        self.blank_vec = [0.0] * TODO# TODO TODO --> GET GLOVE EMB SIZE HERE 
        self.word_to_vector = self.get_word_to_vector_dict()
        self.vocab = set(self.word_to_vector.keys())


    def split_into_batches(self, pos_id_pairs, neg_id_pairs, batch_size):
        TODO --> make sure this batch has the same number of questions as the
        ubuntu batch
        pass


    def get_word_to_vector_dict(self): 
        # create the word_to_vector dictionary

        f = open(self.vectors_path, 'r')

        word_to_vector = {}
        count=0
        for line in f.readlines():
            count += 1
            if self.debug and count > 1000: break
            split = line.strip().split(" ")
            word = split[0]

            vec = [float(val) for val in split[1:]]
            print('vector length is')
            print(len(vec))
            assert False

            word_to_vector[word] = vec

        f.close()
        return word_to_vector

    def get_data_path(self, is_pos, dev_or_test):
        if is_pos and dev_or_test == 'dev': return params.andr_dev_pos_path
        elif is_pos and dev_or_test =='test': return params.andr_test_pos_path
        elif not is_pos and dev_or_test == 'dev': return params.andr_dev_neg_path
        elif not is_pos and dev_or_test == 'test': return params.andr_test_neg_path
        return

    def get_id_pairs(self, is_pos, dev_or_test):
        """ is_pos: boolean
            dev_or_test: string, can either be 'dev' or 'test'
        """
        data_path = self.get_data_path(is_pos, dev_or_test)

        f = open(data_path, 'r')
        id_pairs = [tuple(line.strip().split()) for line in f.readlines()]
        f.close()
        return id_pairs

    def get_all_id_pairs(self):
        dev_pos = get_id_pairs(self, is_pos=True, dev_or_test='dev')
        dev_neg = get_id_pairs(self, is_pos=False, dev_or_test='dev')
        test_pos = get_id_pairs(self, is_pos=True, dev_or_test='test')
        test_neg = get_id_pairs(self, is_pos=False, dev_or_test='test')

        id_pairs = dev_pos + dev_neg + test_pos + test_neg
        random.shuffle(id_pairs)
        return id_pairs


    def get_question_dict(self):
        f = open(self.tokens_path, 'r')
        
        id_to_question = {}
        for line in f.readlines():
            split = line.strip().split("\t")

            id_num = split[0].strip()
            title = split[1].strip()
            body = split[2].strip() if len(split) == 3 else None

            id_to_question[id_num] = (title, body)

        f.close()
        return id_to_question


    def sequence_to_vec(self, seq, max_seq_len=100):
        vec = [self.word_to_vector[w] for w in seq.split() if w in self.vocab]
        # pad title with blanks
        len_seq = min(len(vec), 100)
        vec.extend([self.blank_vec for _ in range(max_seq_len - len_seq)])
        # asserting the max sequence length
        vec = vec[:max_seq_len]
        vec = Variable(torch.FloatTensor(vec), requires_grad=False)
        return vec