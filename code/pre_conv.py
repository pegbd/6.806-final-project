from __future__ import print_function
import torch
import numpy as np
from torch.autograd import Variable
import random
import string

"""
For use in converting sequences of words into sequences of their vector representations

Takes in data from ../askubuntu-master/vector/vectors_pruned.200.txt
"""

class PreConv:
    """
    """
    def __init__(self, data_type='train', debug=False, vectors_path='', emb_channels=200):
        """
        data_type: must be one of: 'train', 'dev', or 'test'
            is 'train' by default.
        """
        self.data_type = data_type
        self.debug = debug
        self.vectors_path = vectors_path
        self.emb_channels = emb_channels

        if self.vectors_path == '':
            self.vectors_path = '../askubuntu-master/vector/vectors_pruned.200.txt'
        else: self.vectors_path = vectors_path
        self.tokens_path = '../askubuntu-master/text_tokenized.txt'
        if data_type == 'train':
            self.data_path = '../askubuntu-master/%s_random.txt'%(data_type)
        else:
            self.data_path = '../askubuntu-master/%s.txt'%(data_type)


        self.blank_vec = [0.0] * self.emb_channels
        self.word_to_vector = self.get_word_to_vector_dict()
        self.vocab = set(self.word_to_vector.keys())



    # wrappers for data types
    def to_int_variable(self, data, requires_grad=True):
        return Variable(torch.IntTensor(data), 
            requires_grad=requires_grad)

    def to_float_variable(self, data):
        return Variable(torch.FloatTensor(data))

    # def to_long_variable(data):
    #     return Variable(torch.LongTensor(data))


    def split_into_batches(self, data, batch_size):
        actual_batch_size = batch_size
        batches = []
        div = int(len(data) / batch_size)
        rem = len(data) % batch_size

        for i in xrange(0, len(data) - rem, batch_size):
            batch = data[i:i+batch_size]
            assert len(batch) == actual_batch_size
            batches.append(batch)

        # batches[-1] = batches[-1] + data[len(data) - batch_size + 1:]

        return batches


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

            word_to_vector[word] = vec

        f.close()
        return word_to_vector

    def get_question_dict(self, force_lowercase=True):
        f = open(self.tokens_path, 'r')
        
        id_to_question = {}
        for line in f.readlines():
            split = line.strip().split("\t")

            id_num = split[0].strip()
            title = split[1].strip()
            body = split[2].strip() if len(split) == 3 else ''

            if force_lowercase:
                id_to_question[id_num] = (title.lower(), body.lower())
            else:
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

    def get_seq_len(self, seq, max_seq_len=100):
        len_seq = min(len([0 for w in seq.split() if w in self.vocab]), 100)   
        return len_seq


    def get_candidate_ids(self):
        # create question -> candidates dictionary
        f = open(self.data_path, 'r')

        question_to_candidates = {}
        count = 0
        for line in f.readlines():
            count += 1
            if self.debug and count > 100: break
            split = line.strip().split('\t')

            question = split[0]
            positive = split[1].split()
            
            if self.data_type == 'train':
                negative = random.sample(split[2].split(' '), 20)
                assert len(negative) == 20
                question_to_candidates[question] = (positive, negative)
            
            else:
                candidates = split[2].split()
                bm_scores = [float(i) for i in split[3].split()]
                question_to_candidates[question] = (positive, candidates, bm_scores)

        f.close()
        return question_to_candidates


    def get_pos_indicies_all(self):
        f = open('../askubuntu-master/%s.txt'%(self.data_type), 'r')
        dev_set = {}
        positive_indices = {}
        for line in f.readlines():
            split = line.strip().split('\t')

            question = split[0]
            positives = split[1].split()
            candidates = split[2].split()

            indices = [1 if i in positives else 0 for i in candidates]

            positive_indices[question] = indices

        f.close()
        return positive_indices



