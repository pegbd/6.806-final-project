from __future__ import print_function
import torch
import numpy as np
from torch.autograd import Variable
import random

import string

import preprocessing

import adversary_params as params

class PreAndroid:
    def __init__(self, debug=False):
        self.debug = debug

        self.vectors_path = params.glove_vecs_path
        self.tokens_path = params.andr_tokens_path

        self.all_seen_words = set(self.get_corpus_vocabulary().keys()).union(self.get_all_ubuntu_words())

        self.blank_vec = [0.0] * params.glove_vecs_n_channels
        self.word_to_vector = self.get_word_to_vector_dict()
        self.vocab = set(self.word_to_vector.keys())

    def split_into_batches(self, id_pairs):
        actual_batch_size = params.batch_size * 22
        actual_batch_size /= 2 # because each elem in id_pairs has 2 questions

        div = int(len(id_pairs) / actual_batch_size)
        rem = len(id_pairs) % actual_batch_size

        batches = []
        for i in xrange(0, len(id_pairs) - rem, actual_batch_size):
            pairs_batch  = id_pairs[i:i+actual_batch_size]
            elem0_batch = [elem[0] for elem in pairs_batch]
            elem1_batch = [elem[1] for elem in pairs_batch]

            batch = elem0_batch + elem1_batch
            batches.append(batch)

            assert len(batch) == actual_batch_size * 2

        # batches[-1] = batches[-1] + data[len(data) - batch_size + 1:]

        return batches

    def eval_split_into_batches(self, is_pos, dev_or_test):
        id_pairs = self.get_id_pairs(is_pos, dev_or_test)

        actual_batch_size = params.batch_size * 22
        actual_batch_size /= 2 # because each elem in id_pairs has 2 questions

        div = int(len(id_pairs) / actual_batch_size)
        rem = len(id_pairs) % actual_batch_size

        left_batches, right_batches = [], []
        for i in xrange(0, len(id_pairs) - rem, actual_batch_size):
            pairs_batch  = id_pairs[i:i+actual_batch_size]
            elem0_batch = [elem[0] for elem in pairs_batch]
            elem1_batch = [elem[1] for elem in pairs_batch]

            left_batches.append(elem0_batch)
            right_batches.append(elem1_batch)

            # assert len(batch) == actual_batch_size * 2

        # batches[-1] = batches[-1] + data[len(data) - batch_size + 1:]
        # instead _
        # append the remainder as a separate batch
        pairs_batch  = id_pairs[len(id_pairs) - actual_batch_size + 1:]
        elem0_batch = [elem[0] for elem in pairs_batch]
        elem1_batch = [elem[1] for elem in pairs_batch]
        left_batches.append(elem0_batch)
        right_batches.append(elem1_batch)

        return left_batches, right_batches

    def get_corpus_vocabulary(self):
        """
        Reads the entire corpus.tsv file and generates a list of all the apparent words.
        Will be used for bag_of_words implementation.
        """

        f = open(self.tokens_path, 'r')
        vocab = set()
        d = {}
        for line in f.readlines():
            split = line.strip().split("\t")

            title = split[1].strip()
            body = split[2].strip() if len(split) == 3 else None

            for w in title.split():
                vocab.add(w.lower())

            if body:
                for w in body.split():
                    vocab.add(w.lower())


        vocab = sorted(list(vocab))

        for i in range(len(vocab)):
            word = vocab[i]
            d[word] = i

        return d

    def get_word_to_vector_dict(self):
        # create the word_to_vector dictionary

        f = open(self.vectors_path, 'r')

        word_to_vector = {}
        count=0
        for line in f.readlines():

            if count % 100 == 0:
                print (count)

            count += 1
            if self.debug and count > 1000: break
            split = line.strip().split(" ")
            word = split[0]

            if word.lower() in self.all_seen_words:
                vec = [float(val) for val in split[1:]]

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
        un_split_pairs = [pair.strip().split() for pair in f.readlines()]
        id_pairs = [(pair[0], pair[1]) for pair in un_split_pairs]
        f.close()
        return id_pairs

    def get_all_id_pairs(self):
        dev_pos = self.get_id_pairs(is_pos=True, dev_or_test='dev')
        dev_neg = self.get_id_pairs(is_pos=False, dev_or_test='dev')
        test_pos = self.get_id_pairs(is_pos=True, dev_or_test='test')
        test_neg = self.get_id_pairs(is_pos=False, dev_or_test='test')

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

            id_to_question[id_num] = (title.lower(), body.lower())

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

    # required for LSTM
    def list_to_vec(self, list, max_seq_len=100):
        vec = [self.word_to_vector[w] for w in list if w in self.vocab]
        # pad title with blanks
        len_seq = min(len(vec), 100)
        vec.extend([self.blank_vec for _ in range(max_seq_len - len_seq)])
        # asserting the max sequence length
        vec = vec[:max_seq_len]
        return vec


    def get_seq_len(self, seq, max_seq_len=100):
        len_seq = min(len([0 for w in seq.split() if w in self.vocab]), 100)
        return len_seq

    def get_all_ubuntu_words(self):
        # create the question dictionary
        f = open('../askubuntu-master/text_tokenized.txt', 'r')
        all_ubuntu_words = set()
        for line in f.readlines():
            split = line.strip().split("\t")
            title = split[1].strip()
            body = split[2].strip() if len(split) == 3 else ""

            for w in title: all_ubuntu_words.add(w.lower())
            for w in body: all_ubuntu_words.add(w.lower())

        f.close()

        return all_ubuntu_words
