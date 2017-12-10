import torch
import numpy as np
from torch.autograd import Variable

"""
For use in converting sequences of words into sequences of their vector representations

Takes in data from ../askubuntu-master/vector/vectors_pruned.200.txt
"""

class PreConv:
    """
    """
    def __init__(self, debug=False):
        self.debug = debug
        self.vectors_path = '../askubuntu-master/vector/vectors_pruned.200.txt'
        self.tokens_path = '../askubuntu-master/text_tokenized.txt'
        self.train_path = '../askubuntu-master/train_random.txt'



    # wrappers for data types
    def to_int_variable(self, data, requires_grad=True):
        return Variable(torch.IntTensor(data), 
            requires_grad=requires_grad)

    def to_float_variable(self, data):
        return Variable(torch.FloatTensor(data))

    # def to_long_variable(data):
    #     return Variable(torch.LongTensor(data))

    def split_into_batches(self, data, batch_size):
        batches = []
        remainder = len(data) % batch_size

        for i in xrange(0, len(data) - remainder, batch_size):
            batch = data[i:i+batch_size]
            batches.append(batch)

        batches[-1] = batches[-1] + data[len(data) - batch_size + 1:]

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

    def get_question_dict(self):
        word_to_vector = self.get_word_to_vector_dict()
        # create the question dictionary
        f = open(self.tokens_path, 'r')

        vocab = set(word_to_vector.keys())
        id_to_question = {}
        for line in f.readlines():
            split = line.strip().split("\t")

            id_num = split[0].strip()
            title = split[1].strip()
            body = split[2].strip() if len(split) == 3 else None

            # convert title and body to array of word vectors
            blank_vec = [0.0] * 200
            title_matrix = [word_to_vector[w] for w in title.split() if w in vocab]
            # pad title with blanks
            len_title = min(len(title_matrix), 100)   
            title_matrix.extend([blank_vec for _ in range(100-len_title)])
            # max sentence length is 100
            title_matrix = title_matrix[:100]
            assert len(title_matrix) == 100

            body_matrix = [word_to_vector[w] for w in body.split() if w in vocab] if body is not None else []
            # pad body with blanks
            len_body = min(len(body_matrix), 100)
            body_matrix.extend([blank_vec for _ in range(100 - len_body)])
            # max sentence length is 100
            body_matrix = body_matrix[:100]
            assert len(body_matrix) == 100

            id_to_question[id_num] = (
                self.to_float_variable(title_matrix),
                self.to_float_variable(body_matrix), 
                float(len_title),
                float(len_body),
            )

        f.close()
        return id_to_question


    def get_candidate_ids(self):
        # create question -> candidates dictionary
        f = open(self.train_path, 'r')

        question_to_candidates = {}
        count = 0
        for line in f.readlines():
            count += 1
            if self.debug and count > 100: break
            split = line.strip().split('\t')

            question = split[0]
            positive = split[1].split(' ')
            negative = split[2].split(' ')

            question_to_candidates[question] = (positive, negative)

        f.close()
        return question_to_candidates
