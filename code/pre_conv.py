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
    def __init__(self):
        self.vectors_path = '../askubuntu-master/vector/vectors_pruned.200.txt'
        self.tokens_path = '../askubuntu-master/text_tokenized.txt'
        self.train_path = '../askubuntu-master/train_random.txt'



    # wrappers for data types
    # def to_int_variable(data):
    #     return Variable(torch.IntTensor(data))

    def to_float_variable(data):
        return Variable(torch.FloatTensor(data))

    # def to_long_variable(data):
    #     return Variable(torch.LongTensor(data))

    def split_into_batches(data, batch_size):
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
        for line in f.readlines():
            split = line.strip().split(" ")
            word = split[0]

            vec = [float(val) for val in split[1:]]
            word_to_vector[word] = vec

        f.close()
        return word_to_vector

    def get_question_dict(self):
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
            title_matrix = [word_to_vector[w] for w in title.split() if w in vocab]
            body_matrix = [word_to_vector[w] for w in body.split() if w in vocab] if body is not None else []

            id_to_question[id_num] = (to_float_variable(title_matrix), to_float_variable(body_matrix))

        f.close()
        return id_to_question


    def get_candidate_ids(self):
        # create question -> candidates dictionary
        f = open(self.train_path, 'r')

        question_to_candidates = {}
        for line in f.readlines():
            split = line.strip().split('\t')

            question = split[0]
            positive = split[1].split(' ')
            negative = split[2].split(' ')

            question_to_candidates[question] = (positive, negative)

        f.close()
        return question_to_candidates
