import torch
import numpy as np
from torch.autograd import Variable

"""
For use in converting sequences of words into sequences of their vector representations

Takes in data from ../askubuntu-master/vector/vectors_pruned.200.txt
"""

# wrappers for data types

def to_float_variable(data):
    return Variable(torch.FloatTensor(data))

def to_long_variable(data):
    return Variable(torch.LongTensor(data))


# create the word_to_vector dictionary

f = open('./askubuntu-master/vector/vectors_pruned.200.txt', 'r')

word_to_vector = {}
for line in f.readlines():
    split = line.strip().split(" ")
    word = split[0]

    vec = np.array([float(val) for val in split[1:]])
    word_to_vector[word] = vec

f.close()

# create the question dictionary
f = open('./askubuntu-master/text_tokenized.txt', 'r')

id_to_question = {}
for line in f.readlines():
    split = line.strip().split("\t")

    id_num = split[0].strip()
    title = split[1].strip()
    body = split[2].strip()

    # convert title and body to array of word vectors
    title_matrix = [word_to_vector[w] for w in title.split()]
    body_matrix = [word_to_vector[w] for w in body.split]

    id_to_question[id_num] = (np.array(title_matrix), np.array(body_matrix))

f.close()
