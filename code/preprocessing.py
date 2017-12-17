import torch
import numpy as np
from torch.autograd import Variable
import copy
import random

"""
For use in converting sequences of words into sequences of their vector representations

Takes in data from ../askubuntu-master/vector/vectors_pruned.200.txt
"""
# constants
EMPTY_WORD_PAD = [0.0 for i in range(200)]
MAX_SEQUENCE_LENGTH = 100
NEGATIVE_CANDIDATE_SIZE = 20

# wrappers for data types
def to_int_variable(data):
    return Variable(torch.IntTensor(data), requires_grad = True)

def to_float_variable(data):
    return Variable(torch.FloatTensor(data), requires_grad = True)

def to_long_variable(data):
    return Variable(torch.LongTensor(data), requires_grad = True)

def split_into_batches(data, batch_size):
    batches = []
    div = int(len(data) / batch_size)
    rem = len(data) % batch_size

    for i in xrange(0, len(data) - rem, batch_size):
        batch = data[i:i+batch_size]
        batches.append(batch)

    batches[-1] = batches[-1] + data[len(data) - batch_size + 1:]

    return batches

# create the word_to_vector dictionary

f = open('../askubuntu-master/vector/vectors_pruned.200.txt', 'r')
print("word to vector")
word_to_vector = {}
for line in f.readlines():
    split = line.strip().split(" ")
    word = split[0]

    vec = [float(val) for val in split[1:]]
    word_to_vector[word] = vec

f.close()

# create the question dictionary
f = open('../askubuntu-master/text_tokenized.txt', 'r')
print("id to question")
vocab = set(word_to_vector.keys())
id_to_question = {}
for line in f.readlines():
    split = line.strip().split("\t")

    id_num = split[0].strip()
    title = split[1].strip()
    body = split[2].strip() if len(split) == 3 else ""

    id_to_question[id_num] = (title.split(), body.split())

f.close()

# create question -> candidates dictionary
f = open('../askubuntu-master/train_random.txt', 'r')
print("question to candidates")
question_to_candidates = {}
for line in f.readlines():
    split = line.strip().split('\t')

    question = split[0]
    positive = split[1].split()
    negative = split[2].split()

    # negative = random.sample(negative, NEGATIVE_CANDIDATE_SIZE)

    question_to_candidates[question] = (positive, negative)

f.close()

# dev set
f = open('../askubuntu-master/dev.txt', 'r')
print("dev set")
dev_set = {}
dev_positive_indices = {}
for line in f.readlines():
    split = line.strip().split('\t')

    question = split[0]
    positives = split[1].split()
    candidates = split[2].split()
    bm_scores = [float(i) for i in split[3].split()]

    indices = [1 if i in positives else 0 for i in candidates]

    dev_positive_indices[question] = indices
    dev_set[question] = (positives, candidates, bm_scores)

f.close()

# test set
f = open('../askubuntu-master/test.txt', 'r')
print("test set")
test_set = {}
test_positive_indices = {}
for line in f.readlines():
    split = line.strip().split('\t')

    question = split[0]
    positives = split[1].split()
    candidates = split[2].split()
    bm_scores = [float(i) for i in split[3].split()]

    indices = [1 if i in positives else 0 for i in candidates]

    test_positive_indices[question] = indices
    test_set[question] = (positives, candidates, bm_scores)

f.close()

def sentence_to_embeddings(s):
    if len(s) > MAX_SEQUENCE_LENGTH:
        s = s[:MAX_SEQUENCE_LENGTH]

    padded_tail = [copy.deepcopy(EMPTY_WORD_PAD) for i in range(MAX_SEQUENCE_LENGTH - len(s))]
    return [word_to_vector[w] if w in vocab else copy.deepcopy(EMPTY_WORD_PAD) for w in s] + padded_tail
