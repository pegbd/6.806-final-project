from pre_android import PreAndroid
from collections import defaultdict
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from meter import AUCMeter

pre_and = PreAndroid()

bag_of_words = pre_and.get_corpus_vocabulary()
print bag_of_words

questions = pre_and.get_question_dict()

dev_pos_pairs = pre_and.get_id_pairs(is_pos=True, dev_or_test='dev')
dev_neg_pairs = pre_and.get_id_pairs(is_pos=False, dev_or_test='dev')
test_pos_pairs = pre_and.get_id_pairs(is_pos=True, dev_or_test='test')
test_neg_pairs = pre_and.get_id_pairs(is_pos=False, dev_or_test='test')

dev_instances = defaultdict(lambda: [[],[]])
test_instances = defaultdict(lambda: [[],[]])

auc_evaluator = AUCMeter()

def seq_to_bow(seq):
    bow = [0] * len(bag_of_words)

    for w in seq.split():
        index = bag_of_words[w.lower()]
        bow[index] += 1

    return np.array(bow)

####################################################################
########################## Instance Dicts ##########################
####################################################################

for pair in dev_pos_pairs:
    key = pair[0]
    val = pair[1]

    dev_instances[key][0].append(val)

for pair in dev_neg_pairs:
    key = pair[0]
    val = pair[1]

    dev_instances[key][1].append(val)

for pair in test_pos_pairs:
    key = pair[0]
    val = pair[1]

    test_instances[key][0].append(val)

for pair in test_neg_pairs:
    key = pair[0]
    val = pair[1]

    dev_instances[key][1].append(val)

####################################################################
########################### Bag of Words ###########################
####################################################################

for instance in dev_instances.keys():

    batch = []
    targets = []
    q_title, q_body = questions[instance]

    # string -> bag of words lsit
    q_title = seq_to_bow(q_title)
    q_body = seq_to_bow(q_body)

    # take the average of the body and title
    avg = (q_title + q_body)/2.0

    batch.append(avg)

    # positive and negatives
    pairs = dev_instances[instance]

    # print "POSITIVE PAIRS"
    # print pairs[0]

    for p in pairs[0]:
        p_title, p_body = questions[p]

        # string -> bag of words lsit
        p_title = seq_to_bow(p_title)
        p_body = seq_to_bow(p_body)

        # take the average of the body and title
        p_avg = (p_title + p_body)/2.0

        batch.append(p_avg)
        targets.append(1)

    for n in pairs[1]:
        n_title, n_body = questions[n]

        # string -> bag of words lsit
        n_title = seq_to_bow(n_title)
        n_body = seq_to_bow(n_body)

        # take the average of the body and title
        n_avg = (n_title + n_body)/2.0

        batch.append(n_avg)
        targets.append(0)

    batch = Variable(torch.FloatTensor(batch))
    targets = np.array(targets)

    # print batch
    print targets.shape

    # print targets

    cosine_scores = F.cosine_similarity(batch[1:], batch[0] , -1)

    auc_evaluator.add(cosine_scores.data.numpy(), targets)

print "\n"
print "AUC VALUE = "
print auc_evaluator.value()
