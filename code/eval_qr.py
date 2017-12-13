import torch
from torch.autograd import Variable
from torch.nn import functional as F
import qr_lstm
from qr_lstm import BiDiLSTM
import preprocessing
import time

"""
helper class used for computing information retrieval metrics, including MAP / MRR / and Precision @ x

https://github.com/taolei87/rcnn/blob/master/code/qa/evaluation.py
"""

class Evaluation:

    def __init__(self, data):
        self.data = data

    def Precision(self,precision_at):
        scores = []
        for item in self.data:
            temp = item[:precision_at]
            if any(val==1 for val in item):
                scores.append(sum([1 if val==1 else 0 for val in temp])*1.0 / len(temp) if len(temp) > 0 else 0.0)
        return sum(scores)/len(scores) if len(scores) > 0 else 0.0

    def MAP(self):
        scores = []
        missing_MAP = 0
        for item in self.data:
            temp = []
            count = 0.0
            for i,val in enumerate(item):
                if val == 1:
                    count += 1.0
                    temp.append(count/(i+1))
            if len(temp) > 0:
                scores.append(sum(temp) / len(temp))
            else:
                missing_MAP += 1
        return sum(scores)/len(scores) if len(scores) > 0 else 0.0

    def MRR(self):
        scores = []
        for item in self.data:
            for i,val in enumerate(item):
                if val == 1:
                    scores.append(1.0/(i+1))
                    break

        return sum(scores)/len(scores) if len(scores) > 0 else 0.0


def evaluate(model):
    questions = preprocessing.id_to_question

    eval_dev_data = []
    eval_test_data = []

    dev_ids = sorted(preprocessing.dev_set.keys())
    test_ids = sorted(preprocessing.test_set.keys())

    ##
    ## Dev Set
    ##

    for i in range(len(dev_ids)):
        print(str(i) + '/' + str(len(dev_ids)))
        qr = dev_ids[i]
        # the word matrices for each sequence in the entire batch
        batch_title_data = []
        batch_body_data = []

        # the ordered, true lengths of each sequence before padding
        batch_title_lengths = []
        batch_body_lengths = []

        # masks for title and body
        batch_title_mask = []
        batch_body_mask = []

        # question of interest
        q = questions[qr]
        pos_indices = preprocessing.dev_positive_indices[qr]

        batch_title_data.append(preprocessing.sentence_to_embeddings(q[0]))
        batch_title_lengths.append(1.0 / min(preprocessing.MAX_SEQUENCE_LENGTH, len(q[0])))
        batch_title_mask.append(qr_lstm.get_mask(q[0], qr_lstm.HIDDEN_SIZES[2]))

        batch_body_data.append(preprocessing.sentence_to_embeddings(q[1]))
        batch_body_lengths.append(1.0 / min(preprocessing.MAX_SEQUENCE_LENGTH, len(q[1])))
        batch_body_mask.append(qr_lstm.get_mask(q[1], qr_lstm.HIDDEN_SIZES[2]))

        # negative examples
        for c in preprocessing.dev_set[qr][1]:
            cand = questions[c]

            batch_title_data.append(preprocessing.sentence_to_embeddings(cand[0]))
            batch_title_lengths.append(1.0 / min(preprocessing.MAX_SEQUENCE_LENGTH, len(cand[0])))
            batch_title_mask.append(qr_lstm.get_mask(cand[0], qr_lstm.HIDDEN_SIZES[2]))

            batch_body_data.append(preprocessing.sentence_to_embeddings(cand[1]))
            batch_body_lengths.append(1.0 / min(preprocessing.MAX_SEQUENCE_LENGTH, len(cand[1])))
            batch_body_mask.append(qr_lstm.get_mask(cand[1], qr_lstm.HIDDEN_SIZES[2]))

        # convert batch data and lengths to Variables
        batch_title_data = preprocessing.to_float_variable(batch_title_data)
        batch_body_data = preprocessing.to_float_variable(batch_body_data)
        batch_title_lengths = preprocessing.to_float_variable(batch_title_lengths)
        batch_body_lengths = preprocessing.to_float_variable(batch_body_lengths)
        batch_title_mask = preprocessing.to_float_variable(batch_title_mask)
        batch_body_mask = preprocessing.to_float_variable(batch_body_mask)

        forward_start = time.time()
        title_states, title_out = model(batch_title_data)
        print ("the title forward lstm took ", time.time() - forward_start)

        forward_start = time.time()
        body_states, body_out = model(batch_body_data)
        print ("the body forward lstm took ", time.time() - forward_start)

        ############################################
        ## Re-arrange Data For Cosine Calculation ##
        ############################################

        title_states = title_states * batch_title_mask
        body_states = body_states * batch_body_mask

        # mean pooling of the hidden states of each question's title and body sequences
        title_states = torch.sum(title_states, dim=1, keepdim=False)
        averaged_title_states = title_states * batch_title_lengths.repeat(title_states.size(dim=1), 1).t()

        body_states = torch.sum(body_states, dim=1, keepdim = False)
        averaged_body_states = body_states * batch_body_lengths.repeat(body_states.size(dim=1), 1).t()

        # take the average between the title and body representations for the final representation
        final_question_reps = (averaged_title_states + averaged_body_states).div(2)

        ###############################################
        ## Calculate Cosines and Construct Eval Data ##
        ###############################################

        cosine_scores = F.cosine_similarity(final_question_reps[1:], final_question_reps[0] , -1)
        scores = list(cosine_scores.data)

        sorted_eval = [x for _,x in sorted(zip(scores,pos_indices), reverse=True)]

        eval_dev_data.append(sorted_eval)


    ##
    ## Duplicate code but on test set
    ##

    for i in range(len(test_ids)):
        print(str(i) + '/' + str(len(test_ids)))
        qr = test_ids[i]
        # the word matrices for each sequence in the entire batch
        batch_title_data = []
        batch_body_data = []

        # the ordered, true lengths of each sequence before padding
        batch_title_lengths = []
        batch_body_lengths = []

        # masks for title and body
        batch_title_mask = []
        batch_body_mask = []

        # question of interest
        q = questions[qr]
        pos_indices = preprocessing.test_positive_indices[qr]

        batch_title_data.append(preprocessing.sentence_to_embeddings(q[0]))
        batch_title_lengths.append(1.0 / min(preprocessing.MAX_SEQUENCE_LENGTH, len(q[0])))
        batch_title_mask.append(qr_lstm.get_mask(q[0], qr_lstm.HIDDEN_SIZES[2]))

        batch_body_data.append(preprocessing.sentence_to_embeddings(q[1]))
        batch_body_lengths.append(1.0 / min(preprocessing.MAX_SEQUENCE_LENGTH, len(q[1])))
        batch_body_mask.append(qr_lstm.get_mask(q[1], qr_lstm.HIDDEN_SIZES[2]))

        # negative examples
        for c in preprocessing.test_set[qr][1]:
            cand = questions[c]

            batch_title_data.append(preprocessing.sentence_to_embeddings(cand[0]))
            batch_title_lengths.append(1.0 / min(preprocessing.MAX_SEQUENCE_LENGTH, len(cand[0])))
            batch_title_mask.append(qr_lstm.get_mask(cand[0], qr_lstm.HIDDEN_SIZES[2]))

            batch_body_data.append(preprocessing.sentence_to_embeddings(cand[1]))
            batch_body_lengths.append(1.0 / min(preprocessing.MAX_SEQUENCE_LENGTH, len(cand[1])))
            batch_body_mask.append(qr_lstm.get_mask(cand[1], qr_lstm.HIDDEN_SIZES[2]))

        # convert batch data and lengths to Variables
        batch_title_data = preprocessing.to_float_variable(batch_title_data)
        batch_body_data = preprocessing.to_float_variable(batch_body_data)
        batch_title_lengths = preprocessing.to_float_variable(batch_title_lengths)
        batch_body_lengths = preprocessing.to_float_variable(batch_body_lengths)
        batch_title_mask = preprocessing.to_float_variable(batch_title_mask)
        batch_body_mask = preprocessing.to_float_variable(batch_body_mask)

        forward_start = time.time()
        title_states, title_out = model(batch_title_data)
        print ("the title forward lstm took ", time.time() - forward_start)

        forward_start = time.time()
        body_states, body_out = model(batch_body_data)
        print ("the body forward lstm took ", time.time() - forward_start)

        ############################################
        ## Re-arrange Data For Cosine Calculation ##
        ############################################

        title_states = title_states * batch_title_mask
        body_states = body_states * batch_body_mask

        # mean pooling of the hidden states of each question's title and body sequences
        title_states = torch.sum(title_states, dim=1, keepdim=False)
        averaged_title_states = title_states * batch_title_lengths.repeat(title_states.size(dim=1), 1).t()

        body_states = torch.sum(body_states, dim=1, keepdim = False)
        averaged_body_states = body_states * batch_body_lengths.repeat(body_states.size(dim=1), 1).t()

        # take the average between the title and body representations for the final representation
        final_question_reps = (averaged_title_states + averaged_body_states).div(2)

        ###############################################
        ## Calculate Cosines and Construct Eval Data ##
        ###############################################

        cosine_scores = F.cosine_similarity(final_question_reps[1:], final_question_reps[0] , -1)
        scores = list(cosine_scores.data)

        sorted_eval = [x for _,x in sorted(zip(scores,pos_indices), reverse=True)]

        eval_test_data.append(sorted_eval)

    evaluation_of_dev = Evaluation(eval_dev_data)
    evaluation_of_test = Evaluation(eval_test_data)

    print("\n")
    print("DEV")
    print("\n")
    print("MAP", evaluation_of_dev.MAP())
    print("MRR", evaluation_of_dev.MRR())
    print("P@1", evaluation_of_dev.Precision(1))
    print("P@5", evaluation_of_dev.Precision(5))

    print("\n")
    print("TEST")
    print("\n")
    print("MAP", evaluation_of_test.MAP())
    print("MRR", evaluation_of_test.MRR())
    print("P@1", evaluation_of_test.Precision(1))
    print("P@5", evaluation_of_test.Precision(5))
