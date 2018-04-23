
from chainer import cuda, Variable
import chainer.functions as F
import copy
import numpy as np
import os
import pickle
# from localnn import LocalNN


vocab_path = os.path.abspath(os.path.dirname(__file__)) + '/../../data/train.02-21.sent.vocab.pkl'
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)
vocab['<s>'] = len(vocab)
vocab['</s>'] = len(vocab)
vocab['<UNK>'] = len(vocab)

# CAUTIONS!!!!!!!
SOS_TOKEN_ID = vocab['<s>']
EOS_TOKEN_ID = vocab['</s>']
INIT_TAG_ID = 45
END_TAG_ID = 46

N_ACTION = 45


class State(object):

    __slots__ = ['eos_token', 'gold_flg', 'leftptrs', 'local_scores', 'n_window_word', 'phi', 'pos_order', 'prev_action', 'rank',  'score_var', 'sent', 'sos_token', 'stack', 'stack_label', 't_step', 'queue']

    def __init__(self, t_step):
        self.t_step = t_step
        self.rank = -1

    def __lt__(self, other):
        return self.score_var.data[0][0] > other.score_var.data[0][0]

    @staticmethod
    def gen_initstate(sent, n_window_word, pos_order, pad_tokens, gpu_flg):
        if gpu_flg:
            xp = cuda.cupy
        else:
            xp = np
        state = State(t_step=0)
        state.sent = sent
        state.gold_flg = True
        state.score_var = Variable(xp.array([[0.]], dtype=xp.float32))  # is it OK ? because I assume this variable is Chainer object
        assert n_window_word % 2 == 1, "n_window_word must be odd : {}".format(n_window_word)
        state.n_window_word = n_window_word
        state.pos_order = pos_order
        state.sos_token = pad_tokens[0]
        state.eos_token = pad_tokens[1]
        state.leftptrs = None
        state.prev_action = None
        state.stack = [state.sos_token for _ in range(state.n_window_word // 2)]
        state.queue = [sent[i] if len(sent) > i else state.eos_token for i in range(state.n_window_word // 2 + 1)]
        state.stack_label = [INIT_TAG_ID for _ in range(state.pos_order)]
        return state

    def shift(self, action_id):
        new_state = State(self.t_step+1)
        new_state.sent = self.sent  # in Liang parser, no this line...
        new_state.n_window_word = self.n_window_word
        new_state.pos_order = self.pos_order
        new_state.eos_token = self.eos_token
        new_state.stack = self.stack[1:] + [self.sent[self.t_step]]
        # new_state.queue = self.queue[1:] + [self.sent[self.n_window_word // 2 + self.t_step + 1]]
        next_queue_idx = self.n_window_word // 2 + self.t_step + 1
        new_state.queue = self.queue[1:] + [self.sent[next_queue_idx] if len(self.sent) > next_queue_idx else new_state.eos_token]
        new_state.stack_label = self.stack_label[1:] + [action_id]
        return new_state

    def take(self, action_id, next_score_var, gold_act_flg):
        new_state = self.shift(action_id)  # shift and tagging
        new_state.score_var = F.reshape(next_score_var, (1, 1))
        new_state.leftptrs = self  # In usual Globally Normalized method, each state has only one pointer
        new_state.prev_action = action_id
        new_state.gold_flg = self.gold_flg and gold_act_flg
        return new_state

    def all_actions(self):
        actions = []
        item = self
        # backward
        while item.leftptrs:
            actions.append(item.prev_action)
            item = item.leftptrs
        actions.reverse()
        return actions

    def extract_feature(self, nchar):
        cur_tokens = self.stack + self.queue
        word_feat = [token.id for token in cur_tokens]
        char_feat = [[copy.deepcopy(token.char_ngram_ids[i]) for token in cur_tokens] for i in range(nchar)]
        return word_feat, char_feat, self.stack_label

    def print_info(self):
        print('<STATE : {}>'.format(self))
        print('t_step: {}'.format(self.t_step))
        print('score_var.data: {}'.format(self.score_var.data))
        print('gold_flg: {}'.format(self.gold_flg))
        print('prev_action: {}'.format(self.prev_action))
        print('rank : {}'.format(self.rank))
        print('stack: {}'.format(self.stack))
        print('queue: {}'.format(self.queue))
        print('stack_label: {}'.format(self.stack_label))
        print('leftptrs: {}'.format(self.leftptrs))
