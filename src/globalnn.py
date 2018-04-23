
from chainer import cuda, Chain, Variable, optimizers, optimizer
import chainer.functions as F
from chainer.functions.array import broadcast
import chainer.links as L
import copy
import dill
from graphviz import Digraph
import numpy as np
import os
import pickle
from pprint import pprint
from scipy.spatial.distance import cosine as cos_sim

from localnn import LocalNN
from state import State


vocab_path = os.path.abspath(os.path.dirname(__file__)) + '/../../data/train.02-21.sent.vocab.pkl'
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)
vocab['<s>'] = len(vocab)
vocab['</s>'] = len(vocab)
vocab['<UNK>'] = len(vocab)


N_ACTION = 45
PAD_ID = -1

np.set_printoptions(threshold=np.inf)

class GlobalNN(object):
    def __init__(self, local_model, beam_width, nchar, pad_tokens, gpu_id=-1, perceptron_flg=False):
        self.local_model = local_model
        self.n_window_word = local_model.window_word
        self.pos_order = local_model.window_pos
        self.beam_width = beam_width
        self.nchar = nchar
        self.pad_tokens = pad_tokens
        self.gpu_id = gpu_id
        self.n_update = 0
        self.perceptron_flg = perceptron_flg
        self.origw_flg = True

        if gpu_id > -1:
            self.xp = cuda.cupy
            self.local_model.to_gpu()
            cuda.get_device(gpu_id).use()
            self.argsort = cupy_topn_idxs
            self.gpu_flg = True
        else:
            self.xp = np
            self.argsort = np_topn_idxs
            self.gpu_flg = False

    def init_optimizer(self, wdecay, method, lr=0.01, momentum=0.9):
        if method == 'momentumSGD':
            self.opt = optimizers.MomentumSGD(lr, momentum)
        elif method == 'Adam':
            self.opt = optimizers.Adam()
        else:
            raise
        self.opt.setup(self.local_model)
        self.opt.add_hook(optimizer.WeightDecay(wdecay))

    # only updating perceptron layer in local model
    def init_perceptron_optimizer(self, wdecay, method, lr=0.01, momentum=0.9):
        if method == 'momentumSGD':
            self.opt = optimizers.MomentumSGD(lr, momentum)
        elif method == 'Adam':
            self.opt = optimizers.Adam()
        else:
            raise
        self.opt.setup(self.local_model.hy2y)
        self.opt.add_hook(optimizer.WeightDecay(wdecay))

    def beam_search(self, sent, gold_poss, train_flg):
        n_word = len(sent)
        n_step = n_word  # In POS tagging, we need (n_word) shift actions
        self.beams = [None] * (n_step + 1)
        self.beams[0] = [State.gen_initstate(sent, self.n_window_word, self.pos_order, self.pad_tokens, self.gpu_flg)]
        # beam search
        gold_rank = 0
        prev_gold_state = self.beams[0][0]
        for i in range(1, n_step+1):
            buf = []
            gold_item = None
            gold_pos = gold_poss[i-1]
            self.cache_local_score(self.beams[i-1])
            state_score_vars = F.concat(tuple([s.score_var for s in self.beams[i-1]]))
            s_a_mat_var = self.action_scores + F.transpose(F.tile(state_score_vars, (N_ACTION, 1)))
            arg_idxs = self.argsort(s_a_mat_var.data, self.beam_width)
            gold_beam_flg = False
            self.beams[i] = []
            for j, idx in enumerate(arg_idxs):
                state_id = int(idx//N_ACTION)
                action_id = int(idx%N_ACTION)
                cur_state = self.beams[i-1][state_id]
                gold_act_flg = action_id == gold_pos
                next_state_var = s_a_mat_var[state_id][action_id]
                next_state = cur_state.take(action_id, next_state_var, gold_act_flg)
                next_state.rank = j
                if state_id == gold_rank and gold_act_flg:
                    gold_beam_flg = True
                    gold_rank = j
                    gold_item = next_state
                self.beams[i].append(next_state)

            if not gold_item:
                gold_state_var = s_a_mat_var[gold_rank][gold_pos]
                gold_item = prev_gold_state.take(gold_pos, gold_state_var, True)

            if (not gold_beam_flg) and train_flg:
                loss = self.cal_crf_loss(self.beams[i], gold_item, early_update_flg=True)
                return loss, i

            prev_gold_state = gold_item

        # In the last beam
        if train_flg:
            loss = self.cal_crf_loss(self.beams[i], gold_item, early_update_flg=False)
            return loss, i
        # in testing or decoding
        else:
            best_state = self.beams[-1][0]
            pred_actions = best_state.all_actions()
            return pred_actions

    def cache_local_score(self, beam):
        n_state = len(beam)
        word_feats, pos_feats = [], []
        # char_feats = [[]] * self.nchar
        char_feats = [[] for _ in range(self.nchar)]
        max_ngrams = [-1] * self.nchar
        for state in beam:
            word_feat, char_feat, pos_feat = state.extract_feature(self.nchar)
            word_feats.append(word_feat)
            pos_feats.append(pos_feat)
            assert len(char_feat) == self.nchar
            for i in range(self.nchar):
                max_val = max(len(ngram) for ngram in char_feat[i])
                if max_val > max_ngrams[i]:
                    max_ngrams[i] = max_val
                char_feats[i].append(char_feat[i])

        # padding
        for i in range(self.nchar):
            max_num = max_ngrams[i]
            # for one_beam in char_feats[i]:
            for j in range(n_state):
                for k in range(self.n_window_word):
                    char_feats[i][j][k] += [PAD_ID] * (max_num-len(char_feats[i][j][k]))

        word_feats = self.xp.array(word_feats, dtype=self.xp.int32)
        char_feats = [self.xp.array(char_feats[i], dtype=self.xp.int32) for i in range(self.nchar)]
        pos_feats = self.xp.array(pos_feats, dtype=self.xp.int32)
        action_scores = self.local_model.decode([word_feats, char_feats, pos_feats], self.perceptron_flg)  # action_scores.shape == (n_state, n_action)
        self.action_scores = action_scores  # type(action_scores) is chainer.Variable
        # self.phis = phis
        # for i, state in enumerate(beam):
        #     # print(i)
        #     state.phi = copy.deepcopy(phis.data[i])


    def cal_crf_loss(self, beam_states, gold_state, early_update_flg):
        if early_update_flg:
            all_states = [gold_state] + beam_states
            gold_idx = 0
        else:
            all_states = beam_states
            gold_idx = gold_state.rank
            assert gold_idx > -1 and gold_idx < self.beam_width
        n_state = len(all_states)
        scores = F.concat(tuple([state.score_var for state in all_states]))
        t = Variable(self.xp.array([gold_idx], dtype=self.xp.int32))

        # for confirm_gradient
        self.probs = copy.deepcopy(F.softmax(scores).data)
        self.gold_idx = gold_idx
        self.gold_state = gold_state

        return F.softmax_cross_entropy(scores, t)

    def update(self, sent, gold_poss):
        loss, n_step = self.beam_search(sent, gold_poss, train_flg=True)
        self.local_model.zerograds()
        loss.backward()
        self.opt.update()
        return loss.data, n_step

    # using weight averaging method (but inefficient)
    def averaged_update(self, sent, gold_poss):
        assert self.origw_flg == True  # chacking whether not using averaged weight
        # prev_params = [copy.deepcopy(p.data) for p in self.local_model.params()]
        self.averaged_weights = [self.xp.zeros_like(p.data) for p in self.local_model.params()]
        loss, n_step = self.beam_search(sent, gold_poss, train_flg=True)
        self.local_model.zerograds()
        loss.backward()
        self.opt.update()
        # cur_params = [copy.deepcopy(p.data) for p in self.local_model.params()]
        for i, p in enumerate(self.local_model.params()):
            self.averaged_weights[i] = (self.n_update / (self.n_update+1)) * averaged_weights[i] + (1 / (self.n_update+1)) * p.data
            # p.data = (self.n_update / (self.n_update + 1))*prev_params[i] + (1 / (self.n_update+1))*cur_params[i]
        self.n_update += 1
        return loss.data, n_step

    def change_to_averagedw(self):
        # saving
        self.orig_weight = [copy.deepcopy(p.data) for p in self.local_model.params()]
        # changing
        for i, p in enumerate(self.local_model.params()):
            p.data = self.averaged_weights[i]
        self.origw_flg = False

    def change_to_origw(self):
        for i, p in enumerate(self.local_model.params()):
            p.data = self.orig_weight[i]
        self.origw_flg = True

    def visualize_lattice(self, out_path=None):
        n_step = max(i for i in range(len(self.beams)) if self.beams[i])
        dot = Digraph()
        dot.graph_attr['rankdir'] = 'LR'
        for i in range(n_step):
            for j, state in enumerate(self.beams[i]):
                s_name = '{}-{}'.format(i, j)
                dot.node(s_name, str(state.prev_action))
        for i in range(1, n_step):
            for j, state in enumerate(self.beams[i]):
                prev_s = state.leftptrs
                prev_s_rank = prev_s.rank if prev_s.rank > -1 else 0
                cur_s_name = '{}-{}'.format(i, j)
                prev_s_name = '{}-{}'.format(i-1, prev_s_rank)
                dot.edge(prev_s_name, cur_s_name)
        dot.render('lattice')
        del dot

    def save(self, model_path):
        if self.gpu_id > -1:
            self.local_model.to_cpu()
            with open(model_path, 'wb') as fw:
                dill.dump(self, fw)
            self.local_model.to_gpu()
        else:
            with open(model_path, 'wb') as fw:
                dill.dump(self, fw)

    @classmethod
    def load(cls, model_path):
        with open(model_path, 'rb') as f:
            globalnn = dill.load(f)
        return globalnn


def np_topn_idxs(array, top_n):
    """
    Args:
        array: 2-dimentional array(ndarray)
    """
    return np.argsort(array, axis=None)[::-1][:top_n]


# maybe inefficient
def cupy_topn_idxs(array, top_n):
    """
    Args:
        array: 2-dimentional array(cupy)
    """
    sort_idxs = []
    _array = copy.deepcopy(array)
    n_state, n_action = _array.shape
    for _ in range(top_n):
        idx = _array.argmax(axis=None)
        _array[int(idx//n_action)][int(idx%n_action)] = - np.inf
        sort_idxs.append(idx)
    return sort_idxs


def test_globalnn():
    dim_wembed = 200
    dim_pembed = 100
    dim_hidden = 500
    window_word = 7
    window_pos = 4
    n_vocab = len(vocab)
    n_pos = 45
    beam_width = 40
    local_model = LocalNN(dim_wembed, dim_pembed, dim_hidden, window_word, window_pos, n_vocab, n_pos)
    global_model = GlobalNN(local_model, beam_width)
    global_model.beam_search_v2([23,4,45,4], [2,1,5,7], True)


if __name__ == '__main__':
    test_globalnn()
