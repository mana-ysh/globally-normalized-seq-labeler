
import pickle
import numpy as np

from chainer import Chain, Variable
import chainer.functions as F
import chainer.links as L

N_ACTION = 45


class LocalNN(Chain):
    def __init__(self, dim_wembed, dim_cembed, dim_pembed, dim_hidden, window_word, window_pos, n_vocab, n_pos, n_char_ngrams, gpu_id=-1):
        self.n_action = n_pos
        assert window_word%2 == 1  # asumming left and right order is equal in word features
        self.window_word = window_word
        self.window_pos = window_pos
        self.nchar = len(n_char_ngrams)
        self.gpu_id = gpu_id
        self.dim_all_wembed = dim_wembed * window_word
        self.dim_all_pembed = dim_pembed * window_pos
        self.dim_all_cembed = dim_cembed * self.nchar * window_word
        self.dim_hidden = dim_hidden
        dim_embed = self.dim_all_wembed + self.dim_all_pembed + self.dim_all_cembed
        super(LocalNN, self).__init__(
            word2embed = L.EmbedID(n_vocab, dim_wembed),
            pos2embed = L.EmbedID(n_pos+2, dim_pembed),
            embed2h = L.Linear(dim_embed, dim_hidden),
            h2y = L.Linear(dim_hidden, self.n_action)
        )
        self.char_embedlinks = [('char_{}gram2embed'.format(i+1),
                          L.EmbedID(n_char_ngrams[i], dim_cembed, ignore_label=-1)) for i in range(self.nchar)]
        for name, link in self.char_embedlinks:
            self.add_link(name, link)

    def add_perceptron_layer(self):
        """
        input: concatenation of hidden and output layer
        output: score distribution of actions
        """
        try:
            self.perceptron_link  = ('hy2y', L.Linear(self.dim_hidden+self.n_action, self.n_action))
        except AttributeError:
            n_action, dim_hidden = self.h2y.W.data.shape
            self.perceptron_link  = ('hy2y', L.Linear(dim_hidden+n_action, n_action))
        self.add_link(*self.perceptron_link)

    def cal_local_loss(self, batch_features, batch_ts):
        ts = Variable(batch_ts)
        ys = self.decode(batch_features)
        loss = F.softmax_cross_entropy(ys, ts)
        return loss, ys

    # calcurating loss in each sentence (not each state)
    def cal_local_loss2(self, gold_feats, pos_seq):
        n_word = len(pos_seq)
        # t_seq = Variable(pos_seq)
        y_seq = self.decode(gold_feats)
        # loss = F.sum(F.softmax_cross_entropy(y_seq, t_seq))
        loss = 0
        l_softmax = F.log_softmax(y_seq)
        for i in range(n_word):
            loss -= l_softmax[i][pos_seq[i]]
        return loss, y_seq

    def decode(self, batch_features, perceptron_flg=False):
        batch_words, batch_chars, batch_poss = batch_features
        assert batch_words.shape[0] == batch_poss.shape[0]
        assert batch_words.shape[0] == batch_chars[0].shape[0]
        assert batch_words.shape[1] == batch_chars[0].shape[1]
        # assert len(batch_chars) == self.nchar
        batchsize = batch_words.shape[0]
        wembeds = self.word2embed(Variable(batch_words))
        pembeds = self.pos2embed(Variable(batch_poss))
        wembeds = F.reshape(wembeds, (batchsize, self.dim_all_wembed))
        pembeds = F.reshape(pembeds, (batchsize, self.dim_all_pembed))
        cembeds = []
        for i in range(self.nchar):
            cembed = F.sum(self.char_embedlinks[i][1](batch_chars[i]), axis=2)  # without normalization ?
            cembeds.append(cembed)
        cembeds = F.reshape(F.concat(tuple(cembeds)), (batchsize, self.dim_all_cembed))
        hs = self.embed2h(F.concat((wembeds, pembeds, cembeds)))
        ys = self.h2y(F.tanh(hs))
        if not perceptron_flg:
            return ys
        else:
            in_perceptron = F.concat((hs, ys))
            perceptron_ys = self.perceptron_link[1](in_perceptron)
            return perceptron_ys

    def perceptron_decode(self, batch_features):
        batch_words, batch_chars, batch_poss = batch_features
        assert batch_words.shape[0] == batch_poss.shape[0]
        assert batch_words.shape[0] == batch_chars[0].shape[0]
        assert batch_words.shape[1] == batch_chars[0].shape[1]
        # assert len(batch_chars) == self.nchar
        batchsize = batch_words.shape[0]
        wembeds = self.word2embed(Variable(batch_words))
        pembeds = self.pos2embed(Variable(batch_poss))
        wembeds = F.reshape(wembeds, (batchsize, self.dim_all_wembed))
        pembeds = F.reshape(pembeds, (batchsize, self.dim_all_pembed))
        cembeds = []
        for i in range(self.nchar):
            cembed = F.sum(self.char_embedlinks[i][1](batch_chars[i]), axis=2)  # without normalization ?
            cembeds.append(cembed)
        cembeds = F.reshape(F.concat(tuple(cembeds)), (batchsize, self.dim_all_cembed))
        hs = self.embed2h(F.concat((wembeds, pembeds, cembeds)))
        ys = self.h2y(F.tanh(hs))
        in_perceptron = F.concat((hs, ys))
        perceptron_ys = self.perceptron_link[1](in_perceptron)
        return perceptron_ys

    def decode_with_phi(self, batch_features):
        batch_words, batch_chars, batch_poss = batch_features
        assert batch_words.shape[0] == batch_poss.shape[0]
        assert batch_words.shape[0] == batch_chars[0].shape[0]
        assert batch_words.shape[1] == batch_chars[0].shape[1]
        # assert len(batch_chars) == self.nchar
        batchsize = batch_words.shape[0]
        wembeds = self.word2embed(Variable(batch_words))
        pembeds = self.pos2embed(Variable(batch_poss))
        wembeds = F.reshape(wembeds, (batchsize, self.dim_all_wembed))
        pembeds = F.reshape(pembeds, (batchsize, self.dim_all_pembed))
        cembeds = []
        for i in range(self.nchar):
            cembed = F.sum(self.char_embedlinks[i][1](batch_chars[i]), axis=2)  # without normalization ?
            cembeds.append(cembed)
        cembeds = F.reshape(F.concat(tuple(cembeds)), (batchsize, self.dim_all_cembed))
        hs = self.embed2h(F.concat((wembeds, pembeds, cembeds)))
        hs = F.tanh(hs)
        ys = self.h2y(hs)
        return ys, hs

    def get_phi(batch_features):
        batch_words, batch_chars, batch_poss = batch_features
        assert batch_words.shape[0] == batch_poss.shape[0]
        assert batch_words.shape[0] == batch_chars[0].shape[0]
        assert batch_words.shape[1] == batch_chars[0].shape[1]
        # assert len(batch_chars) == self.nchar
        batchsize = batch_words.shape[0]
        wembeds = self.word2embed(Variable(batch_words))
        pembeds = self.pos2embed(Variable(batch_poss))
        wembeds = F.reshape(wembeds, (batchsize, self.dim_all_wembed))
        pembeds = F.reshape(pembeds, (batchsize, self.dim_all_pembed))
        cembeds = []
        for i in range(self.nchar):
            cembed = F.sum(self.char_embedlinks[i][1](batch_chars[i]), axis=2)  # without normalization ?
            cembeds.append(cembed)
        cembeds = F.reshape(F.concat(tuple(cembeds)), (batchsize, self.dim_all_cembed))
        hs = self.embed2h(F.concat((wembeds, pembeds, cembeds)))
        return F.tanh(hs)

    def make_oov_vector(self):
        oov_embed = np.array(np.sum(self.word2embed.W.data, axis=0) / len(self.word2embed.W.data), dtype=np.float32)
        self.word2embed.W.data[-1] = oov_embed

    def save(self, model_path):
        if self.gpu_id > -1:
            self.to_cpu()
            with open(model_path, 'wb') as fw:
                pickle.dump(self, fw)
            self.to_gpu()
        else:
            with open(model_path, 'wb') as fw:
                pickle.dump(self, fw)

    @classmethod
    def load(cls, model_path):
        with open(model_path, 'rb') as f:
            localnn = pickle.load(f)
        return localnn
