
import linecache
import numpy as np
import re


SOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
UNK_TOKEN = '<UNK>'
NUM_TOKEN = '<NUM>'

SOW_CHAR = '<SOW>'
EOW_CHAR = '<EOW>'

NUMBER_MATCHER = r'[0-9]'


class Token(object):
    def __init__(self, word, word_id):
        self.surface = word  # basically not be preprocessed word
        self.id = word_id

    def set_char_ngrams(self, char_ngram_surfaces, char_ngram_ids):
        self.char_ngram_surfaces = char_ngram_surfaces
        self.char_ngram_ids = char_ngram_ids

    def set_cluster_id(self, cluster_id):
        self.cluster_id = cluster_id

    def print_info(self):
        print('surface: {}'.format(self.surface))
        print('id: {}'.format(self.id))
        print('char_ngram_surfaces: {}'.format(self.char_ngram_surfaces))
        print('char_ngram_ids: {}'.format(self.char_ngram_ids))


# managing token or character n-gram id
class TokenManager(object):
    def __init__(self, word_vocab, char_vocabs, word_lower_flg=False, num_flg=False):
        self.word_vocab = word_vocab
        self.char_vocabs = char_vocabs
        self.word_lower_flg = word_lower_flg
        self.num_flg = num_flg
        self.n = len(char_vocabs)
        self.surface2token = {}
        self.initialize()

    def gen_token(self, raw_word):
        prep_word = self.preprocess(raw_word)
        word_id = self.word_vocab[prep_word] if prep_word in self.word_vocab else self.word_vocab.unk_id
        token = Token(raw_word, word_id)
        n_char = len(raw_word)
        char_ngram_ids = []
        char_ngram_surfaces = []
        for i in range(self.n):
            char_ngram_id = []
            char_ngram_surface = []
            v = self.char_vocabs[i]
            char_seq = ([SOW_CHAR] * i) + list(raw_word) + ([EOW_CHAR] * i)
            for j in range(n_char+i):
                chars = ''.join(char_seq[j:j+i+1])
                chars_id = v[chars] if chars in v else v.unk_id
                char_ngram_surface.append(chars)
                char_ngram_id.append(chars_id)
            char_ngram_surfaces.append(char_ngram_surface)
            char_ngram_ids.append(char_ngram_id)
        token.set_char_ngrams(char_ngram_surfaces, char_ngram_ids)
        return token

    def add_token(self, raw_surface):
        if raw_surface not in self.surface2token:
            token = self.gen_token(raw_surface)
            self.surface2token[raw_surface] = token
        else:
            pass

    def get_token(self, raw_surface):
        if raw_surface in self.surface2token:
            return self.surface2token[raw_surface]
        else:
            return self.gen_token(raw_surface)

    def initialize(self):
        for word in self.word_vocab.id2word:
            self.add_token(word)

    def preprocess(self, raw_word):
        if re.search(NUMBER_MATCHER, raw_word) and self.num_flg:
            new_word = NUM_TOKEN
        elif self.word_lower_flg:
            new_word = raw_word.lower()
        else:  # do nothing
            new_word = raw_word
        return new_word


class Vocab(object):
    def __init__(self):
        self.id2word = []
        self.word2id = {}

    def __contains__(self, word):
        return word in self.id2word

    def __getitem__(self, word):
        return self.word2id[word]

    def __len__(self):
        return len(self.id2word)

    def add(self, word):
        if word not in self:
            self.word2id[word] = len(self.id2word)
            self.id2word.append(word)

    # add SOS, EOS and UNK tokens
    def add_special_tokens(self):
        self.add_sos_token()
        self.add_eos_token()
        self.add_num_token()
        self.add_unk_token()

    def add_sos_token(self, tok=SOS_TOKEN):
        self.add(tok)
        self.sos_token = tok
        self.sos_id = self[tok]

    def add_eos_token(self, tok=EOS_TOKEN):
        self.add(tok)
        self.eos_token = tok
        self.eos_id = self[tok]

    def add_unk_token(self, tok=UNK_TOKEN):
        self.add(tok)
        self.unk_token = tok
        self.unk_id = self[tok]

    def add_num_token(self, tok=NUM_TOKEN):
        self.add(tok)
        self.num_token = tok
        self.num_id = self[tok]

    @classmethod
    def load(cls, data_path, min_freq=-1):
        v = Vocab()
        with open(data_path) as f:
            if min_freq < 0:
                for line in f:
                    assert len(line.split()) == 1
                    w = line.strip()
                    v.add(w)
            else:
                for line in f:
                    w, freq = line.strip().split()
                    if int(freq) > min_freq:
                        v.add(w)
        return v


def sent_tag_iter(data_path, rand_flg=True):
    n_line = sum(1 for _ in open(data_path)) // 3  # one sample have three line
    if rand_flg:
        line_idxs = np.random.permutation(n_line)
    else:
        line_idxs = [i for i in range(n_line)]
    sent = ''
    tags = ''
    for idx in line_idxs:
        sent = linecache.getline(data_path, 3*idx+1).strip().split()
        tags = linecache.getline(data_path, 3*idx+2).strip().split()
        assert len(sent) == len(tags)
        yield [sent, tags]


def batch_sent_tag_iter(data_path, batchsize, rand_flg=True):
    n_line = sum(1 for _ in open(data_path)) // 3  # one sample have three line
    if rand_flg:
        line_idxs = np.random.permutation(n_line)
    else:
        line_idxs = [i for i in range(n_line)]
    batch_sent = []
    batch_tags = []
    for cnt, idx in enumerate(line_idxs):
        sent = linecache.getline(data_path, 3*idx+1).strip().split()
        tags = linecache.getline(data_path, 3*idx+2).strip().split()
        assert len(sent) == len(tags)
        batch_sent.append(sent)
        batch_tags.append(tags)
        if (cnt + 1)%batchsize == 0:
            yield [batch_sent, batch_tags]
            batch_sent = []
            batch_tags = []
    if len(batch_sent) != 0:
        yield [batch_sent, batch_tags]


if __name__ == '__main__':
    wv = Vocab()
    wv.add('as')
    wv.add_special_tokens()
    c1v = Vocab()
    c1v.add('a')
    c1v.add('s')
    c1v.add_unk_token()
    c2v = Vocab()
    c2v.add('as')
    c2v.add('s'+EOW_CHAR)
    c2v.add_unk_token()

    print(wv.word2id)
    print(c1v.word2id)
    print(c2v.word2id)
    tm = TokenManager(wv, [c1v, c2v])
    a = tm.get_token('s')

    print(a.surface)
    print(a.id)
    print(a.char_ngram_ids)
