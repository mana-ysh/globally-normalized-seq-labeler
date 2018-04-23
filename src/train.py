
import argparse
from datetime import datetime
import logging
import numpy as np
import os
import pickle
import re
import time

import chainer
from chainer import optimizer, optimizers

from globalnn import GlobalNN
from localnn import LocalNN
from utils import Vocab, TokenManager, sent_tag_iter


posdict_path = os.path.abspath(os.path.dirname(__file__)) + '/../../data/pos_dict.pkl'
# vocab_path = os.path.abspath(os.path.dirname(__file__)) + '/../../data/train.02-21.sent.vocab.pkl'
wvocab_path = os.path.abspath(os.path.dirname(__file__)) + '/../../data/train02-21.wordfreq'
cvocab_paths = [os.path.abspath(os.path.dirname(__file__)) + '/../../data/train02-21.1charlist',
                os.path.abspath(os.path.dirname(__file__)) + '/../../data/train02-21.2charlist',
                os.path.abspath(os.path.dirname(__file__)) + '/../../data/train02-21.3charlist']


SOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
UNK_token = '<UNK>'
N_POS = 45

DIR_NAME = 'pos_beam_gb'

MAX = 10**10

with open(posdict_path, 'rb') as f:
    pos2id = pickle.load(f)
pos2id['<s>'] = len(pos2id)
pos2id['</s>'] = len(pos2id)

np.random.seed(46)


def train(args):
    if args.log:
        log_dir = args.log
    else:
        log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '{}_{}'.format(DIR_NAME, datetime.now().strftime('%Y%m%d_%H:%M')))

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # setting for logging
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    log_path = os.path.join(log_dir, 'log')
    file_handler = logging.FileHandler(log_path)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.info('Arguments...')
    for arg, val in vars(args).items():
        logger.info('{} : {}'.format(arg, val))

    logger.info('building vocabs...')
    word_v = Vocab.load(wvocab_path, min_freq=args.min_freq)
    word_v.add_special_tokens()
    # pos_v = Vocab.load(pvocab_path)
    char_vs = []
    for i in range(args.nchar):
        v = Vocab.load(cvocab_paths[i])
        v.add_unk_token()
        char_vs.append(v)

    token_manager = TokenManager(word_v, char_vs, args.lowercase_flg, args.num_flg)
    n_char_ngrams = [len(v) for v in char_vs]

    if args.local_model:
        logger.info('using pretrained local model...')
        localnn = LocalNN.load(args.local_model)
    else:
        localnn = LocalNN(args.wembed, args.cembed, args.pembed, args.hidden, args.wordwindow, args.poswindow, len(word_v), N_POS, n_char_ngrams)

    if args.init_softmax:
        logger.info('reinitialize softmax layer...')
        localnn.h2y.W.data[:] = np.random.random(localnn.h2y.W.shape)

    pad_tokens = [token_manager.get_token(word_v.sos_token),
                  token_manager.get_token(word_v.eos_token)]

    model = GlobalNN(localnn, args.beamwidth, args.nchar, pad_tokens, args.gpu_id)
    model.init_optimizer(args.wdecay, args.opt, args.lr, args.momentum)

    n_update = 0
    for epoch in range(args.epoch):
        logger.info('start {} epoch'.format(epoch + 1))
        sum_loss = 0
        sum_step = 0
        n_comp = 0
        n_word = 0
        start_epoch = time.time()
        logger.info('current learning rate: {}'.format(model.opt.lr))
        for i, [words, poss] in enumerate(sent_tag_iter(args.train_data)):
            tokens = [token_manager.get_token(word) for word in words]
            gold_poss = [pos2id[pos] for pos in poss]
            loss, n_step = model.update(tokens, gold_poss)
            if n_update%args.decay_step:
                model.opt.lr *= args.lrdecay
                logger.info('decaying learning rate: {}'.format(model.opt.lr))
            sum_loss += loss
            sum_step += n_step
            n_word += len(words)
            if n_step == len(words):
                n_comp += 1
            n_update += 1
            logger.info('loss = {} : step = {}/{}'.format(loss, n_step, len(words)))
            logger.info('done {} updates in this epoch'.format(i+1))
        logger.info('sum_loss : {}'.format(sum_loss))
        logger.info('sum_step : {}'.format(sum_step))
        logger.info('n_comp   : {}'.format(n_comp))
        logger.info('{} sec for training in this epoch'.format(time.time() - start_epoch))

        if (epoch+1)%args.evalstep == 0:
            if args.valid_data:
                logger.info('validation...')
                n_corr, n_word = evaluation(model, args.valid_data, token_manager, pos2id)
                logger.info('accuracy in validation : {} / {} = {}'.format(n_corr, n_word, float(n_corr/n_word)))

            if args.test_data:
                logger.info('testing...')
                n_corr, n_word = evaluation(model, args.test_data, token_manager, pos2id)
                logger.info('accuracy in testing : {} / {} = {}'.format(n_corr, n_word, float(n_corr/n_word)))

        logger.info('saving model...')
        model_path = os.path.join(log_dir, 'epoch{}.model'.format(epoch+1))
        model.save(model_path)

    logger.info('DONE ALL')


def evaluation(model, data_path, token_manager, pos2id):
    n_word = 0
    n_corr = 0
    for [words, poss] in sent_tag_iter(data_path):
        tokens = [token_manager.get_token(word) for word in words]
        gold_poss = [pos2id[pos] for pos in poss]
        pred_poss = model.beam_search(tokens, gold_poss, train_flg=False)
        assert len(gold_poss) == len(pred_poss)
        n_corr += sum(1 for i in range(len(gold_poss)) if gold_poss[i] == pred_poss[i])
        n_word += len(gold_poss)
    return n_corr, n_word


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train_data')
    p.add_argument('--valid_data', default=None)
    p.add_argument('--test_data', default=None)
    p.add_argument('--min_freq', default=-1, type=int)
    p.add_argument('--lowercase_flg', action='store_true')
    p.add_argument('--num_flg', action='store_true')
    p.add_argument('--opt', default='Adam')
    p.add_argument('--lr', default=0.01, type=float)
    p.add_argument('--lrdecay', default=1.0, type=bool)
    p.add_argument('--decay_step', default=MAX, type=int)
    p.add_argument('--init_softmax', default=False, type=bool)
    p.add_argument('--momentum', default=0.9, type=float)
    p.add_argument('--evalstep', default=1, type=int)
    p.add_argument('--wordwindow', type=int, help='window-size (must be odd)')
    p.add_argument('--poswindow', type=int)
    p.add_argument('--wembed', type=int)
    p.add_argument('--cembed', type=int)
    p.add_argument('--pembed', type=int)
    p.add_argument('--hidden', type=int)
    p.add_argument('--epoch', type=int)
    p.add_argument('--wdecay', type=float)
    p.add_argument('--beamwidth', default=8, type=int)
    p.add_argument('--gpu_id', default=-1, type=int)
    p.add_argument('--log')
    p.add_argument('--local_model', default=None)
    p.add_argument('--nchar', default=3, type=int)

    train(p.parse_args())
