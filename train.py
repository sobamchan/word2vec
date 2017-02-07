import numpy as np
from chainer import optimizers
from chainer.utils import walker_alias
from tqdm import tqdm

import argparse
from word2vec import Word2Vec
from lib.utility import make_batch_set

import sobamchan_utility
import sobamchan_vocabulary
import sobamchan_slack
utility = sobamchan_utility.Utility()
slack = sobamchan_slack.Slack()

def get_args():
    parser = argparse.ArgumentParser('word2vec train')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--bs', type=int, default=10, help='batch size')
    parser.add_argument('--ws', type=int, default=10, help='window size')
    parser.add_argument('--embed', type=int, default=300, help='embedding dimention size')
    parser.add_argument('--negative', type=int, default=5, help='negative sampling size')

    return parser.parse_args()

def train(args):
    slack.s_print('here we go', channel='output')
    vocabulary = sobamchan_vocabulary.Vocabulary()
    for line in utility.readlines_from_filepath(args.data):
        vocabulary.new(line)

    vocab_num = len(vocabulary)
    datasize = len(vocabulary.dataset)

    model = Word2Vec(vocab_num, args.embed)
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    power = utility.np_float32(0.75)
    sampler = walker_alias.WalkerAlias(power)

    for e in tqdm(range(args.epoch)):
        loss_sum = 0
        sffindx = np.random.permutation(datasize)
        for pos in range(0, datasize, args.bs):
            ids = sffindx[pos:pos+args.bs]
            xb, yb, tb = make_batch_set(vocabulary.dataset, ids, sampler, args.negative, args.ws)
            model.cleargrads()
            xb = model.prepare_input(xb, np.int32, volatile=False)
            yb = model.prepare_input(yb, np.int32, volatile=False)
            tb = model.prepare_input(tb, np.int32, volatile=False)
            loss = model(xb, yb, tb)
            loss.backward()
            optimizer.update()
            loss_sum += float(loss.data) * len(xb)
        slack.s_print(loss_sum/datasize, 'output')

    model.save_model('./word2vec.model')

if __name__ == '__main__':
    args = get_args()
    train(args)
