import numpy as np
import chainer
from chainer import Variable
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

import sobamchan_chainer
import sobamchan_utility
utility = sobamchan_utility.Utility()


class Word2Vec(sobamchan_chainer.Model):

    def __init__(self, vocab_num, n_units):
        super(Word2Vec, self).__init__(
            embed=L.EmbedID(vocab_num, n_units),
        )

    def __call__(self, xb, yb, tb):
        y = self.fwd(xb, yb)
        return F.sigmoid_cross_entropy(y, tb)

    def fwd(self, x, y):
        x = self.embed(x)
        y = self.embed(y)
        return F.sum(x * y, axis=1)
