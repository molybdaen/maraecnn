__author__ = 'Molybdaen'

import math

import theano
import theano.tensor as T
import numpy as np
from theano import shared

from general.config import Config


class RAESoftmaxClassifier(object):

    def __init__(self):
        self.scale_init_weights_by = 1.0
        self.input = T.matrix(name='input', dtype=theano.config.floatX)
        self.target = T.matrix(name='target', dtype=theano.config.floatX)
        self.Wlabel = shared(np.asarray(random_weights(2, 2*Config.EMBED_SIZE, scale_by=self.scale_init_weights_by), dtype=theano.config.floatX))
        self.blabel = shared(np.asarray(np.zeros((2,1)), dtype=theano.config.floatX), broadcastable=(False, True))

        soft_upd, soft_out = self._costGrad_Softmax()
        p_soft_upd, p_soft_out = self._predict_Softmax()

        self.trainFunction = theano.function(inputs=[self.input, self.target], updates=soft_upd, outputs=soft_out, allow_input_downcast=True)
        self.predictFunction = theano.function(inputs=[self.input], updates=p_soft_upd, outputs=p_soft_out, allow_input_downcast=True)

    def fit(self, features, label):
        return self.trainFunction(features, label)

    def predict(self, features):
        return self.predictFunction(features)

    def _costGrad_Softmax(self):
        x = T.dot(self.Wlabel, self.input) + self.blabel
        e = T.exp(x)
        es = T.sum(e, axis=0)
        pred = e / es
        cEErr = -T.sum((T.log(pred) * self.target), axis=0)
        summedError = T.mean(cEErr)
        gW = T.grad(summedError, self.Wlabel)
        gb = T.grad(summedError, self.blabel)
        output = [summedError, pred]
        updates = [(self.Wlabel, self.Wlabel - 0.5 * gW), (self.blabel, self.blabel - 0.5 * gb)]
        return updates, output

    def mySoftmax(self, x, label):
        netIn = np.dot(self.Wlabel, x) + self.blabel
        e = np.exp(netIn)
        es = np.sum(e, axis=0)
        pred = e / es
        cEErr = -np.sum((np.log(pred) * label), axis=0)
        deltaG = pred-label
        output = [cEErr, pred]
        self.Wlabel -= 0.1 * np.dot(deltaG, x.T)
        return output

    def _predict_Softmax(self):
        x = T.dot(self.Wlabel, self.input) + self.blabel
        e = T.exp(x)
        es = T.sum(e, axis=0)
        pred = e / es
        output = [pred]
        updates = []
        return updates, output

sqrt3 = math.sqrt(3.0)

def random_weights(nin, nout, scale_by=1./sqrt3, power=0.5):
    return (np.random.rand(nin, nout) * 2.0 - 1.0) * scale_by * sqrt3 / pow(nin,power)