__author__ = 'Molybdaen'

import theano
import theano.tensor as T
import numpy as np
from general.config import Config


class CNNGraph:

    def __init__(self, categorySize):

        self.categorySize = categorySize
        rng = np.random.RandomState(23455)
        W_bound = np.sqrt(6. / (Config.CNN_WINDOW_SIZE*Config.CNN_EMBED_SIZE + Config.CNN_HIDDEN_SIZE))
        self.W1 = theano.shared(np.asarray(np.random.uniform(low=-W_bound, high=W_bound, size=(Config.CNN_HIDDEN_SIZE, Config.CNN_WINDOW_SIZE*Config.CNN_EMBED_SIZE)), dtype=theano.config.floatX), borrow=True)
        self.b1 = theano.shared(np.asarray(np.zeros((Config.CNN_HIDDEN_SIZE,1)), dtype=theano.config.floatX), borrow=True, broadcastable=(False, True))

        W_bound = np.sqrt(6. / (Config.CNN_HIDDEN_SIZE + Config.CNN_HIDDEN_SIZE))
        self.W2 = theano.shared(np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=(80, Config.CNN_HIDDEN_SIZE)), dtype=theano.config.floatX), borrow=True)
        self.b2 = theano.shared(np.asarray(np.zeros((80,1)), dtype=theano.config.floatX), borrow=True, broadcastable=(False, True))

        W_bound = np.sqrt(6. / (Config.CNN_HIDDEN_SIZE + 2))
        self.W3 = theano.shared(np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=(categorySize, Config.CNN_HIDDEN_SIZE)), dtype=theano.config.floatX), borrow=True)
        self.b3 = theano.shared(np.asarray(np.zeros((categorySize,1)), dtype=theano.config.floatX), borrow=True)

        self.input = T.matrix(name="sentenceInput", dtype=theano.config.floatX)
        self.target = T.matrix(name='target', dtype=theano.config.floatX)
        self.learningRate = T.scalar(name='learningRate', dtype=theano.config.floatX)

        convNet_upd, convNet_out = self._costGrad_convNet()
        convNetPredict_upd, convNetPredict_out = self._costPred_convNet()

        self.theanoConvNetFunc = theano.function(inputs=[self.input, self.target], updates=convNet_upd, outputs=convNet_out)
        self.theanoConvNetPredictFunc = theano.function(inputs=[self.input], updates=convNetPredict_upd, outputs=convNetPredict_out)

    def dropout(self):
        w1 = self.W1.get_value().copy()
        indices = [r > 0.8 for r in np.random.rand(Config.CNN_HIDDEN_SIZE)]
        for i, b in enumerate(indices):
            if b:
                w1[i] = np.zeros(Config.CNN_WINDOW_SIZE*Config.CNN_HIDDEN_SIZE)
        self.W1.set_value(w1)

    def restoreDropout(self):
        self.W1.set_value(self.w1orig)

    def convNetFit(self, input, target, alphaW):
        return self.theanoConvNetFunc(input, target, alphaW)

    def convNetFitBatch(self, inputs, targets, alphaW):
        batchSize = len(inputs)
        accGW1 = np.asarray(np.zeros((Config.CNN_HIDDEN_SIZE, Config.CNN_WINDOW_SIZE*Config.CNN_EMBED_SIZE)), dtype=theano.config.floatX)
        accGb1 = np.asarray(np.zeros((Config.CNN_HIDDEN_SIZE,1)), dtype=theano.config.floatX)
        accGW3 = np.asarray(np.zeros((self.categorySize, Config.CNN_HIDDEN_SIZE)), dtype=theano.config.floatX)
        accGb3 = np.asarray(np.zeros((self.categorySize,1)), dtype=theano.config.floatX)
        GLs = []
        preds = []
        error = 0.0
        self.w1orig = self.W1.get_value().copy()
        for (example, target) in zip(inputs, targets):
            # self.dropout()
            (err, pred, gL, gW1, gb1, gW3, gb3) = self.theanoConvNetFunc(example, target)
            accGW1 += gW1
            accGb1 += gb1
            accGW3 += gW3
            accGb3 += gb3
            error += err
            preds.append(pred)
            GLs.append(gL)
        reg = (np.sum(self.W1.get_value()**2.0) + np.sum(self.W3.get_value()**2.0))
        error = (1. / batchSize) * error + 0.5 * Config.CNN_LAMBDA * reg
        self.W1.set_value(self.W1.get_value() - (alphaW * (1. / batchSize) * accGW1 + Config.CNN_LAMBDA * self.W1.get_value()))
        self.b1.set_value(self.b1.get_value() - alphaW * (1. / batchSize) * accGb1)
        self.W3.set_value(self.W3.get_value() - (alphaW * (1. / batchSize) * accGW3 + Config.CNN_LAMBDA * self.W3.get_value()))
        self.b3.set_value(self.b3.get_value() - alphaW * (1. / batchSize) * accGb3)
        return (error, preds, GLs)

    def convNetPredict(self, input):
        return self.theanoConvNetPredictFunc(input)

    def getParams(self):
        return (self.W1.get_value(), self.b1.get_value(), self.W2.get_value(), self.b2.get_value(), self.W3.get_value(), self.b3.get_value())

    def _costGrad_convNet(self):
        activation = T.dot(self.W1, self.input) + self.b1
        fmaps = T.tanh(activation)
        repr = T.reshape(T.max(fmaps, axis=1), (Config.CNN_HIDDEN_SIZE,1))

        g = T.dot(self.W3, repr) + self.b3
        pred = T.nnet.softmax(g.T).T
        cEErr = -T.sum((self.target * T.log(pred)))

        gW1 = T.grad(cEErr, self.W1)
        gW3 = T.grad(cEErr, self.W3)
        gb1 = T.grad(cEErr, self.b1)
        gb3 = T.grad(cEErr, self.b3)
        gInput = T.grad(cEErr, self.input)

        output = [cEErr, pred, gInput, gW1, gb1, gW3, gb3]
        updates = []
        return updates, output

    def _costPred_convNet(self):
        activation = T.dot(self.W1, self.input) + self.b1
        fmaps = activation
        repr = T.reshape(T.max(fmaps, axis=1), (Config.CNN_HIDDEN_SIZE,1))
        g = T.dot(self.W3, repr) + self.b3
        pred = T.nnet.softmax(g.T).T
        output = [repr, pred]
        updates = []
        return updates, output
