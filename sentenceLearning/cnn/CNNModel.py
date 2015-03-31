__author__ = 'Molybdaen'

import time
import sys
import logging

import numpy as np
import theano

from general.config import Config, ModelEnum
from general.analysis import SignalLogger
from general.streams import DataSet
from general.analysis import Accuracy
from util.movingaverage import MovingAverage
from CNNParameters import CNNParameters
from CNNGraph import CNNGraph

from general.vocabulary import _padding_key


class CNNModel(object):

    def __init__(self, dictionary, categorySize, epochs=10):
        self.categorySize = categorySize
        self.dictionary = dictionary
        self.epochs = epochs
        self.params = CNNParameters(embeddingMatrix=dictionary.embeddings, catSize=categorySize)
        self.graph = CNNGraph(categorySize)

    def _embed(self, wordIndices):
        sLen = len(wordIndices)
        lIdx = sLen - Config.CNN_WINDOW_SIZE + 1
        windows = []
        for i in xrange(0, lIdx):
            embeds = self.params.embeddings[wordIndices[i:i+Config.CNN_WINDOW_SIZE]]
            windows.append(np.hstack(embeds))
        return np.vstack(windows).T

    def _embedBatch(self, batch):
        indices = []
        examples = []
        labels = []
        for (e, l) in batch:
            fWordIndices = e
            if Config.CNN_PAD_SENTENCE:
                padding = [self.dictionary.vocab.id(_padding_key)] * (Config.CNN_WINDOW_SIZE / 2)
                fWordIndices = padding + e + padding
            indices.append(fWordIndices)
            examples.append(self._embed(fWordIndices))
            labels.append(self._getLabelRepresentation(l))
        return (indices, examples, labels)

    def _updateEmbeddings(self, gEmbeddings, wordIndices):
        for w in xrange(0, np.shape(gEmbeddings)[1]):
            self.params.embeddings[wordIndices[w:w+Config.CNN_WINDOW_SIZE]] -= gEmbeddings[:,w].reshape((Config.CNN_WINDOW_SIZE, Config.CNN_EMBED_SIZE))

    def _getLabelRepresentation(self, label):
        labelRep = np.asarray(np.zeros((self.categorySize,1)), dtype=theano.config.floatX)
        if label < 0.0 or label >= self.categorySize:
            print >> sys.stderr, "Error: Bad label datatype. Check 0 <= label < categorySize!"
        else:
            labelRep[int(label), 0] = 1.0
        return labelRep

    def learnFeatures(self, data):
        error = MovingAverage()
        signals = SignalLogger("epoch", "batch", "error")

        for e in xrange(self.epochs):
            logging.info("PROGRESS: Starting epoch %d / %d" % (e+1, self.epochs))
            alphaW = max(0.00001, Config.CNN_LR * (1 - 1.0 * float(e+1) / self.epochs))
            alphaL = max(0.00001, Config.CNN_ELR * (1 - 1.0 * float(e+1) / self.epochs))
            lastSum = 0
            sumCount = 0
            for (exampleCount, d) in enumerate(data):
                (indices, examples, labels) = self._embedBatch(d)
                (err, preds, GLs) = self.graph.convNetFitBatch(examples, labels, alphaW)
                for i in xrange(len(examples)):
                    self._updateEmbeddings(alphaL * GLs[i], indices[i])
                lastSum += err
                sumCount += 1.0
                error.add(err)
                signals.add(e, sumCount, err)
                if data.count % Config.INTERRUPT_VERYFREQUENT == 0:
                    avgErr = float(lastSum) / sumCount
                    logging.info("PROGRESS: Trained examples %d / %d" % (data.count, data.getSize()))
                    logging.info("STATS: Error: %s, (mean over last %d examples): %.5f" % (str(error), Config.INTERRUPT_VERYFREQUENT, avgErr))
                    lastSum = 0
                    sumCount = 0.0

        return signals

    def extractFeatures(self, data):
        reprs = []
        preds = []
        labels = []
        for (i, d) in enumerate(data):
            fWordIndices = d[0]
            if Config.CNN_PAD_SENTENCE:
                padding = [self.dictionary.vocab.id(_padding_key)] * (Config.CNN_WINDOW_SIZE / 2)
                fWordIndices = padding + d[0] + padding
            embInp = self._embed(fWordIndices)
            (repr, pred) = self.graph.theanoConvNetPredictFunc(embInp)
            reprs.append(repr)
            preds.append(int(np.argmax(pred.flatten())))
            labels.append(int(d[1]))
        return reprs, preds, labels

    def extractFeatures_single(self, wordSequence):
        wordIndices = [self.dictionary.vocab.id(word) for word in wordSequence]
        if Config.CNN_PAD_SENTENCE:
            padding = [self.dictionary.vocab.id(_padding_key)] * (Config.CNN_WINDOW_SIZE / 2)
            fWordIndices = padding + wordIndices + padding
        embInp = self._embed(fWordIndices)
        (repr, pred) = self.graph.theanoConvNetPredictFunc(embInp)
        labelPred = int(np.argmax(pred.flatten()))
        return repr, pred, labelPred

    def fit(self, data):
        strtTime = time.time()
        lastSum = 0
        for e in xrange(Config.CNN_EPOCHS):
            print "epoch: %d" % e
            for (i, d) in enumerate(data):
                if len(d[0]) >= Config.CNN_WINDOW_SIZE:
                    fWordIndices = d[0]
                    embInp = self._embed(fWordIndices)
                    (err, pred, gL) = self.graph.theanoConvNetFunc(embInp, self._getLabelRepresentation(d[1]))
                    self._updateEmbeddings(gL, fWordIndices)
                    lastSum += err
                    if i%1000 == 0:
                        print "example: %d " % i
                        avgErr = lastSum / 1000.0
                        print "Error: %.4f" % avgErr
                        lastSum = 0
        endTime = time.time()
        print "Took Time %.3f seconds" % (endTime-strtTime)

    def predict(self, data):
        preds = []
        labels = []
        for (i, d) in enumerate(data):
            if len(d[0]) >= Config.CNN_WINDOW_SIZE:
                embInp = self._embed(d[0])
                (pred) = self.graph.theanoConvNetPredictFunc(embInp)
                preds.append(int(np.argmax(pred, axis=1)))
                labels.append(int(d[1]))
        return preds, labels

    def __str__(self):
        return "<ModelName: %s, CategorySize: %d, Epochs: %d>" % (ModelEnum.cnn.name, self.categorySize, self.epochs)
