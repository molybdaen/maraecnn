__author__ = 'Johannes'

import sys
import string
import cPickle
import os
from os.path import join
import gzip
import numpy as np
import json
from general.config import Config, DataEnum, DataType, T_Data_Category
from util import exceptions as e


class ExampleStream(object):
    def __init__(self, dataIdentifier, vocabulary=None, minExampleLength=Config.CNN_WINDOW_SIZE):
        if not isinstance(dataIdentifier, DataType):
            raise e.NoValidPreprocessorException("Supply valid data identifier from Config.DataEnum.")
        self.dataIdentifier = dataIdentifier
        self.trainFileStream = FileStream(dataIdentifier)
        self.minExampleLength = minExampleLength
        self.count = 0
        self.labels = {}
        self.vocabulary = vocabulary
        self.iter = None

    def __iter__(self):
        for f in self.trainFileStream.get_iterator():
            self.count = 0
            for l in gzip.open(f, mode="r"):
                if self.vocabulary is None:
                    if self.dataIdentifier.dType == T_Data_Category.labeled_sentence_dataset:
                        (text, label) = json.loads(l)
                        if len(text.split()) >= self.minExampleLength:
                            floatLabel = float(label)
                            self.labels[floatLabel] = 1
                            yield [text, floatLabel]
                    else:
                        if len(l.split()) >= self.minExampleLength:
                            yield l
                else:
                    if self.dataIdentifier.dType == T_Data_Category.labeled_sentence_dataset:
                        (text, label) = json.loads(l)
                        if len(text.split()) >= self.minExampleLength:
                            floatLabel = float(label)
                            self.labels[floatLabel] = 1
                            yield [[self.vocabulary.id(w.strip()) for w in string.split(text)], float(label)]
                    else:
                        if len(l.split()) >= self.minExampleLength:
                            textIds = [self.vocabulary.id(w.strip()) for w in string.split(l)]
                            yield textIds
                self.count += 1

    def get_iterator(self):
        if self.iter is None:
            return self.__iter__()
        else:
            return self.iter

    def getCategorySize(self):
        return len(self.labels)

    def __getstate__(self):
        return self.trainFileStream.__getstate__(), self.count, self.dataIdentifier, self.vocabulary, self.labels, self.minExampleLength

    def __setstate__(self, state):
        fStreamState, count, dataId, vocabulary, labels, minExampleLength = state
        self.dataIdentifier = dataId
        self.vocabulary = vocabulary
        self.labels = labels
        self.minExampleLength = minExampleLength
        self.trainFileStream.__setstate__(fStreamState)
        self.iter = self.__iter__()
        while count != self.count:
            self.iter.next()
        print "fastforward to example %d." % (self.count + 1)


class FileStream(object):

    def __init__(self, dataIdentifier):
        self.dataIdentifier = dataIdentifier
        self.fIdx = 0
        self.filename = ''
        self.iter = None

    def __iter__(self):
        trainDir = join(self.dataIdentifier.getPath(), Config.DIRNAME_DATA_TRAIN)
        for f in os.listdir(trainDir):
            if f.endswith(Config.DATA_FILE_EXT):
                self.fIdx += 1
                self.filename = join(trainDir, str(f))
                yield self.filename

    def get_iterator(self):
        if self.iter is None:
            return self.__iter__()
        else:
            return self.iter

    def __getstate__(self):
        return self.fIdx, self.dataIdentifier

    def __setstate__(self, state):
        fIdx, dataId = state
        self.dataIdentifier = dataId
        self.iter = self.__iter__()
        while self.fIdx < fIdx - 1:
            self.iter.next()
        print "fastforward to file %d." % (self.fIdx + 1)


class MinibatchStream(object):

    def __init__(self, dataIdentifier, batchSize):
        self.count = -1
        self.batchSize = batchSize
        self.get_train_example = ExampleStream(dataIdentifier)

    def __iter__(self):
        minibatch = []
        if not hasattr(self, "get_train_example"):
            self.get_train_example = ExampleStream()
        for e in self.get_train_example.get_iterator():
            minibatch.append(e)
            if len(minibatch) >= self.batchSize:
                assert len(minibatch) == self.batchSize
                self.count += 1
                yield minibatch
                minibatch = []
        if len(minibatch) > 0:
            self.count += 1
            yield minibatch

    def __getstate__(self):
        return ((self.get_train_example.__getstate__(), self.count, self.batchSize),)

    def __setstate__(self, state):
        print >> sys.stderr, ("__setstate__(%s)..." % state)
        self.count = state[0][1]
        self.batchSize = state[0][2]
        self.get_train_example = ExampleStream(state[0][0][2])
        self.get_train_example.__setstate__(state[0][0])


class DataSet(object):

    def __init__(self, dataId, vocabulary, data=None, numFolds=1, minExampleLength=Config.CNN_WINDOW_SIZE):
        self.dataId = dataId
        self.vocabulary = vocabulary
        self._data = []
        self._size = 0
        self._numLabels = 0
        self._minExampleLength = minExampleLength
        self.splitNumber = None
        self.numFolds = numFolds
        self.batchSize = 1
        if data is None:
            self._load()
        else:
            self._data = data
            self._size = len(data)
        self.foldSize = int(self._size / self.numFolds)
        self.permutation = np.arange(self._size)
        np.random.shuffle(self.permutation)

    def _load(self):
        exampleStream = ExampleStream(dataIdentifier=self.dataId, vocabulary=self.vocabulary, minExampleLength=self._minExampleLength)
        for e in exampleStream:
            self._data.append(e)
        self._size = len(self._data)
        self._numLabels = exampleStream.getCategorySize()

    def setBatchMode(self, batchSize=10):
        self.batchSize = batchSize

    def getSplit(self, foldNumber):
        trainData = []
        testData = []
        i = 0
        splitMinIdx = foldNumber * self.foldSize
        splitMaxIdx = (foldNumber + 1) * self.foldSize
        while i < self._size:
            if splitMinIdx <= i < splitMaxIdx:
                testData.append(self._data[self.permutation[i]])
            else:
                trainData.append(self._data[self.permutation[i]])
            i += 1
        trainDataSet = DataSet(dataId=self.dataId, vocabulary=self.vocabulary, data=trainData, numFolds=self.numFolds, minExampleLength=self._minExampleLength)
        trainDataSet.splitNumber = foldNumber
        trainDataSet._numLabels = self._numLabels
        trainDataSet.batchSize = self.batchSize
        testDataSet = DataSet(dataId=self.dataId, vocabulary=self.vocabulary, data=testData, numFolds=self.numFolds, minExampleLength=self._minExampleLength)
        testDataSet.splitNumber = foldNumber
        testDataSet._numLabels = self._numLabels
        testDataSet.batchSize = self.batchSize
        return (trainDataSet, testDataSet)

    def getCategorySize(self):
        return self._numLabels

    def getSize(self):
        return self._size

    def __iter__(self):
        self.count = 0
        if self.batchSize == 1:
            for e in self._data:
                self.count += self.batchSize
                yield e
        else:
            minibatch = []
            for e in self._data:
                minibatch.append(e)
                if len(minibatch) >= self.batchSize:
                    self.count += self.batchSize
                    yield minibatch
                    minibatch = []

    def getRandomBatch(self):
        return [self._data[i] for i in np.random.randint(self._size, size=self.batchSize)]

    def __getstate__(self):
        return ((self.dataId, self.vocabulary, self._data, self._size, self._numLabels, self._minExampleLength, self.splitNumber, self.numFolds, self.batchSize, self.foldSize, self.permutation),)

    def __setstate__(self, state):
        (self.dataId, self.vocabulary, self._data, self._size, self._numLabels, self._minExampleLength, self.splitNumber, self.numFolds, self.batchSize, self.foldSize, self.permutation) = state[0]

    def __str__(self):
        splitStr = ""
        if self.splitNumber is None:
            splitStr = "none"
        else:
            splitStr = str(self.splitNumber) + " of " + str(self.numFolds)
        return "<Dataset dataId: %s, size: %d, minExampleLength: %d, numLabels: %d, batchSize: %d, vocabulary: %s, split: %s>" % (str(self.dataId), self._size, self._minExampleLength, self._numLabels, self.batchSize, str(self.vocabulary), splitStr)


class LanguageCorpus(object):

    def __init__(self, dataIdentifier):
        self.dataId = dataIdentifier
        self.trainFileStream = FileStream(dataIdentifier)
        self.count = 0
        self.iter = None

    def __iter__(self):
        for f in self.trainFileStream.get_iterator():
            self.count = 0
            for l in gzip.open(f, mode="r"):
                # Either read from labeled datasets or unlabeled corpora
                if self.dataId.isLabeled():
                    splitArr = string.split(json.loads(l)[0])
                    yield splitArr
                else:
                    yield string.split(l)

    def get_iterator(self):
        if self.iter is None:
            return self.__iter__()
        else:
            return self.iter

    def __getstate__(self):
        return self.trainFileStream.__getstate__(), self.count

    def __setstate__(self, state):

        fIdx, count = state

        # Fastforward file stream
        self.trainFileStream.__setstate__(fIdx)

        # Fastforward example stream
        self.iter = self.__iter__()
        while count != self.count:
            self.iter.next()
        print "fastforward to example %d" % count

        self.iterminibatch = []
        for e in self._data:
            minibatch.append(e)
            if len(minibatch) >= Config.RAE_MINIBATCH_SIZE:
                assert len(minibatch) == Config.RAE_MINIBATCH_SIZE
                self.count += 1
                yield minibatch
                minibatch = []
        yield minibatch
        minibatch = []
