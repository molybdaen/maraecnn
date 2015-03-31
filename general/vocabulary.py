__author__ = 'Molybdaen'

import gzip
import sys
import numpy as np
import theano

from general.config import Config
from util import exceptions as e
from util.utils import normalizeEmbeddings_scaleNormal, normalizeEmbeddingsL2, normalizeEmbeddings


_unknown_key = "*UNKNOWN*"
_padding_key = "*PADDING*"


class VocabWord(object):
    def __init__(self, word, count, index=-1):
        self.word = word
        self.count = count
        self.index = index

    def __lt__(self, other):
        return self.count < other.count

    def __str__(self):
        vals = ['%s:%r' % (key, self.__dict__[key]) for key in sorted(self.__dict__) if not key.startswith('_')]
        return "<" + ', '.join(vals) + ">"


class Vocabulary(object):
    def __init__(self, minCount=10):
        self.minCount = minCount
        self.reverseMap = []

        self.vocab = {}
        self.vocab[_unknown_key] = VocabWord(word=_unknown_key, count=sys.maxint)
        self.vocab[_padding_key] = VocabWord(word=_padding_key, count=sys.maxint)
        self.tokenCount = long(0)
        self.wordCount = long(0)
        self.lineCount = long(0)

    def readFromDataStream(self, dataStream):
        if dataStream.dataIdentifier.isLabeled():
            for (example, label) in dataStream:
                self.lineCount += 1
                tokenList = example.split()
                for t in tokenList:
                    self.tokenCount += 1
                    self._add(t)
        else:
            for example in dataStream:
                self.lineCount += 1
                tokenList = example.split()
                for t in tokenList:
                    self.tokenCount += 1
                    self._add(t)
        self.close()

    def readFromRawFile(self, filename):
        try:
            if filename.endswith(Config.DATA_FILE_EXT):
                h_train_data = gzip.open(filename, mode="r")
            else:
                h_train_data = open(filename, mode="r")
            for l in h_train_data:
                self.lineCount += 1
                tokenList = l.split()
                for t in tokenList:
                    self.tokenCount += 1
                    self._add(t)
            self.close()
            # self._toUnicodeType()
        except:
            e.DataFileException("Could not read file %s." % filename)

    def _add(self, word):
        if word in self.vocab:
            self.vocab[word].count += 1
        else:
            self.vocab[word] = VocabWord(word=word, count=1)

    def put(self, word, count):
        self.vocab[word] = VocabWord(word=word, count=count)

    def close(self):
        self.wordCount = len(self.vocab)
        self._buildIndex()

    def _buildIndex(self):
        vocabList = list(self.vocab.values())
        vocabList.sort(reverse=True)
        self.vocab = {}
        for vocabItem in vocabList:
            if vocabItem.count >= self.minCount:
                vocabItem.index = len(self.reverseMap)
                self.vocab[vocabItem.word] = vocabItem
                self.reverseMap.append(vocabItem)
        self.wordCount = len(self.vocab)

    def _toUnicodeType(self):
        _uniMap = {}
        _uniRevMap = [None] * len(self.reverseMap)
        for k in self.vocab:
            _uk = k.decode("utf-8")
            vocabItem = self.vocab[k]
            vocabItem.word = _uk
            _uniMap[_uk] = vocabItem
            _uniRevMap[self.vocab[k].index] = vocabItem
        self.vocab = _uniMap
        self.reverseMap = _uniRevMap

    def exists(self, key):
        return key in self.vocab

    def id(self, key):
        uk = unicode(key)
        if uk in self.vocab:
            return self.vocab[uk].index
        else:
            return self.vocab[_unknown_key].index

    def key(self, id):
        return self.reverseMap[id].word

    def counts(self, id):
        return self.reverseMap[id].count

    def size(self):
        return len(self.reverseMap)

    @property
    def all(self):
        return [i.word for i in self.reverseMap]

    @property
    def len(self):
        assert len(self.vocab) == len(self.reverseMap)
        return len(self.reverseMap)

    def __contains__(self, item):
        return (item in self.vocab)

    def __len__(self):
        return len(self.reverseMap)

    def __str__(self):
        return "<VocabSize: %d, MinCount: %r>" % (self.wordCount, self.minCount)

    def printFull(self):
        return "<VocabSize: %d, MinCount: %r, Vocab: \n%s>" % (self.wordCount, self.minCount, "\n".join([str(i) for i in self.reverseMap]))


class Dictionary(object):
    def __init__(self, vocab, embeddings=None):
        self.vocab = vocab
        if embeddings is None:
            self.embeddings = np.asarray((np.random.rand(vocab.size(), Config.EMBED_SIZE) - 0.5) * 2.0 * Config.INIT_EMBED_RANGE, dtype=theano.config.floatX)
        else:
            print embeddings.shape[0]
            print vocab.size()
            assert embeddings.shape[0] == vocab.size()-2
            self.embeddings = embeddings

    def put(self, wordIdx, wordEmbedding):
        self.embeddings[wordIdx,:] = wordEmbedding

    def initFrom(self, baseDictionary):
        c = 0
        for vocabItem in self.vocab.reverseMap:
            baseIdx = baseDictionary.vocab.id(vocabItem.word)
            if baseIdx > 2:
                c += 1
            baseEmb = baseDictionary.embeddings[baseIdx]
            self.embeddings[vocabItem.index] = baseEmb
        print "Replaced %d words from language model." % c
        # This is suggested for CW embeddings from Turian et al.
        # self.embeddings = normalizeEmbeddings_scaleNormal(self.embeddings)
        # ...whereas mikoholov et al. in W2V skip-gram model use L2 normalization prior to evaluating word vector quality
        # self.embeddings = normalizeEmbeddingsL2(self.embeddings)
        self.embeddings *= 0.1/np.std(self.embeddings)

    def getVocab(self):
        return self.vocab

    def getEmbeddings(self):
        return self.embeddings


if __name__ == "__main__":
    vocabulary = Vocabulary(minCount=5)
    vocabulary.readFromRawFile(r"C:\MA\sentiment\datasets\bopang-moviereviews\train\train_n53_t1062_l2.txt.gz")
    print vocabulary