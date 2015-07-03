__author__ = 'johannesjurgovsky'

from gensim.models.word2vec import Word2Vec
import numpy as np
import cPickle
from os import path
from gensim import matutils
import logging
import gzip
from os.path import join
from general.config import DataEnum, DataType, Config
import codecs

def getEmbeddingsAndVocab(w2vModelFilename, rebuild=False):
    if path.exists(w2vModelFilename):
        p, f = path.split(w2vModelFilename)
        fName = f.split('.')[0]
        matFile = path.join(p, fName + "-mat.npy")
        vocFile = path.join(p, fName + "-voc.pkl")
        if not path.exists(matFile) or not path.exists(vocFile):
            model = Word2Vec.load_word2vec_format(w2vModelFilename, binary=False)
            np.save(matFile, model.syn0)
            cPickle.dump(model.vocab, open(vocFile, "w"))
        m = np.load(matFile)
        v = cPickle.load(open(vocFile, "r"))
        return m, v


class LineSentence(object):
    """Simple format: one sentence = one line; words already preprocessed and separated by whitespace."""
    def __init__(self, filename):
        """
        `source` can be either a string or a file object.

        Example::

            sentences = LineSentence('myfile.txt')

        Or for compressed files::

            sentences = LineSentence('compressed_text.txt.bz2')
            sentences = LineSentence('compressed_text.txt.gz')

        """
        self.source = codecs.open(filename, mode='r', encoding="cp1252")

    def any2utf8(self, text, errors='strict', encoding='utf8'):
        """Convert a string (unicode or bytestring in `encoding`), to bytestring in utf8."""
        if isinstance(text, unicode):
            return text
        # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
        return unicode(text, encoding, errors=errors)

    def __iter__(self):
        """Iterate through the lines in the source."""
        for line in self.source:
            yield line.split()#self.any2utf8(line).split()


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    sentences_path = join("../../data/language/corpora/wikipedia/train/train_all_data_1000000000.txt")
    sentences = LineSentence(sentences_path)
    model = Word2Vec(sg=1, hs=0, negative=20, min_count=100, size=100, workers=4, window=9) # an empty model, no training
    model.build_vocab(sentences=sentences)  # can be a non-repeatable, 1-pass generator
    model.train(sentences=LineSentence(sentences_path))  # can be a non-repeatable, 1-pass generator
    model.save_word2vec_format("../../data/language/model/model.w2v", fvocab="../../data/language/model/vocab.w2v", binary=True)

    new_model = Word2Vec.load_word2vec_format("../../data/language/model/model.w2v", fvocab="../../data/language/model/vocab.w2v", binary=True)

