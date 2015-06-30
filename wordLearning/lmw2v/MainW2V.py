__author__ = 'Molybdaen'

""" https://code.google.com/p/word2vec/ """
""" http://radimrehurek.com/gensim/models/word2vec.html
http://radimrehurek.com/2013/09/deep-learning-with-word2vec-and-gensim/"""

import logging
import cPickle
from general.config import ModelEnum, DataEnum, Config
from general.vocabulary import Vocabulary, Dictionary
from os.path import join

import gensim
import numpy as np
import string
from gensim import matutils

from general.streams import LanguageCorpus

def similarity(w1, w2, emb, model):
    return model.most_similar_cosmul(w1, w2)

def buildDictionaryFromTurian():
    o = open("C:/MA/language/corpora/turian/train/turian_cw_unscaled_100.txt", "rb")
    titles, x = [], []
    for l in o:
        toks = string.split(l)
        titles.append(toks[0])
        x.append([float(f) for f in toks[1:]])
    # x = np.array(x)
    # print x.shape
    languageVocab = Vocabulary()
    for i, key in enumerate(titles):
        languageVocab.put(key, len(titles)-i)
    languageVocab.close()
    languageDictionary = Dictionary(vocab=languageVocab)
    for (idx, t) in enumerate(titles):
        languageDictionary.put(languageVocab.id(t), x[idx])
    cPickle.dump(languageDictionary, open(join(DataEnum.turian.getPath(), Config.DIRNAME_DATA_TRAIN, Config.FILE_LANGUAGE_DICTIONARY), "w"))

def buildDictionary(w2v_model):
    # Extract trained embeddings + vocabulary and wrap it into Dictionary
    languageVocab = Vocabulary(minCount=w2v_model.min_count)
    for key, vocabWord in w2v_model.vocab.iteritems():
        if vocabWord.index == 0:
            print key
            print vocabWord
        if vocabWord.index == 2299:
            print key
            print vocabWord
        languageVocab.put(key, vocabWord.count)
    languageVocab.close()
    languageDictionary = Dictionary(vocab=languageVocab)
    for vocabWord in languageVocab.reverseMap:
        if vocabWord.word in w2v_model:
            languageDictionary.put(vocabWord.index, w2v_model[vocabWord.word])
    cPickle.dump(languageDictionary, open(join(dataId.getPath(), Config.DIRNAME_DATA_TRAIN, Config.FILE_LANGUAGE_DICTIONARY), "w"))

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

dataId = DataEnum.wikipedia
langCorp = LanguageCorpus(dataId)

model = gensim.models.Word2Vec(sentences=langCorp, sg=1, hs=0, negative=20, min_count=50, size=100, workers=4, window=9) # an empty model, no training
model.build_vocab(sentences=langCorp)  # can be a non-repeatable, 1-pass generator
model.train(sentences=langCorp)  # can be a non-repeatable, 1-pass generator
model.save(join(ModelEnum.w2v.getPath(), Config.FILE_MODEL))

new_model = gensim.models.Word2Vec.load(join(ModelEnum.w2v.getPath(), Config.FILE_MODEL))
print similarity(["movie", "amazingly"], ["boring"], new_model.syn0, new_model)
print new_model.syn0.shape
buildDictionary(new_model)

# buildDictionaryFromTurian()

# langDict = cPickle.load(open(join(dataId.getPath(), Config.DIRNAME_DATA_TRAIN, Config.FILE_LANGUAGE_DICTIONARY), "r"))

# new_model.accuracy(join(ModelEnum.w2v.getPath(), r'questions-words.txt'))
# print new_model.most_similar(positive=["money", 'gain'], negative=['earn'])
# print new_model.doesnt_match("movie film actor play role".split())
# print new_model.similarity('first', 'second')
# print new_model['sentence']
