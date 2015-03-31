__author__ = 'Johannes'

import datetime
from os.path import join
import os
import sys
import gzip
import string
import logging

import theano
import numpy as np

from general.config import Config
import preprocessing.wordmap as wordmapping
from preprocessing.wordmap import WordMap
from util import exceptions


class LoadStateError(Exception):

    def __init__(self, snapshotDir, runId):
        self.snapshotDir = snapshotDir
        self.runId = runId

def translateToSentence(indexedExample, vocabulary):
    return " ".join([vocabulary.key(w) for w in indexedExample])

def normalizeEmbeddings(L):
    lengths = np.linalg.norm(L, axis=1)
    nL = L / lengths.reshape(len(lengths), 1)
    return nL

def normalizeEmbeddingsL2(L):
    # Rescale embedding vectors to unit length
    # Use cosine similarity for evaluation of word vectors
    return (L / np.sqrt((L ** 2).sum(-1))[..., np.newaxis]).astype(np.float32)

def normalizeEmbeddings_scale01(L):
    mmax = np.max(L)
    mmin = np.min(L)
    nL = (L - mmin) / (mmax - mmin)
    return nL

def normalizeEmbeddings_scaleNormal(L):
    # Assumption is that embedding values follow normal distribution.
    # So we center by subtracting mean and rescale by dividing by stddev.
    # Multiplication with Config.EMBED_SCALING reduces absolute variance of rescaled embeddings so most of the embed values are in [-1;1]
    # Use Euclidean distance for evaluation of word vectors. cosine might work as well
    return Config.EMBED_SCALING * ((L - np.mean(L))/np.std(L))

