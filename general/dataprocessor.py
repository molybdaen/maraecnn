__author__ = 'Johannes'

#!/usr/bin/env python
# -*- coding: latin-1 -*-

import os
import gzip
import sys
from os.path import join
import json
from random import shuffle

from general.tokenizer import Tokenizer
from util import exceptions as e
from general.config import Config, DataEnum, DataType, TokenizerEnum
from util.datacleaning import killgremlins


class IProcessor(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def build(self):
        """ Interface method for preprocessing raw language corpus or dataset.
        Transforms input data into a single, cleaned, gzipped training file.
        Directory structure is as follows:
            root
             ^- language
                ^- corpora      <-- language corpora reside in here
                   ^- amazon
                      ^- raw    <-- input files
                      ^- train  <-- single output file of preprocessed data
                   ^- wikipedia
                      ...
             ^- sentiment
                ^- datasets         <-- datasets for sentiment classification reside in here
                   ^- moviereviews
                      ^- raw        <-- input files
                      ^- train      <-- single output file of preprocessed data
                   ^- twitter
                      ...
        """
        raise NotImplementedError("This is an interface method. Implement it in subclass.")

    def shuffle(self):
        shuffled = False
        for f in os.listdir(self.pathToTrain):
            if f.endswith(Config.DATA_FILE_EXT):
                data = []
                filename = join(self.pathToTrain, str(f))
                print >> sys.stderr, "CHECK: Reading file: %s" % str(f)
                train_h = gzip.open(filename, 'r')
                for (idx, example) in enumerate(train_h):
                    data.append(example)
                train_h.close()
                print >> sys.stderr, "CHECK: ...Shuffling..."
                shuffle(data)
                print >> sys.stderr, "CHECK: Writing shuffled file."
                if os.path.exists(filename):
                    os.remove(filename)
                s_train_h = gzip.open(filename, 'w')
                for (idx, example) in enumerate(data):
                    s_train_h.write(example)
                s_train_h.close()
                shuffled = True
        if shuffled:
            print >> sys.stderr, "CHECK: Shuffled training data."
        else:
            print >> sys.stderr, "ERROR: Nothing to shuffle in training directory."


class WikipediaCorpusProcessor(IProcessor):

    PROC_CONF = DataEnum.wikipedia

    def __init__(self, tokenizer, maxTokens):
        super(WikipediaCorpusProcessor, self).__init__(tokenizer)

        self.maxTokens = maxTokens
        self.fileCount = long(0)
        self.exampleCount = long(0)
        self.tokenCount = long(0)
        self.pathToRaw = join(WikipediaCorpusProcessor.PROC_CONF.getPath(), Config.DIRNAME_DATA_RAW)
        self.pathToTrain = join(WikipediaCorpusProcessor.PROC_CONF.getPath(), Config.DIRNAME_DATA_TRAIN)
        # self.vocab = {}

    def build(self):
        for f in os.listdir(self.pathToRaw):
            if f.endswith(Config.DATA_FILE_EXT):
                self.fileCount += 1

        trainDataFilename = join(self.pathToTrain, "%s_all_data_%d.txt%s" % (Config.TRAIN_SENTENCES_PREFIX, self.maxTokens, Config.DATA_FILE_EXT))
        if not os.path.exists(self.pathToTrain):
            os.makedirs(self.pathToTrain)
        h_train_data = gzip.open(trainDataFilename, mode="w")

        remainingToks = 0
        avgTokens = self.maxTokens / self.fileCount
        for f in os.listdir(self.pathToRaw):
            if f.endswith(Config.DATA_FILE_EXT):
                tokensToRead = avgTokens + remainingToks
                print "#Tokens to read from file %s: %d" % (str(f), tokensToRead)
                rawDataFile = join(self.pathToRaw, str(f))
                remainingToks = self._read(rawDataFile, h_train_data, tokensToRead)
                print "Total #examples: %d" % self.exampleCount
                print "Total #tokens: %d" % self.tokenCount
        h_train_data.close()

    def _read(self, inputRawFile, h_train_data, tokToRead):
        h_raw_data = gzip.open(inputRawFile, mode="r")
        tmpTokCount = 0
        text = ""
        label = 0
        for line in h_raw_data:
            if tmpTokCount > tokToRead:
                break
            if self.exampleCount % 100000 == 0:
                print "Processed %d lines (%d tokens) in file." % (self.exampleCount, tmpTokCount)
            tokenized = []
            for token in line.split():
                ntoks = self.tokenizer.normalize(token)
                tokenized += ntoks
            tmpTokCount += len(tokenized)
            self.exampleCount += 1
            text = " ".join(tokenized)
            h_train_data.write(text)#json.dumps([text, label]))
            h_train_data.write('\n')
        h_raw_data.close()
        self.tokenCount += tmpTokCount
        remainingToks = tokToRead - tmpTokCount
        if remainingToks > 0:
            return remainingToks
        else:
            return 0


class AmazonCorpusProcessor(IProcessor):

    PROC_CONF = DataEnum.amazon

    def __init__(self, tokenizer, maxTokens):
        super(AmazonCorpusProcessor, self).__init__(tokenizer)

        self.maxTokens = maxTokens
        self.fileCount = long(0)
        self.exampleCount = long(0)
        self.tokenCount = long(0)
        self.pathToRaw = join(AmazonCorpusProcessor.PROC_CONF.getPath(), Config.DIRNAME_DATA_RAW)
        self.pathToTrain = join(AmazonCorpusProcessor.PROC_CONF.getPath(), Config.DIRNAME_DATA_TRAIN)
        # self.vocab = {}

    def build(self):
        for f in os.listdir(self.pathToRaw):
            if f.endswith(Config.DATA_FILE_EXT):
                self.fileCount += 1

        trainDataFilename = join(self.pathToTrain, "%s_all_data_%d.txt%s" % (Config.TRAIN_SENTENCES_PREFIX, self.maxTokens, Config.DATA_FILE_EXT))
        if not os.path.exists(self.pathToTrain):
            os.makedirs(self.pathToTrain)
        h_train_data = gzip.open(trainDataFilename, mode="w")

        remainingToks = 0
        avgTokens = self.maxTokens / self.fileCount
        for f in os.listdir(self.pathToRaw):
            if f.endswith(Config.DATA_FILE_EXT):
                tokensToRead = avgTokens + remainingToks
                print "#Tokens to read from file %s: %d" % (str(f), tokensToRead)
                rawDataFile = join(self.pathToRaw, str(f))
                remainingToks = self._read(rawDataFile, h_train_data, tokensToRead)
                print "Total #examples: %d" % self.exampleCount
                print "Total #tokens: %d" % self.tokenCount
        h_train_data.close()


    def _read(self, inputRawFile, h_train_data, tokToRead):
        h_raw_data = gzip.open(inputRawFile, mode="r")
        tmpTokCount = 0
        text = ""
        label = 0
        for line in h_raw_data:
            if tmpTokCount > tokToRead:
                break
            if line[0:12] == "review/text:":
                data = line[12:]
                tokenized = []
                for token in data.split():
                    ntoks = self.tokenizer.normalize(token)
                    tokenized += ntoks
                tmpTokCount += len(tokenized)
                self.exampleCount += 1
                text = " ".join(tokenized)
                h_train_data.write(text)
                h_train_data.write('\n')
        h_raw_data.close()
        self.tokenCount += tmpTokCount
        remainingToks = tokToRead - tmpTokCount
        if remainingToks > 0:
            return remainingToks
        else:
            return 0


class AmazonReviewProcessor(IProcessor):

    PROC_CONF = DataEnum.amazonreviews

    def __init__(self, tokenizer, maxTokens):
        super(AmazonReviewProcessor, self).__init__(tokenizer)
        self.maxTokens = maxTokens
        self.fileCount = long(0)
        self.exampleCount = long(0)
        self.tokenCount = long(0)
        self.labels = {}
        self.pathToRaw = join(AmazonReviewProcessor.PROC_CONF.getPath(), Config.DIRNAME_DATA_RAW)
        self.pathToTrain = join(AmazonReviewProcessor.PROC_CONF.getPath(), Config.DIRNAME_DATA_TRAIN)

    def build(self):
        for f in os.listdir(self.pathToRaw):
            if f.endswith(Config.DATA_FILE_EXT):
                self.fileCount += 1

        trainDataFilename = join(self.pathToTrain, "%s_all_data_%d.txt%s" % (Config.TRAIN_SENTENCES_PREFIX, self.maxTokens, Config.DATA_FILE_EXT))
        if not os.path.exists(self.pathToTrain):
            os.makedirs(self.pathToTrain)
        h_train_data = gzip.open(trainDataFilename, mode="w")

        remainingToks = 0
        avgTokens = self.maxTokens / self.fileCount
        for f in os.listdir(self.pathToRaw):
            if f.endswith(Config.DATA_FILE_EXT):
                tokensToRead = avgTokens + remainingToks
                print "#Tokens to read from file %s: %d" % (str(f), tokensToRead)
                rawDataFile = join(self.pathToRaw, str(f))
                remainingToks = self._read(rawDataFile, h_train_data, tokensToRead)
                print "Total #examples: %d" % self.exampleCount
                print "Total #tokens: %d" % self.tokenCount
        h_train_data.close()
        os.rename(trainDataFilename, join(self.pathToTrain, "%s_n%d_t%d_l%d.txt%s" % (Config.TRAIN_SENTENCES_PREFIX, self.exampleCount, self.tokenCount, len(self.labels), Config.DATA_FILE_EXT)))

    def _read(self, inputRawFile, h_train_data, tokToRead):
        h_raw_data = gzip.open(inputRawFile, mode="r")
        tmpTokCount = 0
        label = 0
        for line in h_raw_data:
            if tmpTokCount > tokToRead:
                break
            if line[0:13] == "review/score:":
                data = line[13:]
                label = float(data)
                self.labels[label] = 1
            if line[0:12] == "review/text:":
                data = line[12:]
                tokenized = []
                for token in data.split():
                    ntoks = self.tokenizer.normalize(token)
                    tokenized += ntoks
                tmpTokCount += len(tokenized)
                self.exampleCount += 1
                text = " ".join(tokenized)
                h_train_data.write(json.dumps([text, label]))
                h_train_data.write('\n')
        h_raw_data.close()
        self.tokenCount += tmpTokCount
        remainingToks = tokToRead - tmpTokCount
        if remainingToks > 0:
            return remainingToks
        else:
            return 0


class BoPangMovieReviewProcessor(IProcessor):

    PROC_CONF = DataEnum.moviereviews

    def __init__(self, tokenizer, maxTokens):
        super(BoPangMovieReviewProcessor, self).__init__(tokenizer)
        self.maxTokens = maxTokens
        self.fileCount = long(0)
        self.exampleCount = long(0)
        self.tokenCount = long(0)
        self.labels = {}
        self.pathToRaw = join(BoPangMovieReviewProcessor.PROC_CONF.getPath(), Config.DIRNAME_DATA_RAW)
        self.pathToTrain = join(BoPangMovieReviewProcessor.PROC_CONF.getPath(), Config.DIRNAME_DATA_TRAIN)

    def build(self):
        for f in os.listdir(self.pathToRaw):
            if f.endswith(Config.DATA_FILE_EXT):
                self.fileCount += 1

        trainDataFilename = join(self.pathToTrain, "%s_all_data_%d.txt%s" % (Config.TRAIN_SENTENCES_PREFIX, self.maxTokens, Config.DATA_FILE_EXT))
        if not os.path.exists(self.pathToTrain):
            os.makedirs(self.pathToTrain)
        h_train_data = gzip.open(trainDataFilename, mode="w")

        remainingToks = 0
        avgTokens = self.maxTokens / self.fileCount
        for f in os.listdir(self.pathToRaw):
            if f.endswith(Config.DATA_FILE_EXT):
                label = self._getLabelForDataPolarity(f)
                tokensToRead = avgTokens + remainingToks
                print "#Tokens to read from file %s: %d" % (str(f), tokensToRead)
                rawDataFile = join(self.pathToRaw, str(f))
                remainingToks = self._read(rawDataFile, h_train_data, tokensToRead, label)
                print "Total #examples: %d" % self.exampleCount
                print "Total #tokens: %d" % self.tokenCount
        h_train_data.close()
        newFileName = join(self.pathToTrain, "%s_n%d_t%d_l%d.txt%s" % (Config.TRAIN_SENTENCES_PREFIX, self.exampleCount, self.tokenCount, len(self.labels), Config.DATA_FILE_EXT))
        if newFileName != trainDataFilename:
            if os.path.exists(newFileName):
                os.remove(newFileName)
            os.rename(trainDataFilename, join(self.pathToTrain, "%s_n%d_t%d_l%d.txt%s" % (Config.TRAIN_SENTENCES_PREFIX, self.exampleCount, self.tokenCount, len(self.labels), Config.DATA_FILE_EXT)))


    def _getLabelForDataPolarity(self, filename):
        if filename[:-len(Config.DATA_FILE_EXT)].endswith(r'neg'):
            return 0.0
        elif filename[:-len(Config.DATA_FILE_EXT)].endswith(r'pos'):
            return 1.0
        else:
            raise e.DataFileException("Error: Expecting Binary Classification Dataset. File must end with 'neg' for files containing negative examples and 'pos' for files containing positive examples.")


    def _getLabelForDataSubjectivity(self, filename):
        if filename.startswith(r'plot'):
            return 0.0
        elif filename.startswith(r'quote'):
            return 1.0
        else:
            raise e.DataFileException("Error: Expecting Binary Classification Dataset. File must end with 'neg' for files containing negative examples and 'pos' for files containing positive examples.")

    def _read(self, inputRawFile, h_train_data, tokToRead, label):
        h_raw_data = gzip.open(inputRawFile, mode="r")
        tmpTokCount = 0
        self.labels[label] = 1
        for l in h_raw_data:
            line = killgremlins(l)
            nToks = line.split()
            if tmpTokCount > tokToRead:
                break
            tokenized = []
            for w in nToks:
                ntoks = self.tokenizer.normalize(w)
                # for (idx, w) in enumerate(ntoks):
                #     w = unicode(w)
                #     if not Config.CASE_SENSITIVE:
                #         w = w.lower()
                #     if w in self.vocab:
                #         self.vocab[w] += 1
                #     else:
                #         self.vocab[w] = 1
                tokenized += ntoks
            tmpTokCount += len(tokenized)
            self.exampleCount += 1
            text = " ".join(tokenized)
            h_train_data.write(json.dumps([text, label]))
            h_train_data.write('\n')
        h_raw_data.close()
        self.tokenCount += tmpTokCount
        remainingToks = tokToRead - tmpTokCount
        if remainingToks > 0:
            return remainingToks
        else:
            return 0


class TestWriteLoad:

    def __init__(self):
        pass

    def initi(self):
        tok = Tokenizer()

        self.path = r'C:\MA'
        self.rawFile = "rt-polarity.neg.gz"
        self.trainFile = "training.gz"

        raw_h = gzip.open(join(self.path, self.rawFile), 'r')
        train_h = gzip.open(join(self.path, self.trainFile), 'w')
        for l in raw_h:
            line = killgremlins(l)
            tokenized = []
            for token in line.split():
                ntoks = tok.normalize(token)
                tokenized.append(" ".join(ntoks))
            text = " ".join(tokenized)
            label = 3.3
            jsDump = json.dumps([text, label])
            train_h.write(jsDump + '\n')
        raw_h.close()
        train_h.close()

    def loadTrainFile(self):

        # wm = WordMap()
        train_h = gzip.open(r'C:\MA\bopang\data\train\train_all_data_224067.txt.gz')


class WikipediaProcessor(IProcessor):
    pass


class SentimentTreebankProcessor(IProcessor):
    pass


class TwitterProcessor(IProcessor):
    pass


class PreProcessor(object):
    def __init__(self, preprocessor=DataEnum.amazon, tokenizer=TokenizerEnum.sentiment, caseSensitive=True, maxTokens=Config.MAX_TOKENS):

        if tokenizer == TokenizerEnum.sentiment:
            tok = Tokenizer(preserve_case=caseSensitive)
        else:
            raise e.NoValidPreprocessorException("Supply valid tokenizer to PreProcessor. See config.PreprocessorEnum and config.TokenizerEnum for available preprocessors and tokenizers.")

        if preprocessor == DataEnum.amazon:
            pp = AmazonCorpusProcessor(tok, maxTokens)
        elif preprocessor == DataEnum.wikipedia:
            pp = WikipediaCorpusProcessor(tok, maxTokens)
        elif preprocessor == DataEnum.amazonreviews:
            pp = AmazonReviewProcessor(tok, maxTokens)
        elif preprocessor == DataEnum.moviereviews:
            pp = BoPangMovieReviewProcessor(tok, maxTokens)
        elif preprocessor == DataEnum.treebank:
            pp = SentimentTreebankProcessor(tok, maxTokens)
        elif preprocessor == DataEnum.twitter:
            pp = TwitterProcessor(tok, maxTokens)
        else:
            raise e.NoValidPreprocessorException("Supply valid preprocessor class to PreProcessor. See config.PreprocessorEnum and config.TokenizerEnum for available preprocessors.")

        self.processor = pp

    def run(self):
        self.processor.build()

    def shuffle(self):
        self.processor.shuffle()

#####################################
# NOTE on Character encoding
#####################################
# Python unicode and json.dump/load unicode look differently for special characters.
# For example: An e with accent is coded as \xe9 in python unicode string objects whereas json dumps it as \u00e9
# If you read the string back into python with json.loads() it gets converted automatically to its original python unicode string as \xe9
# In case you need to read a string like this manually from a file, you have to call pythonUniCodeString = str_with_tricky_json_coding.decode("unicode-escape") to get rid of the \u00e9 notation and back to the \xe9.

if __name__ == "__main__":
    pp = PreProcessor(preprocessor=DataEnum.wikipedia, tokenizer=TokenizerEnum.sentiment)
    pp.run()




