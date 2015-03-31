__author__ = 'Johannes'

import itertools
import matplotlib.pyplot as plt
import tsne
import numpy as np

from general.config import Config
import textwrap
import logging
from gensim import utils, matutils
from six import iteritems, itervalues, string_types

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger("analysis.wordaccuracy")

class Accuracy(object):

    def __init__(self, predictions, goldLabels, catSize):
        # row index is gold label
        # column index is predicted label

        head = "Classified Examples: %d" % len(goldLabels)
        corr = 0
        corrAsZero = 0
        corrAsOne = 0
        wrongAsOne = 0
        wrongAsZero = 0
        for (idx, g) in enumerate(goldLabels):
            if predictions[idx] == 0 and goldLabels[idx] == 0:
                corrAsZero += 1
                corr += 1
            if predictions[idx] == 1 and goldLabels[idx] == 1:
                corrAsOne += 1
                corr += 1
            if predictions[idx] == 1 and goldLabels[idx] == 0:
                wrongAsOne += 1
                corr += 1
            if predictions[idx] == 0 and goldLabels[idx] == 1:
                wrongAsZero += 1
                corr += 1
        cz = "Correct as Zero: %d" % corrAsZero
        co = "Correct as One : %d" % corrAsOne
        wz = "Wrong as Zero: %d" % wrongAsZero
        wo = "Wrong as One : %d" % wrongAsOne
        ct = "Correct overall: %d" % corr

        self.info = head + "\n" + cz + "\n" + co + "\n" + wz + "\n" + wo + "\n" + ct + "\n"

        self.confMatInt = np.array([z.count(x) for z in [zip(goldLabels,predictions)] for x in itertools.product([0,1],repeat=2)]).reshape(catSize,catSize)
        self.confusionMatrix = np.array(self.confMatInt, dtype=float)
        diag = np.diag(self.confusionMatrix)
        colSums = np.sum(self.confusionMatrix, axis=0)
        rowSums = np.sum(self.confusionMatrix, axis=1)
        self.Precision = np.sum(diag / colSums) / catSize
        self.Recall = np.sum(diag / rowSums) / catSize
        self.Accuracy = np.sum(diag) / np.sum(self.confusionMatrix)
        self.F1 = (2 * self.Precision * self.Recall) / (self.Precision + self.Recall + 1e-10)

    def __str__(self):
        return str(self.info) + "Confusion Matrix\n" + str(self.confMatInt) +  "\nAccuracy { Precision: %.5f, Recall: %.5f, Accuracy: %.5f, F1: %.5f }" % (self.Precision, self.Recall, self.Accuracy, self.F1)


class SignalLogger(object):
    def __init__(self, *signalNames):
        self.signals = {}
        self.signalNames = signalNames
        for signalName in self.signalNames:
            self.signals[signalName] = []

    def add(self, *signalValues):
        for i in xrange(len(self.signalNames)):
            self.signals[self.signalNames[i]].append(signalValues[i])

    def getMatlabStr(self, signalName):
        return "%s = [%s];" % (signalName, ", ".join(str(float(i)) for i in self.signals[signalName]))

    def __str__(self):
        stringies = []
        for signalName in self.signalNames:
            stringies.append(self.getMatlabStr(signalName))
        return "\n".join(stringies)


class WordAccuracy(object):
    """
        word accuracy calculations from gensim.word2vec
        Compute accuracy of the word vectors. `questions` is a filename where lines are
        4-tuples of words, split into sections by ": SECTION NAME" lines.
        See https://code.google.com/p/word2vec/source/browse/trunk/questions-words.txt for an example.

        The accuracy is reported (=printed to log and returned as a list) for each
        section separately, plus there's one aggregate summary at the end.

        Use `restrict_vocab` to ignore all questions containing a word whose frequency
        is not in the top-N most frequent words (default top 30,000).

        This method corresponds to the `compute-accuracy` script of the original C word2vec.
    """

    def __init__(self, dictionary, questions):
        self.vocab = dictionary.getVocab()
        self.emb = dictionary.getEmbeddings()
        self.questions = questions
        self.init_sims()

    def init_sims(self):
        logger.info("precomputing L2-norms of word weight vectors")
        self.embnorm = (self.emb / np.sqrt((self.emb ** 2).sum(-1))[..., np.newaxis]).astype(np.float32)

    def most_similar(self, positive=[], negative=[], topn=10):
        if isinstance(positive, string_types) and not negative:
            # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
            positive = [positive]

        # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
        positive = [(word, 1.0) if isinstance(word, string_types + (np.ndarray,))
                                else word for word in positive]
        negative = [(word, -1.0) if isinstance(word, string_types + (np.ndarray,))
                                 else word for word in negative]

        # compute the weighted average of all words
        all_words, mean = set(), []
        for word, weight in positive + negative:
            if isinstance(word, np.ndarray):
                mean.append(weight * word)
            elif word in self.vocab:
                mean.append(weight * self.embnorm[self.vocab.id(word)])
                all_words.add(self.vocab.id(word))
            else:
                raise KeyError("word '%s' not in vocabulary" % word)
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)

        dists = np.dot(self.embnorm, mean)
        if not topn:
            return dists
        best = np.argsort(dists)[::-1][:topn + len(all_words)]
        # ignore (don't return) words from the input
        result = [(self.vocab.key(sim), float(dists[sim])) for sim in best if sim not in all_words]
        return result[:topn]

    @staticmethod
    def log_accuracy(section):
        correct, incorrect = len(section['correct']), len(section['incorrect'])
        if correct + incorrect > 0:
            logger.info("%s: %.1f%% (%i/%i)" %
                (section['section'], 100.0 * correct / (correct + incorrect),
                correct, correct + incorrect))

    def accuracy(self, restrict_vocab=30000, most_similar=most_similar):
        ok_vocab = dict(sorted(iteritems(self.vocab.vocab),
                               key=lambda item: -item[1].count)[:restrict_vocab])
        ok_index = set(v.index for v in itervalues(ok_vocab))

        sections, section = [], None
        for line_no, line in enumerate(utils.smart_open(self.questions)):
            line = utils.to_unicode(line)
            if line.startswith(': '):
                # a new section starts => store the old section
                if section:
                    sections.append(section)
                    self.log_accuracy(section)
                section = {'section': line.lstrip(': ').strip(), 'correct': [], 'incorrect': []}
            else:
                if not section:
                    raise ValueError("missing section header before line #%i in %s" % (line_no, self.questions))
                try:
                    a, b, c, expected = [word for word in line.split()]
                except:
                    logger.info("skipping invalid line #%i in %s" % (line_no, self.questions))
                if a not in ok_vocab or b not in ok_vocab or c not in ok_vocab or expected not in ok_vocab:
                    logger.debug("skipping line #%i with OOV words: %s" % (line_no, line.strip()))
                    continue

                ignore = set(self.vocab.id(v) for v in [a, b, c])  # indexes of words to ignore
                predicted = None
                # find the most likely prediction, ignoring OOV words and input words
                for index in np.argsort(most_similar(self, positive=[b, c], negative=[a], topn=False))[::-1]:
                    if index in ok_index and index not in ignore:
                        predicted = self.vocab.key(index)
                        # if predicted != expected:
                            # logger.debug("%s: expected %s, predicted %s" % (line.strip(), expected, predicted))
                        break
                if predicted == expected:
                    section['correct'].append((a, b, c, expected))
                else:
                    section['incorrect'].append((a, b, c, expected))
        if section:
            # store the last section, too
            sections.append(section)
            self.log_accuracy(section)

        total = {
            'section': 'total',
            'correct': sum((s['correct'] for s in sections), []),
            'incorrect': sum((s['incorrect'] for s in sections), []),
        }
        self.log_accuracy(total)
        sections.append(total)
        return sections


class PhraseAccuracy(object):
    """
    compute similarities of phrase representations, cosine similarity
    """
    def __init__(self, examples, emb):
        self.examples = examples
        self.emb = emb
        self.init_sims()

    def init_sims(self):
        self.emb = np.squeeze(self.emb)
        lengths = np.sqrt(np.sum(self.emb**2., axis=1))
        self.embnorm = self.emb / lengths[:, np.newaxis]
        self.embnorm = (self.emb / np.sqrt((self.emb ** 2).sum(-1))[..., np.newaxis]).astype(np.float32)
        self.phr_sims = np.dot(self.embnorm, self.embnorm.T)

    def accuracy(self, topn=10):
        sections = {"positive": [], "negative":[]}
        for (i, phr) in enumerate(self.phr_sims):
            bestn = np.argsort(phr)[::-1][:topn + 1]
            result = [(self.examples[sim_phr_i], float(phr[sim_phr_i])) for sim_phr_i in bestn if sim_phr_i != i]
            correct = len([(correctNeighbor) for correctNeighbor in result if correctNeighbor[0][1] == self.examples[i][1]])
            acc = correct/float(topn)
            if self.examples[i][1] == 1.0:
                sections["positive"].append(acc)
            else:
                sections["negative"].append(acc)
        return [(key, sum(sections[key])/float(len(sections[key]))) for key in sections]


def plotEmbeddings(embeddings, labels, showInfo=[]):
    X = embeddings.astype(np.float64)
    X *= 1.0 / np.std(X)
    Y = tsne.tsne(X, no_dims=2, initial_dims=40, perplexity=5);
    for i in xrange(len(labels)):
        if labels[i][2] == 1:
            color = "green"

        else:
            color = "red"

        if labels[i][3]:
            txt = "X"
        else:
            txt = "."
        plt.annotate(
            txt,
            xy = (Y[i,0], Y[i,1]), xytext = (0, 0), textcoords = 'offset points', size=5,# ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = color, alpha = 0.5))
        if i in showInfo:
            plt.annotate(
            textwrap.fill(labels[i][1], 20),
            xy = (Y[i,0], Y[i,1]), xytext = (40, 40), textcoords = 'offset points', size=14,# ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = "lightgrey", alpha = 0.9),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'),
            zorder= 1000)
    plt.show()

def plotLanguageModel(sourceDictionary, targetVocabulary, topN=100):
    s = sourceDictionary.embeddings.shape[1]
    print "Embed size: %d" % s
    myVocab = targetVocabulary.vocab.values()
    svocab = sorted(myVocab, reverse=True)
    X = []
    labels = []
    for i in xrange(0, min(len(svocab), topN)):
        word = svocab[i].word
        baseIdx = sourceDictionary.vocab.id(word)
        X.append(sourceDictionary.embeddings[baseIdx].reshape((1,s)))
        labels.append(word)
    X = np.vstack(X)
    X = X.astype(np.float64)
    X *= 1.0 / np.std(X)
    print X.shape
    # print X[17]
    Y = tsne.tsne(X, no_dims=2, initial_dims=30, perplexity=20);
    for i in xrange(len(labels)):
        plt.annotate(
            labels[i],
            xy = (Y[i,0], Y[i,1]), xytext = (0, 0), textcoords = 'offset points', size=20)
    plt.show()

if __name__ == "__main__":

    # Eval word representations

    # import cPickle
    # from os.path import join
    # from general.config import Config, DataEnum, ModelEnum
    #
    # questions = join(ModelEnum.w2v.getPath(), r'questions-words.txt')
    #
    # for dataId in [DataEnum.turian, DataEnum.amazon, DataEnum.wikipedia]:
    #     langDict = cPickle.load(open(join(dataId.getPath(), Config.DIRNAME_DATA_TRAIN, Config.FILE_LANGUAGE_DICTIONARY), "r"))
    #     print dataId
    #     print np.shape(langDict.getEmbeddings())
    #     wanalyzer = WordAccuracy(langDict, questions)
    #     wanalyzer.accuracy(restrict_vocab=50000)

    # Eval phrase representations
    import cPickle
    from scipy.io import savemat
    from sentenceLearning.training.trainer import SentenceRepresentationModel
    from general.config import ModelEnum
    from os.path import join
    langId = "amazon"
    split = 3

    raeModel = SentenceRepresentationModel(dictionary=None, categorySize=2, modelId=ModelEnum.rae, loadExistingModel="_lang="+str(langId)+"_split="+str(split)+"_")
    cnnModel = SentenceRepresentationModel(dictionary=None, categorySize=2, modelId=ModelEnum.cnn, loadExistingModel="_lang="+str(langId)+"_split="+str(split)+"_")
    data = cPickle.load(open(join(ModelEnum.cnn.getPath(), Config.FILE_MODEL_DATA), "r"))
    (trainData, testData) = data.getSplit(split)
    (embs, preds, labs) = raeModel.extract(testData)
    myembs = np.squeeze(np.array(embs).astype(np.float32))
    (cembs, cpreds, clabs) = cnnModel.extract(testData)
    mycembs = np.squeeze(np.array(cembs).astype(np.float32))

    plotEmbeddings(mycembs, [(i,d[0],clabs[i], cpreds[i]==clabs[i]) for i,d in enumerate(testData)])
    # pos = [embs[i] for i,l in enumerate(labs) if l == 1]
    # neg = [embs[i] for i,l in enumerate(labs) if l == 0]
    # raepos = {"raepos": np.squeeze(np.array(pos).astype(np.float32))}
    # raeneg = {"raeneg": np.squeeze(np.array(neg).astype(np.float32))}
    # dembs = {"raeembs": myembs}
    # savemat(r'C:\Users\Johannes\Documents\MATLAB\raeembs.mat', dembs)
    # savemat(r'C:\Users\Johannes\Documents\MATLAB\raepos.mat', raepos)
    # savemat(r'C:\Users\Johannes\Documents\MATLAB\raeneg.mat', raeneg)
    #
    # print len(labs)
    # print sum(labs)
    #
    # myexamples = [(" ".join([testData.vocabulary.key(widx) for widx in d[0]]), d[1]) for d in testData]
    # phr_acc = PhraseAccuracy(myexamples, myembs)
    # for i in xrange(1,100):
    #     r = phr_acc.accuracy(topn=i)
    #     print "%.4f, %.4f;" % (r[0][1], r[1][1])
    #
    # print "nowcnn"
    # (cembs, cpreds, clabs) = cnnModel.extract(testData)
    # mycembs = np.squeeze(np.array(cembs).astype(np.float32))
    # cpos = [cembs[i] for i,l in enumerate(clabs) if l == 1]
    # cneg = [cembs[i] for i,l in enumerate(clabs) if l == 0]
    # cnnpos = {"cnnpos": np.squeeze(np.array(cpos).astype(np.float32))}
    # cnnneg = {"cnnneg": np.squeeze(np.array(cneg).astype(np.float32))}
    # dcembs = {"cnnembs": mycembs}
    # savemat(r'C:\Users\Johannes\Documents\MATLAB\cnnembs.mat', dcembs)
    # savemat(r'C:\Users\Johannes\Documents\MATLAB\cnnpos.mat', cnnpos)
    # savemat(r'C:\Users\Johannes\Documents\MATLAB\cnnneg.mat', cnnneg)
    # mycexamples = [(" ".join([testData.vocabulary.key(widx) for widx in d[0]]), d[1]) for d in testData]
    # cphr_acc = PhraseAccuracy(mycexamples, mycembs)
    # for i in xrange(1,100):
    #     r = cphr_acc.accuracy(topn=i)
    #     print "%.4f, %.4f;" % (r[0][1], r[1][1])

    # e = SignalLogger("precision", "recall")
    # e.add(3.4, 4.2)
    # e.add(2.3, 3.1)
    # print e.signals
    # print e.getMatlabStr("precision")
    # pred = np.array([0,1,0,1,0,0])
    # gold = np.array([1,0,1,0,1,1])
    # Accuracy(pred, gold, 2)

    # def getNearestNeighbors(sentId, myList, k=10):
    #     #queryWord = embeds[wordmap.id(key)]
    #     queryWord = myList[sentId][1]
    #     dists = [np.linalg.norm(queryWord-myList[otherId][1]) for otherId in xrange(len(myList))]
    #     # neighbors = []
    #     # minId = 0
    #     # minDist = 9999999.9
    #     # for b in range(k):
    #     #     for (i, wd) in enumerate(dists):
    #     #         if wd < minDist:
    #     #             minId = i
    #     #             minDist = wd
    #     #     neighbors.append((wm.key(minId), minDist))
    #     idxs = np.argsort(dists)
    #     neighbors = [(myList[x][0], dists[x]) for x in idxs[:k]]
    #     # fh.write(wm.key(idx) + " " + str(neighbors)+'\n')
    #     # fh.write(str(neighbors)+'\n')
    #     return neighbors
    #
    # repr_h = open(join(Config.DIR_OUTPUT, r'rae_repr_3.txt'), 'r')
    # myFancyList = []
    # for l in repr_h:
    #     [sentence, features, unFeatures] = json.loads(l)
    #     feats = np.asarray(features)
    #     unFeats = np.asarray(unFeatures)
    #     myFancyList.append((sentence, feats, unFeats))
    # repr_h.close()
    #
    # for (idx, e) in enumerate(myFancyList):
    #     print "Sentence Id %d" % idx
    #     neighbs = getNearestNeighbors(idx, myFancyList)
    #     for n in neighbs:
    #         print n
