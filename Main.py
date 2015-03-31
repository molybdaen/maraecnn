__author__ = 'Molybdaen'

from general.dataprocessor import PreProcessor
from general.streams import DataSet, ExampleStream
from general.vocabulary import Vocabulary, Dictionary
from general.config import Config, DataEnum, TokenizerEnum, ModelEnum
from general.analysis import Accuracy, plotEmbeddings, SignalLogger
import cPickle
from sentenceLearning.training.trainer import SentenceRepresentationModel
from os.path import join
import numpy as np
from sklearn.linear_model import LogisticRegression
import logging
from util.utils import translateToSentence
from general.analysis import plotLanguageModel

if __name__ == "__main__":

    def sigmoid(x):
        return 1. / (1 + np.exp(-x))

    def normalize(train, test):
        all = np.hstack((train, test))
        mmax = np.max(all)
        mmin = np.min(all)

        ntrain = (train - mmin) / (mmax - mmin)
        ntest = (test - mmin) / (mmax - mmin)
        return ntrain, ntest


    def showSentenceRepresentations():
        dataStream = ExampleStream(DataEnum.moviereviews)
        v = Vocabulary(minCount=2)
        v.readFromDataStream(dataStream)
        langId = "wikipedia"
        split = 1
        data = cPickle.load(open(join(ModelEnum.cnn.getPath(), Config.FILE_MODEL_DATA), "r"))
        prefix = "_lang="+str(langId)+"_split="+str(split)+"_"
        (trainData, testData) = data.getSplit(split)
        cnnModel = SentenceRepresentationModel(dictionary=None, categorySize=trainData.getCategorySize(), modelId=ModelEnum.rae, loadExistingModel=prefix)
        logging.info("CLASSIFICATION: Extracting Features...")
        (testFeatures, testpredictions, testLabels) = cnnModel.extract(testData)
        X = np.array(testFeatures)[0:100,:,0]
        print X.shape
        labels = []
        for (i, e) in enumerate(testData):
            if i < 100:
                if e[1] == 0.0:
                    labels.append((str(i), translateToSentence(e[0], v), 0))
                else:
                    labels.append((str(i), translateToSentence(e[0], v), 1))
                print i
                print translateToSentence(e[0], v)
                print e[1]
        plotEmbeddings(X, labels, showInfo=[])

    def classify(data, model, langId, split):
        logging.basicConfig(filename=join(ModelEnum.rae.getPath(), "logging_classification.log"), filemode='w', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        data.setBatchMode(1)
        prefix = "_lang="+str(langId)+"_split="+str(split)+"_"
        (trainData, testData) = data.getSplit(split)
        # model = SentenceRepresentationModel(dictionary=None, categorySize=trainData.getCategorySize(), modelId=ModelEnum.cnn, loadExistingModel=prefix)
        logging.info("CLASSIFICATION: Extracting Features...")
        (trainFeatures, predictions, labels) = model.extract(trainData)
        (testFeatures, testpredictions, testLabels) = model.extract(testData)

        # trainL = []
        # testL = []
        # for (i, e) in enumerate(trainFeatures):
        #     if labels[i] == 1:
        #         trainL.append(np.asarray([[0], [1]]))
        #     else:
        #         trainL.append(np.asarray([[1], [0]]))
        # trainSs = np.hstack(trainFeatures)
        # trainLs = np.hstack(trainL)
        # for (i, e) in enumerate(testFeatures):
        #     if testLabels[i] == 1:
        #         testL.append(np.asarray([[0], [1]]))
        #     else:
        #         testL.append(np.asarray([[1], [0]]))
        # testSs = np.hstack(testFeatures)
        # testLs = np.hstack(testL)
        #
        # (nTrainSs, nTestSs) = (trainSs, testSs)
        # rng = np.random.RandomState(123)
        # # construct SdA
        # sda = SdA(input=nTrainSs, label=trainLs, n_ins=100, hidden_layer_sizes=[130, 50], n_outs=2, corruption_levels=[0.1, 0.01], numpy_rng=rng)
        # # pre-training orig 150/400 layers 80/20 noise 0.2/0.2
        # sda.pretrain(lr=0.1, epochs=30)
        # # fine-tuning orig lr=0.5 L2_reg=0.5
        # sda.finetune(lr=0.5, L2_reg=0.0, epochs=100)
        # # test
        # predLs = sda.predict(nTestSs)
        #
        # predLIdx = (np.argmax(predLs, axis=0))
        # testLIdx = (np.argmax(testLs, axis=0))
        # accDeep = Accuracy(predLIdx, testLIdx, 2)
        # logging.info("CLASSIFICATION DEEP RESULT: %s" % str(accDeep))
        X = np.hstack(trainFeatures).T
        y = labels
        classifier = LogisticRegression()
        mod = classifier.fit(X,y)
        preds = mod.predict(np.hstack(testFeatures).T)
        accLogReg = Accuracy(preds, testLabels, 2)
        logging.info("CLASSIFICATION LOGREG RESULT: %s" % str(accLogReg))

    def classifySentence(sentence, model):
        wordSequence = sentence.split()
        (rep, pred, labpred) = model.model.extractFeatures_single(wordSequence)
        return (rep, pred, labpred)


    # langId = "amazon"
    # split = 3

    # raeModel = SentenceRepresentationModel(dictionary=None, categorySize=2, modelId=ModelEnum.rae, loadExistingModel="_lang="+str(langId)+"_split="+str(split)+"_")
    # cnnModel = SentenceRepresentationModel(dictionary=None, categorySize=2, modelId=ModelEnum.cnn, loadExistingModel="_lang="+str(langId)+"_split="+str(split)+"_")


    folds = 10

    pp = PreProcessor(preprocessor=DataEnum.moviereviews, tokenizer=TokenizerEnum.sentiment, caseSensitive=False, maxTokens=1000000)
    pp.run()
    pp.shuffle()

    # Build Vocabulary
    dataStream = ExampleStream(DataEnum.moviereviews)
    v = Vocabulary(minCount=2)
    v.readFromDataStream(dataStream)
    print v

    # langDict = cPickle.load(open(join(DataEnum.amazon.getPath(), Config.DIRNAME_DATA_TRAIN, Config.FILE_LANGUAGE_DICTIONARY), "r"))
    # dictionary = Dictionary(v)
    # dictionary.initFrom(langDict)
    # plotLanguageModel(langDict, v, 500)



    # Build Dataset
    data = DataSet(dataId=DataEnum.moviereviews, vocabulary=v, numFolds=folds, minExampleLength=2)
    # cPickle.dump(data, open(join(ModelEnum.cnn.getPath(), Config.FILE_MODEL_DATA), "w"))
    # data = cPickle.load(open(join(ModelEnum.cnn.getPath(), Config.FILE_MODEL_DATA), "r"))

    dictionary = None
    langId = ""
    for i in xrange(10):
        for langIdx in xrange(4):
            if langIdx == 0:
                ########################################
                # Random init
                # Build Dictionary with embeddings
                data.setBatchMode(20)
                dictionary = Dictionary(v)
                langId = "random"
            if langIdx == 1:
                ########################################
                # Wikipedia init
                # Build Dictionary with embeddings
                data.setBatchMode(20)
                langDict = cPickle.load(open(join(DataEnum.wikipedia.getPath(), Config.DIRNAME_DATA_TRAIN, Config.FILE_LANGUAGE_DICTIONARY), "r"))
                dictionary = Dictionary(v)
                dictionary.initFrom(langDict)
                langId = "wikipedia"
            if langIdx == 2:
                ########################################
                # Amazon init
                # Build Dictionary with embeddings
                data.setBatchMode(20)
                langDict = cPickle.load(open(join(DataEnum.amazon.getPath(), Config.DIRNAME_DATA_TRAIN, Config.FILE_LANGUAGE_DICTIONARY), "r"))
                dictionary = Dictionary(v)
                dictionary.initFrom(langDict)
                langId = "amazon"
            if langIdx == 3:
                ########################################
                # Turian C&W init
                # Build Dictionary with embeddings
                data.setBatchMode(10)
                langDict = cPickle.load(open(join(DataEnum.turian.getPath(), Config.DIRNAME_DATA_TRAIN, Config.FILE_LANGUAGE_DICTIONARY), "r"))
                dictionary = Dictionary(v)
                dictionary.initFrom(langDict)
                langId = "turian"

            prefix = "_lang="+str(langId)+"_split="+str(i)+"_"
            (trainData, testData) = data.getSplit(i)

            cnnModel = SentenceRepresentationModel(dictionary=dictionary, categorySize=trainData.getCategorySize(), modelId=ModelEnum.cnn, epochs=15, BFGS_iter=30)
            (errSignals, model) = cnnModel.train(trainData, prefix)
            cnnModel.save(prefix)

            classify(data, cnnModel, langId, i)
            data.setBatchMode(1)