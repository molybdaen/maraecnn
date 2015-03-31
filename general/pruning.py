__author__ = 'Johannes'

import numpy
import cPickle
from sentenceLearning.training.trainer import SentenceRepresentationModel
from os.path import join
from sklearn.linear_model import LogisticRegression
from general.analysis import Accuracy
from general.config import Config, DataEnum, TokenizerEnum, ModelEnum

prunedWDir = r'C:\MA\sentiment\pruning'


def pruneCNNWeights(cnnModel, data, split, mode):
    W1orig = cnnModel.graph.W1.get_value().copy()
    b1orig = cnnModel.graph.b1.get_value().copy()
    Lorig = cnnModel.params.embeddings.copy()

    W1tmp = cnnModel.graph.W1.get_value().copy()
    b1tmp = cnnModel.graph.b1.get_value().copy()
    Ltmp = cnnModel.params.embeddings.copy()

    W1abs = numpy.abs(W1orig)
    b1abs = numpy.abs(b1orig)
    Labs = numpy.abs(Lorig)

    sortedW1 = numpy.sort(W1abs.flatten())
    sortedb1 = numpy.sort(b1abs.flatten())
    sortedL = numpy.sort(Labs.flatten())
    for i in xrange(21):
        if i - 1 == -1:
            dropThreshold = 0.0
        else:
            dropThreshold = 0.8 + 0.01 * (i-1)
        fileName = "cnn_params"+mode+"_pLvl="+str(int(dropThreshold*100))+"_"
        thresholdWeightW1 = sortedW1[int(dropThreshold*len(sortedW1))]
        thresholdWeightb1 = sortedb1[int(dropThreshold*len(sortedb1))]
        thresholdWeightL = sortedL[int(dropThreshold*len(sortedL))]
        # thresholdWeight = numpy.max(sortedW1)
        W1tmp[W1abs < thresholdWeightW1] = 0.0
        b1tmp[b1abs < thresholdWeightb1] = 0.0
        Ltmp[Labs < thresholdWeightL] = 0.0
        # print "Set all values in [-%.5f, +%.5f] to Zero. (%d weights) (%.2f)" % (thresholdWeight, thresholdWeight, int(dropThreshold*len(sortedParams)), dropThreshold)
        print "Set all values in [-%.5f, +%.5f] of W1 to Zero. (Zero=%d weights / Non-Zero=%d weights) (%.1f percent)" % (thresholdWeightW1, thresholdWeightW1, int(dropThreshold*len(sortedW1)), len(W1tmp.nonzero()[0]), dropThreshold*100)
        print "Set all values in [-%.5f, +%.5f] of b1 to Zero. (Zero=%d weights / Non-Zero=%d weights) (%.1f percent)" % (thresholdWeightb1, thresholdWeightb1, int(dropThreshold*len(sortedb1)), len(b1tmp.nonzero()[0]), dropThreshold*100)
        print "Set all values in [-%.5f, +%.5f] of L to Zero. (Zero=%d weights / Non-Zero=%d weights) (%.1f percent)" % (thresholdWeightL, thresholdWeightL, int(dropThreshold*len(sortedL)), len(Ltmp.nonzero()[0]), dropThreshold*100)

        numpy.save(join(prunedWDir, (fileName + "W.npy")), W1tmp)
        numpy.save(join(prunedWDir, (fileName + "b.npy")), b1tmp)
        numpy.save(join(prunedWDir, (fileName + "L.npy")), Ltmp)

        cnnModel.graph.W1.set_value(W1tmp)
        cnnModel.graph.b1.set_value(b1tmp)
        cnnModel.params.embeddings = Ltmp

        data.setBatchMode(1)
        (trainData, testData) = data.getSplit(split)

        print "CLASSIFICATION: Extracting Features..."
        (trainFeatures, predictions, labels) = cnnModel.extractFeatures(trainData)
        (testFeatures, testpredictions, testLabels) = cnnModel.extractFeatures(testData)
        X = numpy.hstack(trainFeatures).T
        y = labels
        classifier = LogisticRegression()
        mod = classifier.fit(X,y)
        preds = mod.predict(numpy.hstack(testFeatures).T)
        accLogReg = Accuracy(preds, testLabels, 2)
        print "CLASSIFICATION LOGREG RESULT: %s" % str(accLogReg)

        W1tmp = W1orig.copy()
        b1tmp = b1orig.copy()
        Ltmp = Lorig.copy()

def pruneRAEWeights(raeModel, data, split):
    paramsOrig = [raeModel.propagation.theta.Wc1.copy(), raeModel.propagation.theta.Wc2.copy(), raeModel.propagation.theta.Wr1.copy(), raeModel.propagation.theta.Wr2.copy(), raeModel.parameters.embeddings.copy()]

    paramsTmp = [raeModel.propagation.theta.Wc1.copy(), raeModel.propagation.theta.Wc2.copy(), raeModel.propagation.theta.Wr1.copy(), raeModel.propagation.theta.Wr2.copy(), raeModel.parameters.embeddings.copy()]
    paramsTmpAbs = [numpy.abs(p) for p in paramsTmp]
    sortedParamsTmpFlat = [numpy.sort(p.flatten()) for p in paramsTmpAbs]

    for i in xrange(20):
        dropThreshold = 0.8 + 0.01 * i
        dropThresholdValues = [sortedP[int(dropThreshold*len(sortedParamsTmpFlat[idx]))] for idx, sortedP in enumerate(sortedParamsTmpFlat)]
        paramsTmp[0][paramsTmpAbs[0] <= dropThresholdValues[0]] = 0.0
        paramsTmp[1][paramsTmpAbs[1] <= dropThresholdValues[1]] = 0.0
        paramsTmp[2][paramsTmpAbs[2] <= dropThresholdValues[2]] = 0.0
        paramsTmp[3][paramsTmpAbs[3] <= dropThresholdValues[3]] = 0.0
        paramsTmp[4][paramsTmpAbs[4] <= dropThresholdValues[4]] = 0.0

        print "Set all values in [-%.5f, +%.5f] of Wc1 to Zero. (Zero=%d weights / Non-Zero=%d weights) (%.1f percent)" % (dropThresholdValues[0], dropThresholdValues[0], int(dropThreshold*len(sortedParamsTmpFlat[0])), len(paramsTmp[0].nonzero()[0]), dropThreshold*100)
        print "Set all values in [-%.5f, +%.5f] of Wc2 to Zero. (Zero=%d weights / Non-Zero=%d weights) (%.1f percent)" % (dropThresholdValues[1], dropThresholdValues[1], int(dropThreshold*len(sortedParamsTmpFlat[1])), len(paramsTmp[1].nonzero()[0]), dropThreshold*100)
        print "Set all values in [-%.5f, +%.5f] of Wr1 to Zero. (Zero=%d weights / Non-Zero=%d weights) (%.1f percent)" % (dropThresholdValues[2], dropThresholdValues[2], int(dropThreshold*len(sortedParamsTmpFlat[2])), len(paramsTmp[2].nonzero()[0]), dropThreshold*100)
        print "Set all values in [-%.5f, +%.5f] of Wr2 to Zero. (Zero=%d weights / Non-Zero=%d weights) (%.1f percent)" % (dropThresholdValues[3], dropThresholdValues[3], int(dropThreshold*len(sortedParamsTmpFlat[3])), len(paramsTmp[3].nonzero()[0]), dropThreshold*100)
        print "Set all values in [-%.5f, +%.5f] of L to Zero. (Zero=%d weights / Non-Zero=%d weights) (%.1f percent)" % (dropThresholdValues[4], dropThresholdValues[4], int(dropThreshold*len(sortedParamsTmpFlat[4])), len(paramsTmp[4].nonzero()[0]), dropThreshold*100)

        raeModel.propagation.theta.Wc1 = paramsTmp[0]
        raeModel.propagation.theta.Wc2 = paramsTmp[1]
        raeModel.propagation.theta.Wr1 = paramsTmp[2]
        raeModel.propagation.theta.Wr2 = paramsTmp[3]
        raeModel.parameters.embeddings = paramsTmp[4]

        data.setBatchMode(1)
        (trainData, testData) = data.getSplit(split)

        print "CLASSIFICATION: Extracting Features..."
        (trainFeatures, predictions, labels) = raeModel.extractFeatures(trainData)
        (testFeatures, testpredictions, testLabels) = raeModel.extractFeatures(testData)
        X = numpy.hstack(trainFeatures).T
        y = labels
        classifier = LogisticRegression()
        mod = classifier.fit(X,y)
        preds = mod.predict(numpy.hstack(testFeatures).T)
        accLogReg = Accuracy(preds, testLabels, 2)
        print "CLASSIFICATION LOGREG RESULT: %s" % str(accLogReg)

        paramsTmp = [p.copy() for p in paramsOrig]

def getSentenceRepresentations(cnnModel, data, split, reps=100, classIdx=0):
    data.setBatchMode(1)
    c = 0
    (trainData, testData) = data.getSplit(split)
    (testFeatures, testpredictions, testLabels) = cnnModel.extractFeatures(testData)
    for (idx, feats) in enumerate(testFeatures):
        if testLabels[idx] == classIdx and c < reps:
            print ", ".join([str(x) for x in feats.flatten()])
            c += 1




if __name__ == "__main__":
    split = 1
    modelMode = "_lang=random_split="+str(split)+"_"
    # cnnModel = SentenceRepresentationModel(dictionary=None, categorySize=2, modelId=ModelEnum.cnn, loadExistingModel=modelMode)
    # data = cPickle.load(open(join(ModelEnum.cnn.getPath(), Config.FILE_MODEL_DATA), "r"))
    # pruneCNNWeights(cnnModel.model, data, split)
    # pruneCNNWeights(cnnModel.model, data, split, modelMode)
    # getSentenceRepresentations(cnnModel.model, data, 0, 1000, 1)
    W = numpy.load(join(prunedWDir, "cnn_params_lang=random_split=1__pLvl=97_W.npy"))
    print W[0]
    print numpy.shape(W)




